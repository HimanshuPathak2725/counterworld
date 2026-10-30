from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counterworld.environment import (
    ACTION_GRID,
    ACTIONS,
    Checkpoint,
    DoubleIntegratorGapEnv,
)
from counterworld.noise import covariance_factor


ZERO_NOISE = np.zeros(4)
ZERO_FACTOR = np.zeros((4, 4))


def make_checkpoint(state, gap_center=0.0, **kwargs) -> Checkpoint:
    return Checkpoint(
        state=np.asarray(state, dtype=float),
        gap_center=gap_center,
        regime="test",
        checkpoint_id="checkpoint",
        **kwargs,
    )


def test_action_grid_contains_nine_discrete_actions() -> None:
    expected = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=2)))
    np.testing.assert_array_equal(ACTION_GRID, expected)
    np.testing.assert_array_equal(ACTIONS, expected)


def test_free_dynamics_use_explicit_action_and_noise() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint(
        [-2.0, 0.4, 0.25, -0.3],
        sigma_p=0.02,
        sigma_v=0.04,
    )
    action = np.asarray([1.0, -1.0])
    canonical_noise = np.asarray([1.0, -2.0, 0.5, -1.0])
    factor = env.environment_factor(checkpoint)

    transition = env.step(
        checkpoint.state,
        action,
        canonical_noise,
        factor,
        checkpoint,
        include_wall=False,
    )

    noise = factor @ canonical_noise
    expected_position = checkpoint.state[:2] + checkpoint.state[2:] + 0.5 * action + noise[:2]
    expected_velocity = checkpoint.state[2:] + action + noise[2:]
    np.testing.assert_allclose(transition.next_state[:2], expected_position)
    np.testing.assert_allclose(transition.next_state[2:], expected_velocity)
    expected_reward = (
        -0.01
        + np.linalg.norm(checkpoint.state[:2] - checkpoint.goal)
        - np.linalg.norm(expected_position - checkpoint.goal)
    )
    assert transition.reward == pytest.approx(expected_reward)
    assert not transition.terminal
    assert not transition.collision
    assert not transition.goal


def test_crossing_through_gap_is_safe() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([-0.5, 0.0, 1.0, 0.0])
    transition = env.step(
        checkpoint.state,
        np.zeros(2),
        ZERO_NOISE,
        ZERO_FACTOR,
        checkpoint,
    )
    np.testing.assert_allclose(transition.next_state, [0.5, 0.0, 1.0, 0.0])
    assert not transition.collision
    assert not transition.terminal


def test_solid_collision_clamps_to_intersection_and_can_disable_wall() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([-0.5, 1.0, 1.0, 0.0])
    transition = env.step(
        checkpoint.state,
        np.zeros(2),
        ZERO_NOISE,
        ZERO_FACTOR,
        checkpoint,
    )
    np.testing.assert_allclose(transition.next_state, [0.0, 1.0, 0.0, 0.0])
    assert transition.reward == -4.0
    assert transition.terminal
    assert transition.collision
    assert not transition.goal

    without_wall = env.step(
        checkpoint.state,
        np.zeros(2),
        ZERO_NOISE,
        ZERO_FACTOR,
        checkpoint,
        include_wall=False,
    )
    np.testing.assert_allclose(without_wall.next_state, [0.5, 1.0, 1.0, 0.0])
    assert not without_wall.collision
    assert not without_wall.terminal


def test_continuous_intersection_prevents_tunneling() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([-2.0, 1.0, 4.0, 0.0])
    transition = env.step(
        checkpoint.state,
        np.zeros(2),
        ZERO_NOISE,
        ZERO_FACTOR,
        checkpoint,
    )
    assert transition.collision
    np.testing.assert_allclose(transition.next_state, [0.0, 1.0, 0.0, 0.0])


def test_initial_wall_contact_and_collinear_motion_are_robust() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([0.0, 0.0, 0.0, 0.0])
    states = np.asarray(
        [
            [0.0, 1.0, 1.0, 0.0],
            [-2.0, 0.5, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    next_states, _, terminal, collision, _ = env.transition_many(
        states,
        np.zeros((3, 2)),
        np.zeros((3, 4)),
        ZERO_FACTOR,
        checkpoint,
    )
    np.testing.assert_array_equal(collision, [True, False, True])
    np.testing.assert_array_equal(terminal, [True, False, True])
    np.testing.assert_allclose(next_states[0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(next_states[1], [-2.0, 1.5, 0.0, 1.0])
    np.testing.assert_allclose(next_states[2], [0.0, 0.3, 0.0, 0.0])


def test_goal_adds_bonus_and_terminates() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([1.3, 0.0, 0.2, 0.0])
    transition = env.step(
        checkpoint.state,
        np.zeros(2),
        ZERO_NOISE,
        ZERO_FACTOR,
        checkpoint,
    )
    np.testing.assert_allclose(transition.next_state[:2], [1.5, 0.0])
    assert transition.reward == pytest.approx(1.19)
    assert transition.terminal
    assert transition.goal
    assert not transition.collision


def test_vectorized_and_scalar_transitions_match_with_batched_factors() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([-1.0, 0.0, 0.0, 0.0])
    states = np.asarray(
        [
            [-0.5, 0.0, 1.0, 0.0],
            [-0.5, 1.0, 1.0, 0.0],
            [1.3, 0.0, 0.2, 0.0],
            [-2.0, 0.5, 0.25, -0.25],
        ]
    )
    actions = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, -1.0]])
    noises = np.asarray(
        [
            [0.1, -0.2, 0.3, 0.1],
            [-0.2, 0.1, 0.4, -0.3],
            [0.0, 0.1, -0.2, 0.2],
            [0.5, -0.4, 0.1, 0.3],
        ]
    )
    base_factor = np.diag([0.01, 0.02, 0.03, 0.04])
    factors = np.stack([base_factor, 2.0 * base_factor, 0.5 * base_factor, base_factor])

    vectorized = env.transition_many(states, actions, noises, factors, checkpoint)
    for index in range(states.shape[0]):
        scalar = env.step(states[index], actions[index], noises[index], factors[index], checkpoint)
        np.testing.assert_allclose(vectorized[0][index], scalar.next_state)
        assert vectorized[1][index] == pytest.approx(scalar.reward)
        assert vectorized[2][index] == scalar.terminal
        assert vectorized[3][index] == scalar.collision
        assert vectorized[4][index] == scalar.goal

    rewards, terminal, goal = env.reward_many(
        states,
        vectorized[0],
        checkpoint,
        collision=vectorized[3],
        collision_points=vectorized[0][:, :2],
    )
    np.testing.assert_allclose(rewards, vectorized[1])
    np.testing.assert_array_equal(terminal, vectorized[2])
    np.testing.assert_array_equal(goal, vectorized[4])


def test_environment_and_model_map_same_canonical_noise_with_different_factors() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint(
        [-2.0, 0.2, 0.1, -0.1],
        sigma_p=0.02,
        sigma_v=0.04,
    )
    action = np.asarray([0.0, 1.0])
    canonical_noise = np.asarray([0.3, -0.5, 0.7, 0.2])
    environment_factor = env.environment_factor(checkpoint)
    model_covariance = np.asarray(
        [
            [0.04, 0.01, 0.0, 0.0],
            [0.01, 0.09, 0.0, 0.0],
            [0.0, 0.0, 0.16, 0.02],
            [0.0, 0.0, 0.02, 0.25],
        ]
    )
    model_factor = covariance_factor(model_covariance)
    assert not np.allclose(environment_factor, model_factor)
    assert not np.allclose(environment_factor @ canonical_noise, model_factor @ canonical_noise)

    results = []
    for factor in (environment_factor, model_factor):
        transition = env.step(
            checkpoint.state,
            action,
            canonical_noise,
            factor,
            checkpoint,
            include_wall=False,
        )
        noise = factor @ canonical_noise
        expected = checkpoint.state.copy()
        expected[:2] += checkpoint.state[2:] + 0.5 * action + noise[:2]
        expected[2:] += action + noise[2:]
        np.testing.assert_allclose(transition.next_state, expected)
        results.append(transition.next_state)
    assert not np.allclose(results[0], results[1])


def test_geometric_detour_returns_valid_zero_noise_actions_when_found() -> None:
    env = DoubleIntegratorGapEnv()
    checkpoint = make_checkpoint([-1.5, 0.0, 0.0, 0.0])
    actions = env.geometric_detour_actions(checkpoint)
    assert actions is not None
    assert actions.shape[1] == 2
    assert actions.shape[0] <= 12

    state = env.reset(checkpoint)
    final = None
    for action in actions:
        assert np.any(np.all(action == ACTION_GRID, axis=1))
        final = env.step(state, action, ZERO_NOISE, ZERO_FACTOR, checkpoint)
        assert not final.collision
        state = final.next_state
        if final.terminal:
            break
    assert final is not None and final.goal
    assert env.geometric_detour_actions(checkpoint, horizon=0) is None
