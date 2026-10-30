from types import SimpleNamespace

import numpy as np

from counterworld.models import RidgeEnsemble
from counterworld.noise import NoiseRegistry
from counterworld.planner import RandomShootingMPC


class TinyEnv:
    goal_radius = 0.1

    def __init__(self, max_steps=3):
        self.max_steps = max_steps

    def environment_factor(self, checkpoint):
        return np.eye(4) * checkpoint.noise_scale

    def reward_many(
        self, states, next_states, checkpoint, collision=False, collision_points=None
    ):
        states = np.asarray(states)
        next_states = np.asarray(next_states)
        old_distance = np.linalg.norm(states[..., :2] - checkpoint.goal, axis=-1)
        new_distance = np.linalg.norm(next_states[..., :2] - checkpoint.goal, axis=-1)
        goal = new_distance <= checkpoint.goal_radius
        reward = -0.01 + old_distance - new_distance + goal.astype(float)
        return np.where(collision, -4.0, reward)

    def transition_many(
        self, states, actions, canonical_noises, factor, checkpoint, include_wall=True
    ):
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        noises = np.asarray(canonical_noises, dtype=float)
        next_states = states.copy()
        next_states[..., :2] = states[..., :2] + states[..., 2:] + 0.5 * actions
        next_states[..., 2:] = states[..., 2:] + actions
        next_states += noises @ np.asarray(factor).T
        collision = np.zeros(states.shape[:-1], dtype=bool)
        if include_wall:
            collision = (
                (states[..., 0] < 0.0)
                & (next_states[..., 0] >= 0.0)
                & (np.abs(next_states[..., 1] - checkpoint.gap_center) > checkpoint.gap_width / 2)
            )
            next_states = next_states.copy()
            next_states[collision, 0] = 0.0
            next_states[collision, 2:] = 0.0
        reward = self.reward_many(states, next_states, checkpoint, collision=collision)
        goal = (
            np.linalg.norm(next_states[..., :2] - checkpoint.goal, axis=-1)
            <= checkpoint.goal_radius
        ) & ~collision
        terminal = collision | goal
        return {
            "next_state": next_states,
            "reward": reward,
            "terminal": terminal,
            "collision": collision,
            "goal": goal,
        }


def registry_normal(registry, stream, indices, shape):
    try:
        return registry.normal(stream, indices, shape=shape)
    except TypeError:
        return registry.normal(stream, *indices, shape=shape)


class RecordingRegistry:
    def __init__(self, seed):
        self.registry = NoiseRegistry(seed)
        self.calls = []

    def normal(self, stream, indices, shape):
        value = registry_normal(self.registry, stream, indices, shape)
        self.calls.append((stream, tuple(indices), shape, value.copy()))
        return value


class FixedCandidatePlanner(RandomShootingMPC):
    def __init__(self, env, model, candidates, **kwargs):
        candidates = np.asarray(candidates, dtype=float)
        super().__init__(
            env,
            model,
            n_candidates=len(candidates),
            horizon=candidates.shape[1],
            **kwargs,
        )
        self.fixed_candidates = candidates

    def candidate_actions(self, noise_registry, checkpoint_id, episode_id, time_step):
        return self.fixed_candidates.copy()


def make_checkpoint(state=(-1.0, 0.0, 0.0, 0.0), noise_scale=0.0):
    return SimpleNamespace(
        state=np.asarray(state, dtype=float),
        goal=np.array([1.5, 0.0]),
        goal_radius=0.1,
        gap_center=1.0,
        gap_width=0.2,
        noise_scale=noise_scale,
    )


def make_model(members=1, exact=True):
    coefficients = np.zeros((members, 7, 4))
    if exact:
        coefficients[:, 0, 0] = 1.0
        coefficients[:, 1, 1] = 1.0
        coefficients[:, 2, 0] = 1.0
        coefficients[:, 2, 2] = 1.0
        coefficients[:, 3, 1] = 1.0
        coefficients[:, 3, 3] = 1.0
        coefficients[:, 4, 0] = 0.5
        coefficients[:, 4, 2] = 1.0
        coefficients[:, 5, 1] = 0.5
        coefficients[:, 5, 3] = 1.0
    return RidgeEnsemble(
        coefficients,
        np.zeros((members, 4, 4)),
        np.zeros((members, 4, 4)),
        np.zeros((members, 1), dtype=int),
    )


def assert_records_equal(left, right):
    np.testing.assert_array_equal(left.states, right.states)
    np.testing.assert_array_equal(left.actions, right.actions)
    np.testing.assert_array_equal(left.rewards, right.rewards)
    np.testing.assert_array_equal(left.selected_indices, right.selected_indices)
    assert left.total_return == right.total_return
    assert left.collision == right.collision
    assert left.goal == right.goal
    assert left.real_environment_steps == right.real_environment_steps
    assert left.model_calls == right.model_calls


def test_candidates_are_reproducible_and_independent_of_model_parameters():
    registry = NoiseRegistry(91)
    first = RandomShootingMPC(TinyEnv(), make_model(exact=True), n_candidates=20, horizon=4)
    second = RandomShootingMPC(TinyEnv(), make_model(exact=False), n_candidates=20, horizon=4)
    candidates = first.candidate_actions(registry, 2, 3, 4)
    np.testing.assert_array_equal(
        candidates, first.candidate_actions(NoiseRegistry(91), 2, 3, 4)
    )
    np.testing.assert_array_equal(
        candidates, second.candidate_actions(NoiseRegistry(91), 2, 3, 4)
    )
    assert not np.array_equal(candidates, first.candidate_actions(registry, 2, 3, 5))


def test_ties_choose_lowest_index_and_controllers_share_candidates_and_noise():
    env = TinyEnv()
    registry = RecordingRegistry(8)
    planner = RandomShootingMPC(
        env, make_model(members=2, exact=False), n_candidates=6, horizon=2, selection_replicates=2
    )
    cp = make_checkpoint()
    action, index, diagnostics = planner.select_action(
        cp.state, cp, registry, 1, 2, 0, controller="model"
    )
    assert index == 0
    np.testing.assert_array_equal(action, diagnostics["candidates"][0, 0])
    np.testing.assert_allclose(
        diagnostics["candidate_returns"], diagnostics["candidate_returns"][0], atol=1e-12
    )

    exact_planner = RandomShootingMPC(
        env, make_model(members=2), n_candidates=6, horizon=2, selection_replicates=2
    )
    _, _, model_diagnostics = exact_planner.select_action(
        cp.state, cp, registry, 1, 2, 0, controller="model"
    )
    _, _, reference_diagnostics = exact_planner.select_action(
        cp.state, cp, registry, 1, 2, 0, controller="reference"
    )
    np.testing.assert_array_equal(
        model_diagnostics["candidates"], reference_diagnostics["candidates"]
    )
    selection_values = [call[3] for call in registry.calls if call[0] == "planner-selection"]
    np.testing.assert_array_equal(selection_values[-2], selection_values[-1])


def test_wall_collision_produces_csr_and_wall_free_model_overestimate():
    env = TinyEnv(max_steps=1)
    cp = make_checkpoint(state=(-0.25, 0.0, 0.0, 0.0))
    candidates = np.array([[[1.0, 0.0]], [[-1.0, 0.0]]])
    planner = FixedCandidatePlanner(
        env, make_model(), candidates, gamma=0.99, selection_replicates=1
    )
    registry = NoiseRegistry(17)
    _, model_index, _ = planner.select_action(
        cp.state, cp, registry, 0, 0, 0, controller="model"
    )
    _, reference_index, _ = planner.select_action(
        cp.state, cp, registry, 0, 0, 0, controller="reference"
    )
    assert model_index == 0
    assert reference_index == 1

    model_environment = planner.run_episode(
        cp, "model", "environment", registry, "csr", 0, 0
    )
    reference_environment = planner.run_episode(
        cp, "reference", "environment", registry, "csr", 0, 0
    )
    model_model = planner.run_episode(cp, "model", "model", registry, "peg", 0, 0)
    assert model_environment.collision
    assert not reference_environment.collision
    assert not model_model.collision
    assert model_environment.total_return == -4.0
    assert model_model.total_return > model_environment.total_return
    assert reference_environment.total_return - model_environment.total_return > 0.0


def test_selection_noise_is_separate_from_indexed_execution_noise():
    registry = RecordingRegistry(23)
    cp = make_checkpoint(noise_scale=0.02)
    planner = FixedCandidatePlanner(
        TinyEnv(max_steps=1), make_model(), np.zeros((1, 1, 2)), selection_replicates=1
    )
    planner.run_episode(cp, "model", "environment", registry, "metric", 4, 7)
    streams = [call[0] for call in registry.calls]
    assert "planner-selection" in streams
    assert "metric" in streams
    actual = next(call for call in registry.calls if call[0] == "metric")
    assert actual[1] == (4, 7, 0)
    np.testing.assert_array_equal(
        actual[3], registry_normal(registry.registry, "metric", (4, 7, 0), (4,))
    )


def test_episode_record_accounting_and_nested_episode_prefixes():
    env = TinyEnv(max_steps=3)
    cp = make_checkpoint()
    planner = FixedCandidatePlanner(
        env, make_model(), np.zeros((1, 1, 2)), selection_replicates=1
    )
    registry = NoiseRegistry(31)
    model_environment = planner.run_episode(
        cp, "model", "environment", registry, "outcome", 3, 0
    )
    assert model_environment.states.shape == (4, 4)
    assert model_environment.actions.shape == (3, 2)
    assert model_environment.rewards.shape == (3,)
    assert model_environment.selected_indices.shape == (3,)
    assert model_environment.real_environment_steps == 3
    assert model_environment.model_calls == 3
    np.testing.assert_allclose(
        model_environment.total_return,
        np.sum(model_environment.rewards * 0.99 ** np.arange(3)),
    )

    reference_environment = planner.run_episode(
        cp, "reference", "environment", registry, "outcome", 3, 0
    )
    assert reference_environment.real_environment_steps == 3
    assert reference_environment.model_calls == 0
    model_model = planner.run_episode(cp, "model", "model", registry, "outcome", 3, 0)
    assert model_model.real_environment_steps == 0
    assert model_model.model_calls == 6

    four = planner.run_episodes(cp, "model", "environment", registry, "nested", 3, 4)
    two = planner.run_episodes(cp, "model", "environment", registry, "nested", 3, 2)
    assert len(four) == 4
    for first, second in zip(two, four):
        assert_records_equal(first, second)


def test_batched_episodes_match_scalar_episode_records():
    candidates = np.array([
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 1.0], [1.0, 0.0]],
        [[-1.0, 0.0], [0.0, 0.0]],
    ])
    planner = FixedCandidatePlanner(TinyEnv(), make_model(members=2), candidates)
    checkpoint = make_checkpoint()
    registry = NoiseRegistry(41)
    for controller in ("model", "reference"):
        for execution in ("environment", "model"):
            expected = [
                planner.run_episode(
                    checkpoint,
                    controller,
                    execution,
                    registry,
                    "metric-evaluation",
                    4,
                    episode_id,
                )
                for episode_id in range(3)
            ]
            actual = planner.run_episodes(
                checkpoint,
                controller,
                execution,
                registry,
                "metric-evaluation",
                4,
                3,
                batched=True,
            )
            for left, right in zip(expected, actual):
                assert_records_equal(left, right)
