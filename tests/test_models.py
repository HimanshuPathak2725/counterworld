from types import SimpleNamespace

import numpy as np
import pytest

from counterworld.models import RidgeEnsemble, TransitionDataset


class RewardEnv:
    goal_radius = 0.1

    def reward_many(
        self, states, next_states, checkpoint, collision=False, collision_points=None
    ):
        states = np.asarray(states)
        next_states = np.asarray(next_states)
        old_distance = np.linalg.norm(states[..., :2] - checkpoint.goal, axis=-1)
        new_distance = np.linalg.norm(next_states[..., :2] - checkpoint.goal, axis=-1)
        goal = new_distance <= checkpoint.goal_radius
        return -0.01 + old_distance - new_distance + goal.astype(float)


def checkpoint(state=(-1.0, 0.0, 0.0, 0.0)):
    return SimpleNamespace(
        state=np.asarray(state, dtype=float),
        goal=np.array([1.5, 0.0]),
        goal_radius=0.1,
    )


def exact_coefficients():
    coefficients = np.zeros((7, 4))
    coefficients[0, 0] = 1.0
    coefficients[1, 1] = 1.0
    coefficients[2, 0] = 1.0
    coefficients[2, 2] = 1.0
    coefficients[3, 1] = 1.0
    coefficients[3, 3] = 1.0
    coefficients[4, 0] = 0.5
    coefficients[4, 2] = 1.0
    coefficients[5, 1] = 0.5
    coefficients[5, 3] = 1.0
    return coefficients


def exact_dataset(noisy_holdout=False):
    rng = np.random.default_rng(19)
    states, actions, next_states, episode_ids = [], [], [], []
    for episode_id in range(6):
        episode_states = rng.normal(size=(10, 4))
        episode_actions = rng.normal(size=(10, 2))
        states.append(episode_states)
        actions.append(episode_actions)
        features = np.column_stack(
            (episode_states, episode_actions, np.ones(len(episode_states)))
        )
        targets = features @ exact_coefficients()
        if noisy_holdout and episode_id >= 4:
            shared = rng.normal(size=(10, 1))
            targets = targets + shared * np.array([1.0, -1.0, 2.0, 0.5])
        next_states.append(targets)
        episode_ids.append(np.full(10, episode_id))
    return TransitionDataset(
        np.concatenate(states),
        np.concatenate(actions),
        np.concatenate(next_states),
        np.concatenate(episode_ids),
    )


def test_deterministic_ridge_recovers_dynamics_with_unpenalized_intercept():
    dataset = exact_dataset()
    model = RidgeEnsemble.fit(
        dataset, [0, 1, 2, 3], [4, 5], n_members=1, seed=4, ridge=0.0
    )
    np.testing.assert_allclose(model.coefficients[0], exact_coefficients(), atol=1e-10)
    states = np.array([[0.2, -0.3, 0.4, 0.5]])
    actions = np.array([[0.6, -0.2]])
    expected = np.column_stack((states, actions, np.ones(1))) @ exact_coefficients()
    np.testing.assert_allclose(model.predict_next(states, actions), expected[None, :, :])
    np.testing.assert_allclose(model.predict_next(states, actions, member_index=0), expected)


def test_bootstrap_resamples_whole_episodes():
    dataset = exact_dataset()
    train_ids = np.array([0, 1, 2, 3])
    ridge = 0.2
    model = RidgeEnsemble.fit(
        dataset, train_ids, [4, 5], n_members=5, seed=11, ridge=ridge
    )
    penalty = np.eye(7) * ridge
    penalty[-1, -1] = 0.0
    assert model.bootstrap_episode_ids.shape == (5, 4)
    for member, sampled_ids in enumerate(model.bootstrap_episode_ids):
        assert np.all(np.isin(sampled_ids, train_ids))
        rows = np.concatenate(
            [np.flatnonzero(dataset.episode_ids == episode_id) for episode_id in sampled_ids]
        )
        features = np.column_stack(
            (dataset.states[rows], dataset.actions[rows], np.ones(len(rows)))
        )
        expected = np.linalg.solve(
            features.T @ features + penalty, features.T @ dataset.next_states[rows]
        )
        np.testing.assert_allclose(model.coefficients[member], expected)


def test_canonical_noise_uses_frozen_holdout_covariance_factor():
    dataset = exact_dataset(noisy_holdout=True)
    model = RidgeEnsemble.fit(
        dataset, [0, 1, 2, 3], [4, 5], n_members=1, seed=2, ridge=0.0
    )
    covariance = model.covariances[0]
    factor = model.covariance_factors[0]
    np.testing.assert_allclose(factor @ factor.T, covariance, atol=1e-8)
    states = np.array([[0.0, 0.2, 0.3, -0.1]])
    actions = np.array([[0.4, 0.7]])
    noise = np.array([[0.3, -0.8, 0.2, 1.1]])
    deterministic = model.predict_next(states, actions, member_index=0)
    stochastic = model.predict_next(states, actions, noise, member_index=0)
    np.testing.assert_allclose(stochastic - deterministic, noise @ factor.T)
    assert np.linalg.norm(stochastic - deterministic) > 0.1


def test_rollout_returns_stop_per_member_after_goal():
    coefficients = np.zeros((2, 7, 4))
    coefficients[:, 0, 0] = 1.0
    coefficients[0, 4, 0] = 0.1
    model = RidgeEnsemble(
        coefficients=coefficients,
        covariances=np.zeros((2, 4, 4)),
        covariance_factors=np.zeros((2, 4, 4)),
        bootstrap_episode_ids=np.zeros((2, 1), dtype=int),
    )
    env = RewardEnv()
    cp = checkpoint(state=(1.3, 0.0, 0.0, 0.0))
    actions = np.ones((1, 2, 2))
    noises = np.zeros((1, 1, 2, 4))
    returns = model.rollout_returns(cp.state, actions, noises, cp, env, gamma=0.99)
    assert returns.shape == (2, 1, 1)
    np.testing.assert_allclose(returns[:, 0, 0], [1.09, -0.0199], atol=1e-12)
    np.testing.assert_allclose(
        model.rollout_returns(cp.state, actions, noises, cp, env, member_index=1),
        returns[1],
    )


def test_dataset_validates_transition_shapes():
    with pytest.raises(ValueError, match="actions"):
        TransitionDataset(
            np.zeros((2, 4)), np.zeros((2, 3)), np.zeros((2, 4)), np.array([0, 1])
        )
