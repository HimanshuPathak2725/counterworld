"""Linear ridge dynamics ensembles and vectorized imagined rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .environment import Checkpoint, DoubleIntegratorGapEnv
from .noise import covariance_factor


@dataclass(frozen=True)
class TransitionDataset:
    states: NDArray[np.float64]
    actions: NDArray[np.float64]
    next_states: NDArray[np.float64]
    episode_ids: NDArray[np.int64]

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=np.float64)
        actions = np.asarray(self.actions, dtype=np.float64)
        next_states = np.asarray(self.next_states, dtype=np.float64)
        raw_episode_ids = np.asarray(self.episode_ids)
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError("states must have shape (N, 4)")
        if actions.shape != (states.shape[0], 2):
            raise ValueError("actions must have shape (N, 2)")
        if next_states.shape != states.shape:
            raise ValueError("next_states must have shape (N, 4)")
        if raw_episode_ids.shape != (states.shape[0],):
            raise ValueError("episode_ids must have shape (N,)")
        if states.shape[0] == 0:
            raise ValueError("dataset must contain at least one transition")
        if not all(np.all(np.isfinite(array)) for array in (states, actions, next_states)):
            raise ValueError("dataset values must be finite")
        if not np.issubdtype(raw_episode_ids.dtype, np.integer):
            if not np.issubdtype(raw_episode_ids.dtype, np.floating) or not np.all(
                np.isfinite(raw_episode_ids)
            ) or not np.all(raw_episode_ids == np.floor(raw_episode_ids)):
                raise ValueError("episode_ids must contain integers")
        episode_ids = raw_episode_ids.astype(np.int64, copy=True)
        object.__setattr__(self, "states", np.array(states, copy=True))
        object.__setattr__(self, "actions", np.array(actions, copy=True))
        object.__setattr__(self, "next_states", np.array(next_states, copy=True))
        object.__setattr__(self, "episode_ids", episode_ids)

    def __len__(self) -> int:
        return self.states.shape[0]


@dataclass(frozen=True)
class RidgeEnsemble:
    """A member-wise linear map from [state, action, 1] to next state.

    ``predict_next(..., member_index=None)`` returns every member's next state.
    ``rollout_returns(..., member_index=None)`` returns every member's returns;
    averaging those returns is a planner operation, not an ensemble-mean next-state
    prediction.
    """

    coefficients: NDArray[np.float64]
    covariances: NDArray[np.float64]
    covariance_factors: NDArray[np.float64]
    bootstrap_episode_ids: NDArray[np.int64]
    ridge: float = 1.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        covariances = np.asarray(self.covariances, dtype=np.float64)
        factors = np.asarray(self.covariance_factors, dtype=np.float64)
        bootstrap_ids = np.asarray(self.bootstrap_episode_ids, dtype=np.int64)
        if coefficients.ndim != 3 or coefficients.shape[1:] != (7, 4):
            raise ValueError("coefficients must have shape (M, 7, 4)")
        members = coefficients.shape[0]
        if members == 0:
            raise ValueError("ensemble must contain at least one member")
        if covariances.shape != (members, 4, 4):
            raise ValueError("covariances must have shape (M, 4, 4)")
        if factors.shape != (members, 4, 4):
            raise ValueError("covariance_factors must have shape (M, 4, 4)")
        if bootstrap_ids.ndim != 2 or bootstrap_ids.shape[0] != members:
            raise ValueError("bootstrap_episode_ids must have shape (M, B)")
        if not all(np.all(np.isfinite(array)) for array in (coefficients, covariances, factors)):
            raise ValueError("ensemble parameters must be finite")
        if not np.isfinite(self.ridge) or self.ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        object.__setattr__(self, "coefficients", np.array(coefficients, copy=True))
        object.__setattr__(self, "covariances", np.array(covariances, copy=True))
        object.__setattr__(self, "covariance_factors", np.array(factors, copy=True))
        object.__setattr__(self, "bootstrap_episode_ids", np.array(bootstrap_ids, copy=True))

    @property
    def n_members(self) -> int:
        return self.coefficients.shape[0]

    @classmethod
    def fit(
        cls,
        dataset: TransitionDataset,
        train_episode_ids: ArrayLike,
        holdout_episode_ids: ArrayLike,
        n_members: int = 8,
        seed: int = 0,
        ridge: float = 1.0,
    ) -> "RidgeEnsemble":
        if not isinstance(n_members, (int, np.integer)) or n_members <= 0:
            raise ValueError("n_members must be a positive integer")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        train_ids = np.asarray(train_episode_ids, dtype=np.int64)
        holdout_ids = np.asarray(holdout_episode_ids, dtype=np.int64)
        if train_ids.ndim != 1 or train_ids.size == 0:
            raise ValueError("train_episode_ids must be a nonempty one-dimensional sequence")
        if holdout_ids.ndim != 1 or holdout_ids.size == 0:
            raise ValueError("holdout_episode_ids must be a nonempty one-dimensional sequence")
        if np.unique(train_ids).size != train_ids.size or np.unique(holdout_ids).size != holdout_ids.size:
            raise ValueError("episode splits must not contain duplicate ids")
        if np.intersect1d(train_ids, holdout_ids).size:
            raise ValueError("training and holdout episodes must be disjoint")
        available = np.unique(dataset.episode_ids)
        if not np.all(np.isin(train_ids, available)) or not np.all(np.isin(holdout_ids, available)):
            raise ValueError("all split episode ids must occur in the dataset")

        holdout_mask = np.isin(dataset.episode_ids, holdout_ids)
        holdout_x = _features(dataset.states[holdout_mask], dataset.actions[holdout_mask])
        holdout_y = dataset.next_states[holdout_mask]
        penalty = np.eye(7, dtype=np.float64) * float(ridge)
        penalty[-1, -1] = 0.0
        rng = np.random.default_rng(seed)
        coefficients = np.empty((n_members, 7, 4), dtype=np.float64)
        covariances = np.empty((n_members, 4, 4), dtype=np.float64)
        factors = np.empty((n_members, 4, 4), dtype=np.float64)
        bootstrap_ids = np.empty((n_members, train_ids.size), dtype=np.int64)

        for member in range(n_members):
            sampled_ids = rng.choice(train_ids, size=train_ids.size, replace=True)
            bootstrap_ids[member] = sampled_ids
            sampled_masks = [dataset.episode_ids == episode_id for episode_id in sampled_ids]
            row_indices = np.concatenate([np.flatnonzero(mask) for mask in sampled_masks])
            x = _features(dataset.states[row_indices], dataset.actions[row_indices])
            y = dataset.next_states[row_indices]
            beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
            residuals = holdout_y - holdout_x @ beta
            denominator = max(residuals.shape[0] - 1, 1)
            covariance = residuals.T @ residuals / denominator
            coefficients[member] = beta
            covariances[member] = covariance
            factors[member] = covariance_factor(covariance.copy(), floor=1e-8)

        return cls(coefficients, covariances, factors, bootstrap_ids, float(ridge))

    def _member_index(self, member_index: int) -> int:
        if not isinstance(member_index, (int, np.integer)):
            raise TypeError("member_index must be an integer or None")
        index = int(member_index)
        if index < 0:
            index += self.n_members
        if index < 0 or index >= self.n_members:
            raise IndexError("member_index is out of range")
        return index

    def predict_next(
        self,
        states: ArrayLike,
        actions: ArrayLike,
        canonical_noises: ArrayLike | None = None,
        member_index: int | None = None,
    ) -> NDArray[np.float64]:
        """Predict next states, with shape ``broadcast_prefix + (M, 4)`` for all members."""
        state_array = np.asarray(states, dtype=np.float64)
        action_array = np.asarray(actions, dtype=np.float64)
        if state_array.ndim < 1 or state_array.shape[-1] != 4:
            raise ValueError("states must have trailing dimension 4")
        if action_array.ndim < 1 or action_array.shape[-1] != 2:
            raise ValueError("actions must have trailing dimension 2")
        prefix = np.broadcast_shapes(state_array.shape[:-1], action_array.shape[:-1])
        if canonical_noises is not None:
            noise_array = np.asarray(canonical_noises, dtype=np.float64)
            if noise_array.ndim < 1 or noise_array.shape[-1] != 4:
                raise ValueError("canonical_noises must have trailing dimension 4")
            prefix = np.broadcast_shapes(prefix, noise_array.shape[:-1])
        else:
            noise_array = np.zeros(prefix + (4,), dtype=np.float64)
        state_array = np.broadcast_to(state_array, prefix + (4,))
        action_array = np.broadcast_to(action_array, prefix + (2,))
        noise_array = np.broadcast_to(noise_array, prefix + (4,))
        features = _features(state_array, action_array)

        if member_index is not None:
            member = self._member_index(member_index)
            prediction = features @ self.coefficients[member]
            return prediction + noise_array @ self.covariance_factors[member].T
        prediction = np.matmul(features, self.coefficients)
        stochastic = np.matmul(noise_array, self.covariance_factors)
        return prediction + stochastic

    def rollout_returns(
        self,
        initial_state: ArrayLike,
        action_sequences: ArrayLike,
        canonical_noises: ArrayLike,
        checkpoint: Checkpoint,
        env: DoubleIntegratorGapEnv,
        gamma: float = 0.99,
        member_index: int | None = None,
    ) -> NDArray[np.float64]:
        """Return wall-free open-loop returns.

        All-member output has shape ``(M, C, R)``; member-specific output has
        shape ``(C, R)``. Goal rewards are included once and later rewards are
        not accumulated.
        """
        initial = np.asarray(initial_state, dtype=np.float64)
        actions = np.asarray(action_sequences, dtype=np.float64)
        noises = np.asarray(canonical_noises, dtype=np.float64)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial_state must have shape (4,) and be finite")
        if actions.ndim != 3 or actions.shape[-1] != 2:
            raise ValueError("action_sequences must have shape (C, H, 2)")
        candidates, horizon = actions.shape[:2]
        if candidates == 0 or horizon == 0:
            raise ValueError("action_sequences must be nonempty")
        if (
            noises.ndim != 4
            or noises.shape[0] != candidates
            or noises.shape[2:] != (horizon, 4)
        ):
            raise ValueError("canonical_noises must have shape (C, R, H, 4)")
        replicates = noises.shape[1]
        if replicates == 0:
            raise ValueError("canonical_noises must contain at least one replicate")
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(noises)):
            raise ValueError("rollout inputs must be finite")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")

        selected_members = (
            np.arange(self.n_members, dtype=np.int64)
            if member_index is None
            else np.array([self._member_index(member_index)], dtype=np.int64)
        )
        member_count = selected_members.size
        states = np.broadcast_to(initial, (member_count, candidates, replicates, 4)).copy()
        active = np.broadcast_to(
            ~_goal_mask(initial, checkpoint, env), (member_count, candidates, replicates)
        ).copy()
        returns = np.zeros((member_count, candidates, replicates), dtype=np.float64)

        for step in range(horizon):
            step_actions = np.broadcast_to(
                actions[None, :, step, None, :], (member_count, candidates, replicates, 2)
            )
            features = _features(states, step_actions)
            member_coefficients = self.coefficients[selected_members]
            next_states = _member_matmul(features, member_coefficients)
            step_noises = np.broadcast_to(
                noises[None, :, :, step, :], (member_count, candidates, replicates, 4)
            )
            next_states += _member_matmul(
                step_noises, self.covariance_factors[selected_members]
            )
            reward_output = env.reward_many(
                states.reshape(-1, 4),
                next_states.reshape(-1, 4),
                checkpoint,
                collision=False,
                collision_points=None,
            )
            reward_values, _, goal_values = _reward_parts(reward_output)
            rewards = np.asarray(reward_values, dtype=np.float64).reshape(
                member_count, candidates, replicates
            )
            returns += (gamma**step) * rewards * active
            if goal_values is None:
                reached_goal = _goal_mask(next_states, checkpoint, env)
            else:
                reached_goal = np.asarray(goal_values, dtype=np.bool_).reshape(
                    member_count, candidates, replicates
                )
            states = np.where(active[..., None], next_states, states)
            active &= ~reached_goal
            if not np.any(active):
                break

        if member_index is not None:
            return returns[0]
        return returns

    def rollout_returns_batched(
        self,
        initial_states: ArrayLike,
        action_sequences: ArrayLike,
        canonical_noises: ArrayLike,
        checkpoint: Checkpoint,
        env: DoubleIntegratorGapEnv,
        gamma: float = 0.99,
        member_index: int | None = None,
    ) -> NDArray[np.float64]:
        """Return ``(M, B, C, R)`` imagined returns for B independent episodes."""
        initial = np.asarray(initial_states, dtype=np.float64)
        actions = np.asarray(action_sequences, dtype=np.float64)
        noises = np.asarray(canonical_noises, dtype=np.float64)
        if initial.ndim != 2 or initial.shape[1] != 4:
            raise ValueError("initial_states must have shape (B, 4)")
        episodes, candidates, horizon = actions.shape[:3]
        replicates = noises.shape[2] if noises.ndim == 5 else 0
        if (
            initial.shape[0] != episodes
            or noises.shape[:3] != (episodes, candidates, replicates)
            or noises.shape[3:] != (horizon, 4)
        ):
            raise ValueError("batched episode, candidate, and replicate dimensions are inconsistent")
        selected_members = (
            np.arange(self.n_members, dtype=np.int64)
            if member_index is None
            else np.array([self._member_index(member_index)], dtype=np.int64)
        )
        member_count = selected_members.size
        states = np.broadcast_to(initial[None, :, None, None, :], (member_count, episodes, candidates, replicates, 4)).copy()
        active = np.broadcast_to(
            ~_goal_mask(initial, checkpoint, env)[:, None, None],
            (member_count, episodes, candidates, replicates),
        ).copy()
        returns = np.zeros((member_count, episodes, candidates, replicates), dtype=np.float64)

        for step in range(horizon):
            step_actions = np.broadcast_to(
                actions[None, :, :, step, None, :],
                (member_count, episodes, candidates, replicates, 2),
            )
            features = _features(states, step_actions)
            member_coefficients = self.coefficients[selected_members]
            next_states = _member_matmul(features, member_coefficients)
            step_noises = np.broadcast_to(
                noises[None, :, :, :, step, :],
                (member_count, episodes, candidates, replicates, 4),
            )
            next_states += _member_matmul(
                step_noises, self.covariance_factors[selected_members]
            )
            reward_output = env.reward_many(
                states.reshape(-1, 4),
                next_states.reshape(-1, 4),
                checkpoint,
                collision=False,
                collision_points=None,
            )
            reward_values, _, goal_values = _reward_parts(reward_output)
            rewards = np.asarray(reward_values, dtype=np.float64).reshape(
                member_count, episodes, candidates, replicates
            )
            returns += (gamma**step) * rewards * active
            reached_goal = (
                _goal_mask(next_states, checkpoint, env)
                if goal_values is None
                else np.asarray(goal_values, dtype=np.bool_).reshape(
                    member_count, episodes, candidates, replicates
                )
            )
            states = np.where(active[..., None], next_states, states)
            active &= ~reached_goal
            if not np.any(active):
                break

        if member_index is not None:
            return returns[0]
        return returns


def _member_matmul(values: NDArray[np.float64], matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    members = values.shape[0]
    flattened = values.reshape(members, -1, values.shape[-1])
    return np.matmul(flattened, matrix).reshape(*values.shape[:-1], matrix.shape[-1])


def _features(states: NDArray[np.float64], actions: NDArray[np.float64]) -> NDArray[np.float64]:
    state_array = np.asarray(states, dtype=np.float64)
    action_array = np.asarray(actions, dtype=np.float64)
    ones = np.ones(state_array.shape[:-1] + (1,), dtype=np.float64)
    return np.concatenate((state_array, action_array, ones), axis=-1)


def _reward_parts(output: Any) -> tuple[Any, Any | None, Any | None]:
    if isinstance(output, dict):
        return output["reward"], output.get("terminal"), output.get("goal")
    if hasattr(output, "reward"):
        return output.reward, getattr(output, "terminal", None), getattr(output, "goal", None)
    if isinstance(output, (tuple, list)):
        if len(output) >= 3:
            return output[0], output[1], output[2]
        if len(output) == 1:
            return output[0], None, None
    return output, None, None


def _goal_mask(
    states: ArrayLike, checkpoint: Checkpoint, env: DoubleIntegratorGapEnv
) -> NDArray[np.bool_]:
    state_array = np.asarray(states, dtype=np.float64)
    goal = np.asarray(checkpoint.goal, dtype=np.float64)
    config = getattr(env, "config", None)
    radius = getattr(
        checkpoint,
        "goal_radius",
        getattr(env, "goal_radius", getattr(config, "goal_radius", 0.1)),
    )
    return np.linalg.norm(state_array[..., :2] - goal, axis=-1) <= radius
