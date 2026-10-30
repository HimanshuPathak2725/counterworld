"""Random-shooting MPC with explicit canonical-noise streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .environment import ACTION_GRID, Checkpoint, DoubleIntegratorGapEnv
from .models import RidgeEnsemble, _goal_mask, _reward_parts
from .noise import NoiseRegistry


@dataclass(frozen=True)
class RolloutRecord:
    states: NDArray[np.float64]
    actions: NDArray[np.float64]
    rewards: NDArray[np.float64]
    total_return: float
    collision: bool
    goal: bool
    real_environment_steps: int
    model_calls: int
    selected_indices: NDArray[np.int64]


class RandomShootingMPC:
    def __init__(
        self,
        env: DoubleIntegratorGapEnv,
        model: RidgeEnsemble,
        n_candidates: int = 512,
        horizon: int = 12,
        gamma: float = 0.99,
        selection_replicates: int = 4,
    ) -> None:
        if not isinstance(n_candidates, (int, np.integer)) or n_candidates <= 0:
            raise ValueError("n_candidates must be a positive integer")
        if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(selection_replicates, (int, np.integer)) or selection_replicates <= 0:
            raise ValueError("selection_replicates must be a positive integer")
        if not np.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and nonnegative")
        self.env = env
        self.model = model
        self.n_candidates = int(n_candidates)
        self.horizon = int(horizon)
        self.gamma = float(gamma)
        self.selection_replicates = int(selection_replicates)

    def candidate_actions(
        self,
        noise_registry: NoiseRegistry,
        checkpoint_id: int,
        episode_id: int,
        time_step: int,
    ) -> NDArray[np.float64]:
        canonical = _normal(
            noise_registry,
            "planner-candidates",
            (checkpoint_id, episode_id, time_step),
            shape=(self.n_candidates, self.horizon),
        )
        grid = np.asarray(ACTION_GRID, dtype=np.float64)
        if grid.shape != (9, 2):
            raise ValueError("ACTION_GRID must have shape (9, 2)")
        split = 0.43072729929545756
        action_indices = np.digitize(canonical, (-split, split))
        return grid[action_indices]

    def select_action(
        self,
        state: ArrayLike,
        checkpoint: Checkpoint,
        noise_registry: NoiseRegistry,
        checkpoint_id: int,
        episode_id: int,
        time_step: int,
        controller: str = "model",
        member_index: int | None = None,
    ) -> tuple[NDArray[np.float64], int, dict[str, Any]]:
        if controller not in ("model", "reference"):
            raise ValueError("controller must be 'model' or 'reference'")
        initial = np.asarray(state, dtype=np.float64)
        if initial.shape != (4,) or not np.all(np.isfinite(initial)):
            raise ValueError("state must have shape (4,) and be finite")
        candidates = self.candidate_actions(
            noise_registry, checkpoint_id, episode_id, time_step
        )
        selection_noise = _normal(
            noise_registry,
            "planner-selection",
            (checkpoint_id, episode_id, time_step),
            shape=(self.n_candidates, self.selection_replicates, self.horizon, 4),
        )
        if controller == "model":
            rollout_returns = self.model.rollout_returns(
                initial,
                candidates,
                selection_noise,
                checkpoint,
                self.env,
                gamma=self.gamma,
                member_index=member_index,
            )
            if member_index is None:
                candidate_returns = rollout_returns.mean(axis=(0, 2))
            else:
                candidate_returns = rollout_returns.mean(axis=1)
        else:
            rollout_returns = self._reference_rollout_returns(
                initial, candidates, selection_noise, checkpoint
            )
            candidate_returns = rollout_returns.mean(axis=1)
        selected_index = int(np.argmax(candidate_returns))
        diagnostics = {
            "candidates": candidates,
            "candidate_returns": candidate_returns,
            "rollout_returns": rollout_returns,
            "selected_index": selected_index,
            "controller": controller,
            "member_index": member_index,
        }
        return candidates[selected_index, 0].copy(), selected_index, diagnostics

    def _reference_rollout_returns(
        self,
        initial_state: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        canonical_noises: NDArray[np.float64],
        checkpoint: Checkpoint,
    ) -> NDArray[np.float64]:
        candidates, replicates, horizon, _ = canonical_noises.shape
        states = np.broadcast_to(initial_state, (candidates, replicates, 4)).copy()
        active = np.broadcast_to(
            ~_goal_mask(initial_state, checkpoint, self.env), (candidates, replicates)
        ).copy()
        returns = np.zeros((candidates, replicates), dtype=np.float64)
        factor = self.env.environment_factor(checkpoint)

        for step in range(horizon):
            step_states = states.reshape(-1, 4)
            step_actions = np.broadcast_to(
                action_sequences[:, step, None, :], (candidates, replicates, 2)
            ).reshape(-1, 2)
            step_noises = canonical_noises[:, :, step, :].reshape(-1, 4)
            transition = self.env.transition_many(
                step_states,
                step_actions,
                step_noises,
                factor,
                checkpoint,
                include_wall=True,
            )
            next_values, reward_values, terminal_values, _, _ = _transition_parts(transition)
            next_states = np.asarray(next_values, dtype=np.float64).reshape(
                candidates, replicates, 4
            )
            rewards = np.asarray(reward_values, dtype=np.float64).reshape(
                candidates, replicates
            )
            terminal = np.asarray(terminal_values, dtype=np.bool_).reshape(
                candidates, replicates
            )
            returns += (self.gamma**step) * rewards * active
            states = np.where(active[..., None], next_states, states)
            active &= ~terminal
            if not np.any(active):
                break
        return returns

    def _reference_rollout_returns_batched(
        self,
        initial_states: NDArray[np.float64],
        action_sequences: NDArray[np.float64],
        canonical_noises: NDArray[np.float64],
        checkpoint: Checkpoint,
    ) -> NDArray[np.float64]:
        episodes, candidates, replicates, _, _ = canonical_noises.shape
        states = np.broadcast_to(initial_states[:, None, None, :], (episodes, candidates, replicates, 4)).copy()
        active = np.broadcast_to(
            ~_goal_mask(initial_states, checkpoint, self.env)[:, None, None],
            (episodes, candidates, replicates),
        ).copy()
        returns = np.zeros((episodes, candidates, replicates), dtype=np.float64)
        factor = self.env.environment_factor(checkpoint)
        horizon = action_sequences.shape[2]

        for step in range(horizon):
            step_states = states.reshape(-1, 4)
            step_actions = np.broadcast_to(
                action_sequences[:, :, step, None, :],
                (episodes, candidates, replicates, 2),
            ).reshape(-1, 2)
            step_noises = canonical_noises[:, :, :, step, :].reshape(-1, 4)
            transition = self.env.transition_many(
                step_states,
                step_actions,
                step_noises,
                factor,
                checkpoint,
                include_wall=True,
            )
            next_values, reward_values, terminal_values, _, _ = _transition_parts(transition)
            next_states = np.asarray(next_values, dtype=np.float64).reshape(
                episodes, candidates, replicates, 4
            )
            rewards = np.asarray(reward_values, dtype=np.float64).reshape(
                episodes, candidates, replicates
            )
            terminal = np.asarray(terminal_values, dtype=np.bool_).reshape(
                episodes, candidates, replicates
            )
            returns += (self.gamma**step) * rewards * active
            states = np.where(active[..., None], next_states, states)
            active &= ~terminal
            if not np.any(active):
                break
        return returns

    def _select_actions_batched(
        self,
        states: NDArray[np.float64],
        checkpoint: Checkpoint,
        noise_registry: NoiseRegistry,
        checkpoint_id: int,
        time_step: int,
        controller: str,
        member_index: int | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        episodes = states.shape[0]
        candidates = np.stack([
            self.candidate_actions(noise_registry, checkpoint_id, episode_id, time_step)
            for episode_id in range(episodes)
        ])
        selection_noise = np.stack([
            _normal(
                noise_registry,
                "planner-selection",
                (checkpoint_id, episode_id, time_step),
                shape=(self.n_candidates, self.selection_replicates, self.horizon, 4),
            )
            for episode_id in range(episodes)
        ])
        if controller == "model":
            rollout_returns = self.model.rollout_returns_batched(
                states,
                candidates,
                selection_noise,
                checkpoint,
                self.env,
                gamma=self.gamma,
                member_index=member_index,
            )
            candidate_returns = (
                rollout_returns.mean(axis=(0, 3))
                if member_index is None
                else rollout_returns.mean(axis=2)
            )
        else:
            rollout_returns = self._reference_rollout_returns_batched(
                states, candidates, selection_noise, checkpoint
            )
            candidate_returns = rollout_returns.mean(axis=2)
        selected_indices = np.argmax(candidate_returns, axis=1).astype(np.int64)
        return candidates[np.arange(episodes), selected_indices, 0].copy(), selected_indices

    def _run_episodes_batched(
        self,
        checkpoint: Checkpoint,
        controller: str,
        execution: str,
        noise_registry: NoiseRegistry,
        stream: str,
        checkpoint_id: int,
        k: int,
        member_index: int | None,
    ) -> list[RolloutRecord]:
        if controller not in ("model", "reference"):
            raise ValueError("controller must be 'model' or 'reference'")
        if execution not in ("environment", "model"):
            raise ValueError("execution must be 'environment' or 'model'")
        initial = _checkpoint_state(checkpoint)
        states = np.broadcast_to(initial, (k, 4)).copy()
        trajectories = [[state.copy()] for state in states]
        action_history = [[] for _ in range(k)]
        reward_history = [[] for _ in range(k)]
        selected_history = [[] for _ in range(k)]
        total_returns = np.zeros(k, dtype=np.float64)
        collision_flags = np.zeros(k, dtype=bool)
        goal_flags = _goal_mask(states, checkpoint, self.env)
        active = ~goal_flags
        real_steps = np.zeros(k, dtype=np.int64)
        model_calls = np.zeros(k, dtype=np.int64)
        config = getattr(self.env, "config", None)
        default_steps = getattr(config, "horizon", self.horizon)
        max_steps = int(getattr(self.env, "max_steps", getattr(self.env, "max_episode_steps", default_steps)))
        factor = self.env.environment_factor(checkpoint) if execution == "environment" else None

        for time_step in range(max_steps):
            if not np.any(active):
                break
            actions, selected = self._select_actions_batched(
                states,
                checkpoint,
                noise_registry,
                checkpoint_id,
                time_step,
                controller,
                member_index,
            )
            if controller == "model":
                model_calls += active
            canonical_noises = np.stack([
                _normal(noise_registry, stream, (checkpoint_id, episode_id, time_step), shape=(4,))
                for episode_id in range(k)
            ])
            if execution == "environment":
                transition = self.env.transition_many(
                    states,
                    actions,
                    canonical_noises,
                    factor,
                    checkpoint,
                    include_wall=True,
                )
                next_states, rewards, terminals, collisions, goal_values = _transition_parts(transition)
                real_steps += active
            else:
                predicted = self.model.predict_next(
                    states, actions, canonical_noises, member_index=member_index
                )
                model_calls += active
                next_states = predicted.mean(axis=0) if member_index is None else predicted
                reward_output = self.env.reward_many(
                    states,
                    next_states,
                    checkpoint,
                    collision=False,
                    collision_points=None,
                )
                rewards, terminals, goal_values = _reward_parts(reward_output)
                collisions = np.zeros(k, dtype=bool)
                if goal_values is None:
                    goal_values = _goal_mask(next_states, checkpoint, self.env)
                if terminals is None:
                    terminals = goal_values

            total_returns += (self.gamma**time_step) * rewards * active
            collision_flags |= collisions
            goal_flags |= goal_values
            for episode_id in range(k):
                if active[episode_id]:
                    action_history[episode_id].append(actions[episode_id].copy())
                    reward_history[episode_id].append(float(rewards[episode_id]))
                    selected_history[episode_id].append(int(selected[episode_id]))
                    trajectories[episode_id].append(next_states[episode_id].copy())
            states = np.where(active[:, None], next_states, states)
            active &= ~terminals

        return [
            RolloutRecord(
                states=np.asarray(trajectories[episode_id], dtype=np.float64),
                actions=np.asarray(action_history[episode_id], dtype=np.float64).reshape(-1, 2),
                rewards=np.asarray(reward_history[episode_id], dtype=np.float64),
                total_return=float(total_returns[episode_id]),
                collision=bool(collision_flags[episode_id]),
                goal=bool(goal_flags[episode_id]),
                real_environment_steps=int(real_steps[episode_id]),
                model_calls=int(model_calls[episode_id]),
                selected_indices=np.asarray(selected_history[episode_id], dtype=np.int64),
            )
            for episode_id in range(k)
        ]

    def run_episode(
        self,
        checkpoint: Checkpoint,
        controller: str,
        execution: str,
        noise_registry: NoiseRegistry,
        stream: str,
        checkpoint_id: int,
        episode_id: int,
        member_index: int | None = None,
    ) -> RolloutRecord:
        """Run one closed-loop episode.

        ``model_calls`` counts batched model API invocations. Reference-controller
        imagined environment transitions are not real environment interactions;
        ``real_environment_steps`` counts executed environment transitions only.
        """
        if controller not in ("model", "reference"):
            raise ValueError("controller must be 'model' or 'reference'")
        if execution not in ("environment", "model"):
            raise ValueError("execution must be 'environment' or 'model'")
        state = _checkpoint_state(checkpoint)
        states = [state.copy()]
        actions: list[NDArray[np.float64]] = []
        rewards: list[float] = []
        selected_indices: list[int] = []
        total_return = 0.0
        collision = False
        goal = bool(_goal_mask(state, checkpoint, self.env))
        real_environment_steps = 0
        model_calls = 0
        config = getattr(self.env, "config", None)
        default_steps = getattr(config, "horizon", self.horizon)
        max_steps = int(
            getattr(self.env, "max_steps", getattr(self.env, "max_episode_steps", default_steps))
        )
        if max_steps <= 0:
            raise ValueError("environment max_steps must be positive")
        factor = self.env.environment_factor(checkpoint) if execution == "environment" else None

        for time_step in range(max_steps):
            if goal:
                break
            action, selected_index, _ = self.select_action(
                state,
                checkpoint,
                noise_registry,
                checkpoint_id,
                episode_id,
                time_step,
                controller=controller,
                member_index=member_index,
            )
            if controller == "model":
                model_calls += 1
            canonical_noise = _normal(
            noise_registry,
                stream, (checkpoint_id, episode_id, time_step), shape=(4,)
            )
            if execution == "environment":
                transition = self.env.transition_many(
                    state[None, :],
                    action[None, :],
                    canonical_noise[None, :],
                    factor,
                    checkpoint,
                    include_wall=True,
                )
                next_values, reward_values, terminal_values, collision_values, goal_values = (
                    _transition_parts(transition)
                )
                next_state = np.asarray(next_values, dtype=np.float64)[0]
                reward = float(np.asarray(reward_values)[0])
                terminal = bool(np.asarray(terminal_values)[0])
                step_collision = bool(np.asarray(collision_values)[0])
                step_goal = bool(np.asarray(goal_values)[0])
                real_environment_steps += 1
            else:
                predicted = self.model.predict_next(
                    state, action, canonical_noise, member_index=member_index
                )
                model_calls += 1
                next_state = predicted.mean(axis=0) if member_index is None else predicted
                reward_output = self.env.reward_many(
                    state[None, :],
                    next_state[None, :],
                    checkpoint,
                    collision=False,
                    collision_points=None,
                )
                reward_values, terminal_values, goal_values = _reward_parts(reward_output)
                reward = float(np.asarray(reward_values)[0])
                step_collision = False
                step_goal = (
                    bool(_goal_mask(next_state, checkpoint, self.env))
                    if goal_values is None
                    else bool(np.asarray(goal_values)[0])
                )
                terminal = (
                    step_goal
                    if terminal_values is None
                    else bool(np.asarray(terminal_values)[0])
                )

            total_return += (self.gamma**time_step) * reward
            collision |= step_collision
            goal |= step_goal
            actions.append(action.copy())
            rewards.append(reward)
            selected_indices.append(selected_index)
            states.append(next_state.copy())
            state = next_state
            if terminal:
                break

        return RolloutRecord(
            states=np.asarray(states, dtype=np.float64),
            actions=np.asarray(actions, dtype=np.float64).reshape(-1, 2),
            rewards=np.asarray(rewards, dtype=np.float64),
            total_return=float(total_return),
            collision=bool(collision),
            goal=bool(goal),
            real_environment_steps=real_environment_steps,
            model_calls=model_calls,
            selected_indices=np.asarray(selected_indices, dtype=np.int64),
        )

    def run_episodes(
        self,
        checkpoint: Checkpoint,
        controller: str,
        execution: str,
        noise_registry: NoiseRegistry,
        stream: str,
        checkpoint_id: int,
        k: int,
        member_index: int | None = None,
        batched: bool = False,
    ) -> list[RolloutRecord]:
        if not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError("k must be a nonnegative integer")
        if k == 0:
            return []
        if k == 1 or not batched:
            return [
                self.run_episode(
                    checkpoint,
                    controller,
                    execution,
                    noise_registry,
                    stream,
                    checkpoint_id,
                    episode_id,
                    member_index=member_index,
                )
                for episode_id in range(int(k))
            ]
        return self._run_episodes_batched(
            checkpoint,
            controller,
            execution,
            noise_registry,
            stream,
            checkpoint_id,
            int(k),
            member_index,
        )


def _normal(
    registry: NoiseRegistry,
    stream: str,
    indices: tuple[int, ...],
    shape: tuple[int, ...],
) -> NDArray[np.float64]:
    try:
        return registry.normal(stream, indices, shape=shape)
    except TypeError as tuple_signature_error:
        try:
            return registry.normal(stream, *indices, shape=shape)
        except TypeError:
            raise tuple_signature_error


def _transition_parts(result: Any) -> tuple[Any, Any, Any, Any, Any]:
    if isinstance(result, dict):
        return (
            result["next_state"],
            result["reward"],
            result["terminal"],
            result["collision"],
            result["goal"],
        )
    if hasattr(result, "next_state"):
        return result.next_state, result.reward, result.terminal, result.collision, result.goal
    return result


def _checkpoint_state(checkpoint: Checkpoint) -> NDArray[np.float64]:
    if hasattr(checkpoint, "state"):
        state = np.asarray(checkpoint.state, dtype=np.float64)
    elif hasattr(checkpoint, "initial_state"):
        state = np.asarray(checkpoint.initial_state, dtype=np.float64)
    else:
        raise AttributeError("checkpoint must expose state or initial_state")
    if state.shape != (4,) or not np.all(np.isfinite(state)):
        raise ValueError("checkpoint state must have shape (4,) and be finite")
    return state
