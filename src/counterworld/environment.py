"""Explicit-state, explicit-noise double integrator with a random-gap wall."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np


ACTION_GRID = np.asarray(list(product((-1.0, 0.0, 1.0), repeat=2)), dtype=float)
ACTIONS = ACTION_GRID


@dataclass
class EnvironmentConfig:
    goal: np.ndarray = field(default_factory=lambda: np.asarray([1.5, 0.0]))
    goal_radius: float = 0.1
    horizon: int = 12
    gamma: float = 0.99
    collision_reward: float = -4.0
    step_cost: float = 0.01


@dataclass
class Checkpoint:
    state: np.ndarray
    gap_center: float
    gap_width: float = 0.6
    goal: np.ndarray = field(default_factory=lambda: np.asarray([1.5, 0.0]))
    sigma_p: float = 0.02
    sigma_v: float = 0.02
    regime: str = ""
    checkpoint_id: str = ""


@dataclass
class Transition:
    next_state: np.ndarray
    reward: float
    terminal: bool
    collision: bool
    goal: bool


class DoubleIntegratorGapEnv:
    wall_y_min = -1.5
    wall_y_max = 1.5

    def __init__(self, config: EnvironmentConfig | None = None) -> None:
        self.config = config if config is not None else EnvironmentConfig()

    def reset(self, checkpoint: Checkpoint) -> np.ndarray:
        state = np.asarray(checkpoint.state, dtype=float)
        if state.shape != (4,):
            raise ValueError("checkpoint.state must have shape (4,)")
        return state.copy()

    def environment_factor(self, checkpoint: Checkpoint) -> np.ndarray:
        sigma_p = float(checkpoint.sigma_p)
        sigma_v = float(checkpoint.sigma_v)
        if not np.isfinite(sigma_p) or not np.isfinite(sigma_v) or sigma_p < 0 or sigma_v < 0:
            raise ValueError("noise standard deviations must be finite and nonnegative")
        return np.diag([sigma_p, sigma_p, sigma_v, sigma_v])

    @staticmethod
    def _goal(checkpoint: Checkpoint) -> np.ndarray:
        goal = np.asarray(checkpoint.goal, dtype=float)
        if goal.shape != (2,):
            raise ValueError("checkpoint.goal must have shape (2,)")
        return goal

    def _solid_y(self, y: np.ndarray, checkpoint: Checkpoint) -> np.ndarray:
        lower = checkpoint.gap_center - checkpoint.gap_width / 2.0
        upper = checkpoint.gap_center + checkpoint.gap_width / 2.0
        within_wall = (y >= self.wall_y_min) & (y <= self.wall_y_max)
        return within_wall & ((y <= lower) | (y >= upper))

    def _wall_intersections(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        checkpoint: Checkpoint,
    ) -> tuple[np.ndarray, np.ndarray]:
        batch_size = states.shape[0]
        start = states[:, :2]
        end = next_states[:, :2]
        delta = end - start
        points = end.copy()
        best_t = np.full(batch_size, np.inf)
        collision = np.zeros(batch_size, dtype=bool)

        def record(mask: np.ndarray, t: np.ndarray, y: np.ndarray) -> None:
            update = mask & (t < best_t)
            best_t[update] = t[update]
            points[update, 0] = 0.0
            points[update, 1] = y[update]
            collision[update] = True

        zeros = np.zeros(batch_size)
        initial = (start[:, 0] == 0.0) & self._solid_y(start[:, 1], checkpoint)
        record(initial, zeros, start[:, 1])

        crossing_t = np.full(batch_size, np.inf)
        nonparallel = delta[:, 0] != 0.0
        np.divide(-start[:, 0], delta[:, 0], out=crossing_t, where=nonparallel)
        safe_crossing_t = np.where(nonparallel, crossing_t, 0.0)
        crossing_y = start[:, 1] + safe_crossing_t * delta[:, 1]
        crossing = (
            nonparallel
            & (crossing_t >= 0.0)
            & (crossing_t <= 1.0)
            & self._solid_y(crossing_y, checkpoint)
        )
        record(crossing, crossing_t, crossing_y)

        lower = checkpoint.gap_center - checkpoint.gap_width / 2.0
        upper = checkpoint.gap_center + checkpoint.gap_width / 2.0
        solid_intervals = (
            (self.wall_y_min, min(lower, self.wall_y_max)),
            (max(upper, self.wall_y_min), self.wall_y_max),
        )
        collinear = (start[:, 0] == 0.0) & (end[:, 0] == 0.0)
        moving_y = delta[:, 1] != 0.0
        inverse_dy = np.zeros(batch_size)
        np.divide(1.0, delta[:, 1], out=inverse_dy, where=moving_y)
        for low, high in solid_intervals:
            if high < low:
                continue
            t_low = (low - start[:, 1]) * inverse_dy
            t_high = (high - start[:, 1]) * inverse_dy
            entry = np.maximum(0.0, np.minimum(t_low, t_high))
            exit = np.minimum(1.0, np.maximum(t_low, t_high))
            entry_y = start[:, 1] + entry * delta[:, 1]
            overlap = (
                collinear
                & moving_y
                & (entry <= exit)
                & (entry <= 1.0)
                & (exit >= 0.0)
                & self._solid_y(entry_y, checkpoint)
            )
            record(overlap, entry, entry_y)

        return collision, points

    def reward_many(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        checkpoint: Checkpoint,
        collision: bool | np.ndarray = False,
        collision_points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply environment reward and goal semantics to supplied next states."""
        states = np.asarray(states, dtype=float)
        next_states = np.asarray(next_states, dtype=float)
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError("states must have shape (B, 4)")
        if next_states.shape != states.shape:
            raise ValueError("next_states must have the same shape as states")
        batch_size = states.shape[0]
        collisions = np.asarray(collision, dtype=bool)
        try:
            collisions = np.broadcast_to(collisions, (batch_size,))
        except ValueError as error:
            raise ValueError("collision must be scalar or have shape (B,)") from error

        evaluated_states = next_states.copy()
        if collision_points is not None:
            points = np.asarray(collision_points, dtype=float)
            if points.shape == (2,):
                points = np.broadcast_to(points, (batch_size, 2))
            if points.shape != (batch_size, 2):
                raise ValueError("collision_points must have shape (B, 2)")
            evaluated_states[collisions, :2] = points[collisions]

        goal = self._goal(checkpoint)
        initial_distance = np.linalg.norm(states[:, :2] - goal, axis=1)
        next_distance = np.linalg.norm(evaluated_states[:, :2] - goal, axis=1)
        reached_goal = (~collisions) & (next_distance <= self.config.goal_radius)
        ordinary_reward = -self.config.step_cost + initial_distance - next_distance
        rewards = np.where(collisions, self.config.collision_reward, ordinary_reward + reached_goal)
        terminal = collisions | reached_goal
        return rewards, terminal, reached_goal

    def step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        canonical_noise: np.ndarray,
        factor: np.ndarray,
        checkpoint: Checkpoint,
        include_wall: bool = True,
    ) -> Transition:
        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)
        canonical_noise = np.asarray(canonical_noise, dtype=float)
        if state.shape != (4,) or action.shape != (2,) or canonical_noise.shape != (4,):
            raise ValueError("step expects state (4,), action (2,), and canonical_noise (4,)")
        next_states, rewards, terminal, collision, goal = self.transition_many(
            state[None, :],
            action[None, :],
            canonical_noise[None, :],
            factor,
            checkpoint,
            include_wall=include_wall,
        )
        return Transition(
            next_state=next_states[0],
            reward=float(rewards[0]),
            terminal=bool(terminal[0]),
            collision=bool(collision[0]),
            goal=bool(goal[0]),
        )

    def transition_many(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        canonical_noises: np.ndarray,
        factor: np.ndarray,
        checkpoint: Checkpoint,
        include_wall: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        states = np.asarray(states, dtype=float)
        actions = np.asarray(actions, dtype=float)
        canonical_noises = np.asarray(canonical_noises, dtype=float)
        factors = np.asarray(factor, dtype=float)
        if states.ndim != 2 or states.shape[1] != 4:
            raise ValueError("states must have shape (B, 4)")
        batch_size = states.shape[0]
        if actions.shape != (batch_size, 2):
            raise ValueError("actions must have shape (B, 2)")
        if canonical_noises.shape != (batch_size, 4):
            raise ValueError("canonical_noises must have shape (B, 4)")
        if factors.shape != (4, 4) and factors.shape != (batch_size, 4, 4):
            raise ValueError("factor must have shape (4, 4) or (B, 4, 4)")

        noise = np.matmul(factors, canonical_noises[..., None])[..., 0]
        next_states = states.copy()
        next_states[:, :2] += states[:, 2:] + 0.5 * actions + noise[:, :2]
        next_states[:, 2:] += actions + noise[:, 2:]

        if include_wall:
            collision, collision_points = self._wall_intersections(states, next_states, checkpoint)
            next_states[collision, :2] = collision_points[collision]
            next_states[collision, 2:] = 0.0
        else:
            collision = np.zeros(batch_size, dtype=bool)
            collision_points = next_states[:, :2]

        rewards, terminal, goal = self.reward_many(
            states,
            next_states,
            checkpoint,
            collision=collision,
            collision_points=collision_points,
        )
        return next_states, rewards, terminal, collision, goal

    def geometric_detour_actions(
        self,
        checkpoint: Checkpoint,
        horizon: int = 12,
    ) -> np.ndarray | None:
        """Find a short zero-noise plan guided by the gap center and goal.

        A fixed-width beam keeps this simple and deterministic, but makes it neither
        complete nor optimal; feasible plans can be missed in tight or unusual cases.
        """
        if horizon < 0:
            raise ValueError("horizon must be nonnegative")
        initial_state = self.reset(checkpoint)
        goal = self._goal(checkpoint)
        if np.linalg.norm(initial_state[:2] - goal) <= self.config.goal_radius:
            return np.empty((0, 2))
        if horizon == 0:
            return None

        states = initial_state[None, :]
        sequences = np.empty((1, 0, 2))
        zero_factor = np.zeros((4, 4))
        needs_crossing = (initial_state[0] < 0.0 < goal[0]) or (
            goal[0] < 0.0 < initial_state[0]
        )
        action_count = len(ACTION_GRID)
        for _ in range(horizon):
            candidate_count = states.shape[0]
            expanded_states = np.repeat(states, action_count, axis=0)
            expanded_actions = np.tile(ACTION_GRID, (candidate_count, 1))
            canonical_noises = np.zeros((candidate_count * action_count, 4))
            next_states, _, _, collision, reached_goal = self.transition_many(
                expanded_states,
                expanded_actions,
                canonical_noises,
                zero_factor,
                checkpoint,
                include_wall=True,
            )
            expanded_sequences = np.concatenate(
                (np.repeat(sequences, action_count, axis=0), expanded_actions[:, None, :]),
                axis=1,
            )
            if np.any(reached_goal):
                return expanded_sequences[np.flatnonzero(reached_goal)[0]]

            valid = ~collision
            states = next_states[valid]
            sequences = expanded_sequences[valid]
            if states.shape[0] == 0:
                break
            targets = np.broadcast_to(goal, (states.shape[0], 2)).copy()
            if needs_crossing:
                before_wall = states[:, 0] < 0.0 if initial_state[0] < 0.0 else states[:, 0] > 0.0
                targets[before_wall] = [0.0, checkpoint.gap_center]
            scores = np.linalg.norm(states[:, :2] - targets, axis=1)
            scores += 0.05 * np.linalg.norm(states[:, 2:], axis=1)
            best = np.argsort(scores, kind="stable")[:32]
            states = states[best]
            sequences = sequences[best]
        return None
