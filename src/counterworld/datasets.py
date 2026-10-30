"""Deterministic development datasets and crossed checkpoint panels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .environment import ACTION_GRID, Checkpoint, DoubleIntegratorGapEnv
from .models import TransitionDataset
from .noise import NoiseRegistry


def make_training_dataset(
    env: DoubleIntegratorGapEnv,
    root_seed: int,
    episodes: int = 64,
    max_steps: int = 64,
) -> TransitionDataset:
    registry = NoiseRegistry(root_seed)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    episode_ids: list[int] = []

    for episode_id in range(episodes):
        action_rng = registry.rng("training-actions", episode_id)
        initial = np.array([
            action_rng.uniform(-2, -1),
            action_rng.uniform(-0.75, 0.75),
            0.0,
            0.0,
        ])
        checkpoint = Checkpoint(
            state=initial,
            gap_center=action_rng.uniform(-0.45, 0.45),
            regime="training",
            checkpoint_id=episode_id,
        )
        state = env.reset(checkpoint)
        for time_step in range(max_steps):
            action = ACTION_GRID[action_rng.integers(len(ACTION_GRID))]
            canonical_noise = registry.normal("training-noise", (episode_id, time_step), shape=(4,))
            transition = env.step(
                state,
                action,
                canonical_noise,
                env.environment_factor(checkpoint),
                checkpoint,
                include_wall=True,
            )
            next_state = transition.next_state
            if next_state[0] >= -0.25:
                break
            states.append(state.copy())
            actions.append(action.copy())
            next_states.append(next_state.copy())
            episode_ids.append(episode_id)
            state = next_state

    if not states:
        raise ValueError("Training collection produced no safe transitions")
    return TransitionDataset(
        states=np.asarray(states),
        actions=np.asarray(actions),
        next_states=np.asarray(next_states),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def make_checkpoint_panel(
    regime: str,
    panel_seed: int,
    count: int = 48,
    gap_width: float = 0.6,
    random_goals: bool = False,
    r0_speed_range: tuple[float, float] | None = None,
) -> list[Checkpoint]:
    if regime not in {"R0", "R1", "R2"}:
        raise ValueError(f"Unknown regime: {regime}")
    if count != 48:
        raise ValueError("The crossed 0.2 panel requires exactly 48 checkpoints")
    source_regime = "R1" if regime == "R2" else regime
    rng = NoiseRegistry(panel_seed).rng(f"checkpoint-panel-{source_regime}")
    checkpoints: list[Checkpoint] = []

    for index in range(count):
        if source_regime == "R0":
            stratum = index // 16
            low, high = [(-1.5, -0.8), (-0.8, -0.4), (-0.4, -0.1)][stratum]
            speed_low, speed_high = r0_speed_range or (0.0, 0.5)
            vx = rng.uniform(speed_low, speed_high)
        else:
            low, high = -0.4, -0.1
            vx = rng.uniform(0.75, 1.5)
        state = np.array([
            rng.uniform(low, high),
            rng.uniform(-0.5, 0.5),
            vx,
            rng.uniform(-0.25, 0.25),
        ])
        goal = (
            np.array([1.5, rng.uniform(-1.0, 1.0)])
            if source_regime == "R0" and random_goals
            else np.array([1.5, 0.0])
        )
        checkpoints.append(Checkpoint(
            state=state,
            goal=goal,
            gap_center=rng.uniform(-0.45, 0.45),
            gap_width=gap_width,
            sigma_v=0.04 if regime == "R2" else 0.02,
            regime=regime,
            checkpoint_id=index if regime == "R0" else 1000 + index,
        ))
    return checkpoints


def stratified_subset(checkpoints: list[Checkpoint], count: int) -> list[Checkpoint]:
    if count > len(checkpoints):
        raise ValueError("Subset cannot exceed panel size")
    indices = np.linspace(0, len(checkpoints) - 1, count, dtype=int)
    return [checkpoints[index] for index in indices]


def with_gap_width(checkpoints: list[Checkpoint], width: float) -> list[Checkpoint]:
    return [replace(checkpoint, gap_width=width) for checkpoint in checkpoints]
