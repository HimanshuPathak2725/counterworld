"""Decision-focused scores with explicit environment and model budgets."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np


def member_count(model) -> int:
    if hasattr(model, "members"):
        return len(model.members)
    if hasattr(model, "n_members"):
        return int(model.n_members)
    return 8


def transition_parts(result):
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


def returns(records) -> np.ndarray:
    return np.asarray([record.total_return for record in records], dtype=float)


def odd_even_mean(records) -> tuple[float, float]:
    values = returns(records)
    return float(values[::2].mean()), float(values[1::2].mean())


def odd_even_values(values: np.ndarray) -> tuple[float, float]:
    return float(values[::2].mean()), float(values[1::2].mean())


def checkpoint_metadata(checkpoint) -> dict:
    metadata = asdict(checkpoint)
    metadata["state"] = np.asarray(metadata["state"]).tolist()
    metadata["goal"] = np.asarray(metadata["goal"]).tolist()
    return metadata


def evaluate_target(env, planner, checkpoint, registry, episode_count: int = 64) -> dict:
    records = planner.run_episodes(
        checkpoint,
        controller="model",
        execution="environment",
        noise_registry=registry,
        stream="outcome-evaluation",
        checkpoint_id=checkpoint.checkpoint_id,
        k=episode_count,
    )
    collisions = np.asarray([record.collision for record in records], dtype=float)
    return {
        "collision_probability": float(collisions.mean()),
        "collision_probability_a": float(collisions[::2].mean()),
        "collision_probability_b": float(collisions[1::2].mean()),
        "target_return_mean": float(returns(records).mean()),
        "target_real_environment_steps": int(sum(record.real_environment_steps for record in records)),
        "target_model_calls": int(sum(record.model_calls for record in records)),
    }


def evaluate_free_scores(env, planner, checkpoint, registry, k: int = 16) -> dict:
    per_member_means = []
    real_model_calls = 0
    for member_index in range(member_count(planner.model)):
        records = planner.run_episodes(
            checkpoint,
            controller="model",
            execution="model",
            noise_registry=registry,
            stream="ensemble-variance",
            checkpoint_id=checkpoint.checkpoint_id,
            k=k,
            member_index=member_index,
        )
        per_member_means.append(float(returns(records).mean()))
        real_model_calls += sum(record.model_calls for record in records)
    return {
        "ensemble_return_variance": float(np.var(per_member_means, ddof=1)),
        "ensemble_member_returns": per_member_means,
        "ensemble_model_calls": int(real_model_calls),
    }


def one_step_error(model, records, registry, checkpoint, state_scale=None) -> dict:
    scale = np.ones(4) if state_scale is None else np.asarray(state_scale, dtype=float)
    squared_errors = []
    split_errors = [[], []]
    for episode_index, record in enumerate(records):
        for time_step, (state, action) in enumerate(zip(record.states[:-1], record.actions)):
            canonical_noise = registry.normal(
                "heldout-return",
                (checkpoint.checkpoint_id, episode_index, time_step),
                shape=(4,),
            )
            predictions = np.stack([
                model.predict_next(state, action, canonical_noise, member_index=member)
                for member in range(member_count(model))
            ])
            actual = np.asarray(record.states[time_step + 1])
            error = np.sum((predictions - actual) ** 2 * scale**2, axis=-1).mean()
            squared_errors.append(error)
            split_errors[episode_index % 2].append(error)
    return {
        "one_step_error": float(np.mean(squared_errors)) if squared_errors else 0.0,
        "one_step_error_a": float(np.mean(split_errors[0])) if split_errors[0] else 0.0,
        "one_step_error_b": float(np.mean(split_errors[1])) if split_errors[1] else 0.0,
    }


def evaluate_paid_scores(env, planner, checkpoint, registry, k: int) -> dict:
    model_records = planner.run_episodes(
        checkpoint,
        controller="model",
        execution="environment",
        noise_registry=registry,
        stream="metric-evaluation",
        checkpoint_id=checkpoint.checkpoint_id,
        k=2 * k,
    )
    reference_records = planner.run_episodes(
        checkpoint,
        controller="reference",
        execution="environment",
        noise_registry=registry,
        stream="metric-evaluation",
        checkpoint_id=checkpoint.checkpoint_id,
        k=k,
    )
    imagined_records = planner.run_episodes(
        checkpoint,
        controller="model",
        execution="model",
        noise_registry=registry,
        stream="metric-evaluation",
        checkpoint_id=checkpoint.checkpoint_id,
        k=2 * k,
    )
    heldout_records = planner.run_episodes(
        checkpoint,
        controller="model",
        execution="environment",
        noise_registry=registry,
        stream="heldout-return",
        checkpoint_id=checkpoint.checkpoint_id,
        k=2 * k,
    )

    model_values = returns(model_records)
    reference_values = returns(reference_records)
    imagined_values = returns(imagined_records)
    heldout_values = returns(heldout_records)
    csr_values = reference_values - model_values[:k]
    peg_values = imagined_values - model_values
    one_step = one_step_error(planner.model, heldout_records, registry, checkpoint)

    return {
        "k": k,
        "csr": float(csr_values.mean()),
        "csr_a": float(csr_values[::2].mean()),
        "csr_b": float(csr_values[1::2].mean()),
        "peg": float(peg_values.mean()),
        "peg_a": float(peg_values[::2].mean()),
        "peg_b": float(peg_values[1::2].mean()),
        "heldout_return": float(heldout_values.mean()),
        "heldout_return_a": float(heldout_values[::2].mean()),
        "heldout_return_b": float(heldout_values[1::2].mean()),
        **one_step,
        "metric_real_environment_steps": int(sum(record.real_environment_steps for record in model_records + reference_records + heldout_records)),
        "metric_model_calls": int(sum(record.model_calls for record in model_records + reference_records + imagined_records + heldout_records)),
    }


def candidate_coverage(env, planner, checkpoint, registry) -> dict:
    actions = planner.candidate_actions(
        registry,
        checkpoint.checkpoint_id,
        episode_id=0,
        time_step=0,
    )
    candidates, horizon = actions.shape[:2]
    replicates = planner.selection_replicates
    repeated_actions = np.repeat(actions, replicates, axis=0)
    canonical_noises = registry.normal(
        "planner-selection",
        (checkpoint.checkpoint_id, 0, 0),
        shape=(candidates, replicates, horizon, 4),
    ).reshape(candidates * replicates, horizon, 4)
    factor = env.environment_factor(checkpoint)
    states = np.repeat(np.asarray(checkpoint.state)[None, :], candidates * replicates, axis=0)
    collisions = np.zeros(candidates * replicates, dtype=bool)

    for time_step in range(horizon):
        result = env.transition_many(
            states,
            repeated_actions[:, time_step],
            canonical_noises[:, time_step],
            factor,
            checkpoint,
            include_wall=True,
        )
        states, _, terminal, collision, _ = transition_parts(result)
        collisions |= collision
        states = np.asarray(states)

    safe_by_replicate = ~collisions.reshape(candidates, replicates)
    strictly_safe = safe_by_replicate.all(axis=1)
    geometrically_feasible = False
    detour = env.geometric_detour_actions(checkpoint, horizon=horizon)
    if detour is not None:
        states = np.asarray(checkpoint.state)[None, :]
        detour_collision = False
        for time_step in range(len(detour)):
            result = env.transition_many(
                states,
                detour[time_step:time_step + 1],
                np.zeros((1, 4)),
                factor,
                checkpoint,
                include_wall=True,
            )
            states, _, _, collision, _ = transition_parts(result)
            detour_collision |= bool(np.asarray(collision)[0])
        geometrically_feasible = not detour_collision

    return {
        "candidate_count": int(candidates),
        "strictly_collision_free_candidates": int(strictly_safe.sum()),
        "candidate_starved": bool(strictly_safe.sum() == 0),
        "geometrically_feasible": bool(geometrically_feasible),
    }


def evaluate_checkpoint(env, planner, checkpoint, registry, k: int, target_episodes: int = 64) -> dict:
    row = checkpoint_metadata(checkpoint)
    row.update(evaluate_target(env, planner, checkpoint, registry, target_episodes))
    row.update(evaluate_free_scores(env, planner, checkpoint, registry))
    row.update(candidate_coverage(env, planner, checkpoint, registry))
    row.update(evaluate_paid_scores(env, planner, checkpoint, registry, k))
    return row
