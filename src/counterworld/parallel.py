"""Deterministic checkpoint-level parallel execution for development pilots."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from .metrics import (
    candidate_coverage,
    checkpoint_metadata,
    evaluate_free_scores,
    evaluate_paid_scores,
    evaluate_target,
)


_PLANNER = None
_ENV = None
_REGISTRY = None
_TARGET_EPISODES = 64


def initialize_worker(planner, env, registry, target_episodes):
    global _PLANNER, _ENV, _REGISTRY, _TARGET_EPISODES
    _PLANNER = planner
    _ENV = env
    _REGISTRY = registry
    _TARGET_EPISODES = target_episodes


def run_task(task):
    kind, checkpoint, k = task
    if kind == "target":
        result = evaluate_target(_ENV, _PLANNER, checkpoint, _REGISTRY, _TARGET_EPISODES)
    elif kind == "free":
        result = evaluate_free_scores(_ENV, _PLANNER, checkpoint, _REGISTRY)
    elif kind == "coverage":
        result = candidate_coverage(_ENV, _PLANNER, checkpoint, _REGISTRY)
    elif kind == "paid":
        result = evaluate_paid_scores(_ENV, _PLANNER, checkpoint, _REGISTRY, k)
    else:
        raise ValueError(f"Unknown task kind: {kind}")
    return kind, checkpoint.checkpoint_id, k, checkpoint_metadata(checkpoint), result


def map_tasks(planner, env, registry, tasks, workers: int, target_episodes: int):
    if not tasks:
        return []
    if workers <= 1:
        initialize_worker(planner, env, registry, target_episodes)
        return [run_task(task) for task in tasks]
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=initialize_worker,
        initargs=(planner, env, registry, target_episodes),
    ) as pool:
        return list(pool.map(run_task, tasks, chunksize=1))
