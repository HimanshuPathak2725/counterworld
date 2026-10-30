"""Development-only pilot runner. No confirmatory execution lives here."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from .artifacts import ArtifactWriter
from .baselines import StateFeatureRidge
from .datasets import make_checkpoint_panel, make_training_dataset, stratified_subset, with_gap_width
from .environment import DoubleIntegratorGapEnv
from .metrics import candidate_coverage, evaluate_free_scores, evaluate_paid_scores, evaluate_target
from .models import RidgeEnsemble
from .noise import NoiseRegistry
from .parallel import map_tasks
from .planner import RandomShootingMPC
from .protocol import PilotConfig
from .statistics import (
    required_cluster_count,
    spearman,
    split_half_reliability,
    target_quality_report,
)


def build_planner(env, seed: int, config: PilotConfig):
    dataset = make_training_dataset(
        env,
        root_seed=config.training_seed + seed,
        episodes=config.training_episodes,
        max_steps=config.training_steps,
    )
    model = RidgeEnsemble.fit(
        dataset,
        train_episode_ids=np.arange(52),
        holdout_episode_ids=np.arange(52, 64),
        n_members=config.ensemble_members,
        seed=config.training_seed + seed,
        ridge=config.ridge,
    )
    return RandomShootingMPC(
        env,
        model,
        n_candidates=config.candidate_count,
        horizon=config.horizon,
        gamma=config.gamma,
        selection_replicates=config.selection_replicates,
    )


def grouped(rows, *keys):
    result = defaultdict(list)
    for row in rows:
        group_key = tuple(row[key] for key in keys)
        result[group_key[0] if len(group_key) == 1 else group_key].append(row)
    return result


def score_arrays(rows, score: str, target: str = "collision_probability"):
    by_seed = grouped(rows, "model_seed")
    score_rows = []
    target_rows = []
    for _, seed_rows in sorted(by_seed.items()):
        ordered = sorted(seed_rows, key=lambda row: row["checkpoint_id"])
        score_rows.append([float(row[score]) for row in ordered])
        target_rows.append([float(row[target]) for row in ordered])
    return np.asarray(score_rows), np.asarray(target_rows)


def mean_spearman(score_rows, target_rows):
    correlations = []
    constants = 0
    for score, target in zip(score_rows, target_rows):
        correlation = spearman(score, target)
        if np.isnan(correlation):
            correlation = 0.0
            constants += 1
        correlations.append(correlation)
    return float(np.mean(correlations)), np.asarray(correlations), constants


def target_summaries(rows) -> dict:
    summaries = {}
    for regime, regime_rows in sorted(grouped(rows, "regime").items()):
        probabilities = np.asarray([row["collision_probability"] for row in regime_rows])
        summaries[regime] = target_quality_report(probabilities)
        reliability = split_half_reliability(
            np.asarray([row["collision_probability_a"] for row in regime_rows]),
            np.asarray([row["collision_probability_b"] for row in regime_rows]),
        )
        summaries[regime]["split_half"] = reliability
        summaries[regime]["checkpoints"] = len(regime_rows)
    return summaries


def sweep_summaries(rows) -> dict:
    summaries = {}
    for (regime, k), group in sorted(grouped(rows, "regime", "k").items()):
        result = {"checkpoints": len(group)}
        for score in ["csr", "heldout_return", "peg"]:
            result[f"{score}_reliability"] = split_half_reliability(
                np.asarray([row[f"{score}_a"] for row in group]),
                np.asarray([row[f"{score}_b"] for row in group]),
            )
        summaries[f"{regime}/k={k}"] = result
    return summaries


def choose_primary_k(rows, config: PilotConfig) -> int:
    candidates = []
    for k in config.k_values:
        subset = [row for row in rows if row["k"] == k]
        if not subset:
            continue
        reliabilities = []
        for regime in sorted({row["regime"] for row in subset}):
            regime_rows = [row for row in subset if row["regime"] == regime]
            for score in ["csr", "heldout_return"]:
                report = split_half_reliability(
                    np.asarray([row[f"{score}_a"] for row in regime_rows]),
                    np.asarray([row[f"{score}_b"] for row in regime_rows]),
                )
                if isinstance(report, dict):
                    reliabilities.append(float(report.get("mean", report.get("reliability", 0.0))))
                else:
                    reliabilities.append(float(report))
        minimum = min(reliabilities, default=0.0)
        if minimum >= config.score_reliability_threshold:
            candidates.append((k, minimum))
    if not candidates:
        return max(config.k_values)
    return min(candidates, key=lambda item: (item[0], -item[1]))[0]


def power_summary(rows, config: PilotConfig, k: int) -> dict:
    subset = [row for row in rows if row["k"] == k]
    baseline_scores = {
        "ensemble_return_variance": "ensemble_return_variance",
        "one_step_error": "one_step_error",
        "heldout_return": "heldout_return",
        "state_feature_baseline": "state_feature_baseline",
    }
    report = {"k": k, "regimes": {}}
    for regime in sorted({row["regime"] for row in subset}):
        regime_rows = [row for row in subset if row["regime"] == regime]
        csr_rows, target_rows = score_arrays(regime_rows, "csr")
        csr_mean, _, _ = mean_spearman(csr_rows, target_rows)
        baseline_reports = {}
        for name, key in baseline_scores.items():
            score_rows, _ = score_arrays(regime_rows, key)
            if name == "heldout_return":
                score_rows = -score_rows
            baseline_mean, _, _ = mean_spearman(score_rows, target_rows)
            per_seed_deltas = []
            for seed_index in range(len(csr_rows)):
                csr_correlation = spearman(csr_rows[seed_index], target_rows[seed_index])
                baseline_correlation = spearman(score_rows[seed_index], target_rows[seed_index])
                csr_correlation = 0.0 if np.isnan(csr_correlation) else csr_correlation
                baseline_correlation = 0.0 if np.isnan(baseline_correlation) else baseline_correlation
                per_seed_deltas.append(csr_correlation - baseline_correlation)
            per_seed_deltas = np.asarray(per_seed_deltas)
            mean_delta = float(per_seed_deltas.mean())
            sd_delta = float(per_seed_deltas.std(ddof=1)) if len(per_seed_deltas) > 1 else 0.0
            required_result = required_cluster_count(
                mean_delta,
                sd_delta,
                margin=config.margin,
                alpha=config.alpha,
                n_baselines=len(baseline_scores),
                power=config.power,
                cap=config.seed_cap,
            )
            required = required_result["n_required"]
            baseline_reports[name] = {
                "baseline_mean_spearman": baseline_mean,
                "csr_mean_spearman": csr_mean,
                "mean_delta": mean_delta,
                "sd_delta": sd_delta,
                "required_n": required,
                "required_n_uncapped": required_result["n_uncapped"],
                "within_cap": bool(np.isfinite(required) and required <= config.seed_cap),
            }
        report["regimes"][regime] = baseline_reports
    report["required_n_max"] = max(
        value["required_n"]
        for regime in report["regimes"].values()
        for value in regime.values()
    )
    return report


def run_pilot(
    stage: str,
    output: str | Path,
    config: PilotConfig,
    seeds=None,
    max_checkpoints: int | None = None,
    k_values=None,
    primary_k: int | None = None,
    gap_width: float = 0.6,
    workers: int | None = None,
    random_goals: bool = False,
    r0_speed_range: tuple[float, float] | None = None,
):
    if stage not in {"targets", "sweep", "power", "all"}:
        raise ValueError(f"Unknown stage: {stage}")
    if workers is not None:
        config = replace(config, workers=workers)
    selected_seeds = list(config.development_seeds if seeds is None else seeds)
    selected_k = list(config.k_values if k_values is None else k_values)
    if stage == "power":
        selected_k = [primary_k or selected_k[-1]]
    writer = ArtifactWriter(output)
    env = DoubleIntegratorGapEnv()
    registry = NoiseRegistry(config.noise_seed)
    panels = {
        regime: with_gap_width(
            make_checkpoint_panel(
                regime,
                config.panel_seed,
                random_goals=random_goals,
                r0_speed_range=r0_speed_range,
            ),
            gap_width,
        )
        for regime in ["R0", "R1", "R2"]
    }
    all_rows = []

    for seed in selected_seeds:
        planner = build_planner(env, seed, config)
        seed_rows = []
        tasks = []
        for regime, panel in panels.items():
            subset = stratified_subset(panel, max_checkpoints or len(panel))
            sweep_subset = stratified_subset(subset, min(config.k_sweep_checkpoint_count, len(subset)))
            sweep_ids = {item.checkpoint_id for item in sweep_subset}
            for checkpoint in subset:
                tasks.append(("target", checkpoint, None))
                if stage in {"sweep", "power", "all"}:
                    tasks.append(("free", checkpoint, None))
                    tasks.append(("coverage", checkpoint, None))
                    should_score = (
                        stage == "all"
                        or (stage == "sweep" and seed in config.k_sweep_seeds)
                        or stage == "power"
                    )
                    if should_score and checkpoint.checkpoint_id in sweep_ids:
                        for k in selected_k:
                            tasks.append(("paid", checkpoint, k))
        results = map_tasks(planner, env, registry, tasks, config.workers, config.target_episodes)
        base_rows = {}
        for kind, checkpoint_id, k, metadata, result in results:
            key = (metadata["regime"], checkpoint_id)
            row = base_rows.setdefault(key, {
                "model_seed": seed,
                **metadata,
                "noise_scale": metadata["sigma_v"],
            })
            row.update(result)
            if kind == "paid":
                seed_rows.append(row.copy())
                all_rows.append(row.copy())
        for row in base_rows.values():
            if "k" not in row:
                seed_rows.append(row)
                all_rows.append(row)
        writer.write_jsonl(f"{stage}_seed_{seed}.jsonl", seed_rows)

    if stage == "targets":
        summary = target_summaries(all_rows)
        writer.write_json("target_quality.json", summary)
    elif stage == "sweep":
        writer.write_json("sweep_summary.json", sweep_summaries(all_rows))
        if primary_k is None:
            primary_k = choose_primary_k(all_rows, config)
        writer.write_json("primary_k.json", {"k": primary_k})
    elif stage == "power":
        fit_rows = [row for row in all_rows if row.get("k") == (primary_k or selected_k[-1])]
        baseline = StateFeatureRidge()
        baseline.fit(
            fit_rows,
            np.asarray([row["collision_probability"] for row in fit_rows]),
            np.asarray([row["model_seed"] for row in fit_rows]),
        )
        predictions = baseline.predict(fit_rows)
        for row, prediction in zip(fit_rows, predictions):
            row["state_feature_baseline"] = float(prediction)
        writer.write_json("state_feature_baseline.json", {
            "coefficients": getattr(baseline, "coef_", None),
            "lambda": getattr(baseline, "lambda_", None),
        })
        writer.write_json("power_analysis.json", power_summary(fit_rows, config, primary_k or selected_k[-1]))
    else:
        writer.write_json("target_quality.json", target_summaries(all_rows))
        writer.write_json("sweep_summary.json", sweep_summaries(all_rows))

    configuration = config.as_dict()
    configuration.update({
        "stage": stage,
        "seeds": selected_seeds,
        "max_checkpoints": max_checkpoints,
        "k_values": selected_k,
        "primary_k": primary_k,
        "gap_width": gap_width,
        "random_goals": random_goals,
        "r0_speed_range": r0_speed_range,
        "development_only": True,
    })
    writer.finalize(configuration)
    return all_rows


def main():
    parser = argparse.ArgumentParser(description="Run development-only Counterworld pilots")
    parser.add_argument("--stage", choices=["targets", "sweep", "power", "all"], default="targets")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default=None, help="Comma-separated development seed ids for a smoke run")
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument("--k-values", default=None)
    parser.add_argument("--primary-k", type=int, default=None)
    parser.add_argument("--gap-width", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--random-goals", action="store_true")
    parser.add_argument("--r0-speed-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"))
    args = parser.parse_args()
    seeds = None if args.seeds is None else [int(value) for value in args.seeds.split(",")]
    k_values = None if args.k_values is None else tuple(int(value) for value in args.k_values.split(","))
    r0_speed_range = None if args.r0_speed_range is None else tuple(args.r0_speed_range)
    run_pilot(
        stage=args.stage,
        output=args.output,
        config=PilotConfig(),
        seeds=seeds,
        max_checkpoints=args.max_checkpoints,
        k_values=k_values,
        primary_k=args.primary_k,
        gap_width=args.gap_width,
        workers=args.workers,
        random_goals=args.random_goals,
        r0_speed_range=r0_speed_range,
    )


if __name__ == "__main__":
    main()
