"""Build the immutable Counterworld showcase evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from counterworld.artifacts import file_hash
from counterworld.datasets import make_checkpoint_panel
from counterworld.environment import DoubleIntegratorGapEnv
from counterworld.noise import NoiseRegistry
from counterworld.pilot import build_planner
from counterworld.protocol import PilotConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "showcase"
ARTIFACTS = ROOT / "artifacts"


def read_json(path):
    return json.loads(path.read_text())


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rounded(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, np.ndarray):
        return np.round(value, 6).tolist()
    if isinstance(value, list):
        return [rounded(item) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def record(record):
    return {
        "states": rounded(record.states),
        "actions": rounded(record.actions),
        "rewards": rounded(record.rewards),
        "total_return": round(record.total_return, 6),
        "collision": record.collision,
        "goal": record.goal,
        "real_environment_steps": record.real_environment_steps,
        "model_calls": record.model_calls,
        "selected_indices": record.selected_indices.tolist(),
    }


def main():
    original_path = ARTIFACTS / "development-targets" / "target_quality.json"
    amended_path = ARTIFACTS / "development-targets-r0-broadened" / "target_quality.json"
    sweep_path = ARTIFACTS / "development-sweep-r0-broadened" / "sweep_summary.json"
    power_path = ARTIFACTS / "development-power-r0-broadened-k16" / "power_analysis.json"
    original = read_json(original_path)
    amended = read_json(amended_path)
    sweep = read_json(sweep_path)
    power = read_json(power_path)

    power_rows = []
    for path in sorted((ARTIFACTS / "development-power-r0-broadened-k16").glob("power_seed_*.jsonl")):
        power_rows.extend(read_jsonl(path))
    regime = "R1"
    median_probability = float(np.median([row["collision_probability"] for row in power_rows if row["regime"] == regime]))
    candidates = [row for row in power_rows if row["regime"] == regime and row["model_seed"] == 0]
    selected = min(candidates, key=lambda row: (abs(row["collision_probability"] - median_probability), row["checkpoint_id"]))

    config = PilotConfig()
    env = DoubleIntegratorGapEnv()
    planner = build_planner(env, 0, config)
    checkpoint = make_checkpoint_panel(regime, config.panel_seed)[selected["checkpoint_id"] - 1000]
    registry = NoiseRegistry(config.noise_seed)
    real = planner.run_episode(checkpoint, "model", "environment", registry, "metric-evaluation", checkpoint.checkpoint_id, 0)
    imagined = planner.run_episode(checkpoint, "model", "model", registry, "metric-evaluation", checkpoint.checkpoint_id, 0)
    reference = planner.run_episode(checkpoint, "reference", "environment", registry, "metric-evaluation", checkpoint.checkpoint_id, 0)

    evidence = {
        "title": "When the Map Is Wrong: Why the Better Warning Sign Lost",
        "status": "Development-only negative result",
        "source_artifacts": {
            f"{path.parent.name}/{path.name}": {"path": str(path), "sha256": file_hash(path)}
            for path in [original_path, amended_path, sweep_path, power_path]
        },
        "target_original": original,
        "target_amended": amended,
        "sweep": sweep,
        "power": power,
        "selection": {
            "rule": "Among model-seed-0 R1 checkpoints, choose the collision probability nearest to the median of all 16 R1 model seeds; break ties by lowest checkpoint id.",
            "median_collision_probability": median_probability,
            "checkpoint_id": checkpoint.checkpoint_id,
            "state": rounded(checkpoint.state),
            "goal": rounded(checkpoint.goal),
            "gap_center": checkpoint.gap_center,
            "gap_width": checkpoint.gap_width,
            "sigma_p": checkpoint.sigma_p,
            "sigma_v": checkpoint.sigma_v,
        },
        "trajectories": {
            "model_in_environment": record(real),
            "model_in_imagination": record(imagined),
            "reference_in_environment": record(reference),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    (OUT / "assets" / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    (OUT / "assets" / "evidence.js").write_text("window.EVIDENCE = " + json.dumps(evidence, separators=(",", ":")) + ";\n")
    print(json.dumps(evidence["selection"], indent=2))
    print("Real collision:", real.collision, "Imagined collision:", imagined.collision, "Reference collision:", reference.collision)


if __name__ == "__main__":
    main()
