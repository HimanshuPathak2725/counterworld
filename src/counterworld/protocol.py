"""Frozen development-stage protocol choices for CW-0.2."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PilotConfig:
    development_seeds: tuple[int, ...] = tuple(range(16))
    k_sweep_seeds: tuple[int, ...] = (0, 1, 2, 3)
    checkpoint_count: int = 48
    k_sweep_checkpoint_count: int = 12
    target_episodes: int = 64
    workers: int = 16
    k_values: tuple[int, ...] = (4, 8, 16, 32)
    candidate_count: int = 512
    selection_replicates: int = 4
    horizon: int = 12
    gamma: float = 0.99
    training_episodes: int = 64
    training_steps: int = 64
    ensemble_members: int = 8
    ridge: float = 1.0
    gap_widths: tuple[float, ...] = (0.6, 0.4, 0.8)
    variance_threshold: float = 0.02
    interior_threshold: float = 0.20
    endpoint_threshold: float = 0.70
    target_reliability_threshold: float = 0.75
    score_reliability_threshold: float = 0.80
    bootstrap_samples: int = 10000
    bootstrap_confidence: float = 0.9875
    power: float = 0.80
    alpha: float = 0.05
    margin: float = 0.10
    seed_cap: int = 200
    panel_seed: int = 500
    training_seed: int = 100
    noise_seed: int = 2000

    def as_dict(self):
        result = asdict(self)
        result["development_scope"] = {
            "collision_distribution_seeds": len(self.development_seeds),
            "k_sweep_seeds": list(self.k_sweep_seeds),
            "checkpoints_per_regime": self.checkpoint_count,
            "k_sweep_checkpoints_per_regime": self.k_sweep_checkpoint_count,
            "narrowing": "Full collision distribution on all development seeds; nested paid-score sweep on four predeclared development seeds and twelve stratified checkpoints per regime. No confirmatory checkpoints or outcomes are used.",
        }
        return result
