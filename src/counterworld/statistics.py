"""NumPy-only statistics for crossed, seed-clustered Counterworld panels."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import NormalDist
from typing import Any

import numpy as np


def midrank(values: Any) -> np.ndarray:
    """Return one-based average ranks, assigning equal ranks to equal values."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("midrank does not accept NaN or infinite values")
    _, inverse, counts = np.unique(
        array, return_inverse=True, return_counts=True
    )
    cumulative = np.cumsum(counts)
    average_ranks = cumulative - (counts - 1) / 2.0
    return average_ranks[inverse].astype(float, copy=False)


def _paired_1d(
    x: Any, y: Any, nan_policy: str
) -> tuple[np.ndarray, np.ndarray]:
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1:
        raise ValueError("correlation inputs must be one-dimensional")
    if x_array.size == 0 or x_array.shape != y_array.shape:
        raise ValueError("correlation inputs must be non-empty and have equal shape")
    missing = np.isnan(x_array) | np.isnan(y_array)
    if np.any(missing):
        if nan_policy == "raise":
            raise ValueError("correlation inputs contain NaN; use nan_policy='omit'")
        x_array = x_array[~missing]
        y_array = y_array[~missing]
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("correlation inputs must be finite after NaN handling")
    return x_array, y_array


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denominator = float(
        np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0.0:
        return float("nan")
    correlation = float(np.dot(x_centered, y_centered) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))


def spearman(x: Any, y: Any, *, nan_policy: str = "raise") -> float:
    """Return Pearson correlation of midranks; constant vectors give NaN."""
    x_array, y_array = _paired_1d(x, y, nan_policy)
    if x_array.size < 2:
        return float("nan")
    return _pearson(midrank(x_array), midrank(y_array))


def spearman_with_constant_policy(
    x: Any, y: Any, *, nan_policy: str = "raise"
) -> tuple[float, bool]:
    """Return ``(correlation, constant_flag)``, mapping undefined cases to zero."""
    x_array, y_array = _paired_1d(x, y, nan_policy)
    if x_array.size < 2:
        return 0.0, True
    correlation = _pearson(midrank(x_array), midrank(y_array))
    if not np.isfinite(correlation):
        return 0.0, True
    return correlation, False


def _row_inputs(
    score_rows: Any, target_rows: Any, nan_policy: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    scores = np.asarray(score_rows, dtype=float)
    targets = np.asarray(target_rows, dtype=float)
    if scores.shape != targets.shape or scores.ndim not in {2, 3}:
        raise ValueError("score_rows and target_rows must have equal 2-D or 3-D shapes")
    if scores.shape[0] == 0 or scores.shape[-1] == 0:
        raise ValueError("row and checkpoint dimensions must be non-empty")
    two_dimensional = scores.ndim == 2
    if two_dimensional:
        scores = scores[:, None, :]
        targets = targets[:, None, :]
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    return scores, targets, two_dimensional


def per_seed_spearman(
    score_rows: Any, target_rows: Any, *, nan_policy: str = "raise"
) -> dict[str, Any]:
    """Correlate checkpoints within each row, never after pooling rows.

    Two-dimensional input is ``(seed/regime row, checkpoint)`` and has a scalar
    mean. Three-dimensional input is ``(seed, regime, checkpoint)`` and has one
    mean per regime. The primary values use the zero-for-constant convention;
    mathematical values retain NaN for sensitivity analyses.
    """
    scores, targets, two_dimensional = _row_inputs(
        score_rows, target_rows, nan_policy
    )
    shape = scores.shape[:2]
    mathematical = np.empty(shape, dtype=float)
    conventional = np.empty(shape, dtype=float)
    constant_mask = np.zeros(shape, dtype=bool)
    for index in np.ndindex(shape):
        x, y = _paired_1d(scores[index], targets[index], nan_policy)
        if x.size < 2:
            correlation = float("nan")
        else:
            correlation = _pearson(midrank(x), midrank(y))
        mathematical[index] = correlation
        if np.isfinite(correlation):
            conventional[index] = correlation
        else:
            conventional[index] = 0.0
            constant_mask[index] = True
    if two_dimensional:
        mathematical = mathematical[:, 0]
        conventional = conventional[:, 0]
        constant_mask = constant_mask[:, 0]
        mean: float | np.ndarray = float(np.mean(conventional))
    else:
        mean = np.mean(conventional, axis=0)
    return {
        "per_seed": conventional,
        "mean": mean,
        "mathematical_per_seed": mathematical,
        "constant_mask": constant_mask,
        "constant_count": int(np.count_nonzero(constant_mask)),
    }


def within_seed_deltas(
    score_rows: Any,
    baseline_rows: Any,
    target_rows: Any,
    *,
    nan_policy: str = "raise",
) -> dict[str, Any]:
    """Subtract baseline correlations from score correlations within each row."""
    score = per_seed_spearman(score_rows, target_rows, nan_policy=nan_policy)
    baseline = per_seed_spearman(baseline_rows, target_rows, nan_policy=nan_policy)
    deltas = score["per_seed"] - baseline["per_seed"]
    mean = np.mean(deltas) if deltas.ndim == 1 else np.mean(deltas, axis=0)
    return {
        "per_seed": deltas,
        "mean": float(mean) if np.ndim(mean) == 0 else mean,
        "mathematical_per_seed": (
            score["mathematical_per_seed"] - baseline["mathematical_per_seed"]
        ),
        "constant_mask": score["constant_mask"] | baseline["constant_mask"],
        "constant_count": int(score["constant_count"] + baseline["constant_count"]),
        "score": score,
        "baseline": baseline,
    }


def mean_with_bootstrap_ci(
    delta_rows: Any,
    confidence: float = 0.9875,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> dict[str, Any]:
    """Return a one-sided percentile lower bound from whole-cluster resamples.

    The first axis identifies model-seed clusters. Every trailing regime or
    checkpoint value moves with its cluster; only whole first-axis rows are
    resampled. A result from :func:`within_seed_deltas` may be passed directly.
    """
    if isinstance(delta_rows, Mapping):
        if "per_seed" not in delta_rows:
            raise ValueError("a delta result mapping must contain 'per_seed'")
        delta_rows = delta_rows["per_seed"]
    deltas = np.asarray(delta_rows, dtype=float)
    if deltas.ndim == 0 or deltas.shape[0] == 0:
        raise ValueError("delta_rows must have a non-empty model-seed first axis")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("delta_rows must contain only finite values")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not isinstance(n_boot, (int, np.integer)) or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")
    if seed is None:
        raise ValueError("an explicit bootstrap seed is required")
    rng = np.random.default_rng(seed)
    n_clusters = deltas.shape[0]
    bootstrap_means = np.empty((int(n_boot),) + deltas.shape[1:], dtype=float)
    for start in range(0, int(n_boot), 1_000):
        stop = min(start + 1_000, int(n_boot))
        indices = rng.integers(0, n_clusters, size=(stop - start, n_clusters))
        bootstrap_means[start:stop] = np.mean(deltas[indices], axis=1)
    mean = np.mean(deltas, axis=0)
    lower = np.quantile(bootstrap_means, 1.0 - confidence, axis=0)
    return {
        "mean": float(mean) if np.ndim(mean) == 0 else mean,
        "lower": float(lower) if np.ndim(lower) == 0 else lower,
        "bootstrap_means": bootstrap_means,
        "confidence": float(confidence),
        "n_boot": int(n_boot),
        "seed": seed,
        "n_clusters": n_clusters,
    }


def split_half_reliability(
    prob_a: Any, prob_b: Any, axis: int = -1, *, nan_policy: str = "raise"
) -> dict[str, Any]:
    """Compute Pearson split-half r and Spearman-Brown reliability per seed.

    The checkpoint axis is moved to the final position. Leading axes are kept;
    for a ``(seed, regime, checkpoint)`` input, means are taken over seeds only.
    A constant or otherwise undefined half pair receives zero reliability.
    """
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    if a.shape != b.shape or a.ndim == 0:
        raise ValueError("prob_a and prob_b must have equal non-scalar shapes")
    a = np.moveaxis(a, axis, -1)
    b = np.moveaxis(b, axis, -1)
    if a.shape[-1] == 0:
        raise ValueError("the checkpoint axis must be non-empty")
    leading_shape = a.shape[:-1]
    flat_a = a.reshape((-1, a.shape[-1]))
    flat_b = b.reshape((-1, b.shape[-1]))
    mathematical_r = np.empty(flat_a.shape[0], dtype=float)
    conventional_r = np.empty(flat_a.shape[0], dtype=float)
    reliability = np.empty(flat_a.shape[0], dtype=float)
    constant_mask = np.zeros(flat_a.shape[0], dtype=bool)
    for index in range(flat_a.shape[0]):
        x, y = _paired_1d(flat_a[index], flat_b[index], nan_policy)
        r = _pearson(x, y)
        mathematical_r[index] = r
        if np.isfinite(r):
            conventional_r[index] = r
            denominator = 1.0 + r
            reliability[index] = 2.0 * r / denominator if denominator != 0.0 else 0.0
            if denominator == 0.0:
                constant_mask[index] = True
        else:
            conventional_r[index] = 0.0
            reliability[index] = 0.0
            constant_mask[index] = True
    per_seed = reliability.reshape(leading_shape)
    mean = np.mean(per_seed) if per_seed.ndim <= 1 else np.mean(per_seed, axis=0)
    return {
        "per_seed": per_seed,
        "mean": float(mean) if np.ndim(mean) == 0 else mean,
        "pearson_r": conventional_r.reshape(leading_shape),
        "mathematical_pearson_r": mathematical_r.reshape(leading_shape),
        "constant_mask": constant_mask.reshape(leading_shape),
        "constant_count": int(np.count_nonzero(constant_mask)),
    }


def required_cluster_count(
    mean_delta: float,
    sd_delta: float,
    margin: float = 0.10,
    alpha: float = 0.05,
    n_baselines: int = 4,
    power: float = 0.80,
    shrink_effect: float = 0.8,
    inflate_sd: float = 1.2,
    cap: int | None = 200,
) -> dict[str, Any]:
    """Apply the pre-registered normal-approximation sample-size rule.

    ``n_required`` is an integer within the cap and infinity otherwise. The
    uncapped calculation and every planning input are retained in the result.
    """
    scalar_inputs = {
        "mean_delta": mean_delta,
        "sd_delta": sd_delta,
        "margin": margin,
        "alpha": alpha,
        "power": power,
        "shrink_effect": shrink_effect,
        "inflate_sd": inflate_sd,
    }
    if not all(np.isfinite(value) for value in scalar_inputs.values()):
        raise ValueError("planning inputs must be finite")
    if sd_delta < 0.0 or inflate_sd < 0.0 or shrink_effect <= 0.0:
        raise ValueError("SDs must be nonnegative and shrink_effect must be positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie strictly between zero and one")
    if not isinstance(n_baselines, (int, np.integer)) or n_baselines <= 0:
        raise ValueError("n_baselines must be a positive integer")
    if alpha / n_baselines >= 1.0:
        raise ValueError("alpha / n_baselines must be less than one")
    if cap is not None and (
        not isinstance(cap, (int, np.integer)) or cap <= 0
    ):
        raise ValueError("cap must be a positive integer or None")
    z_alpha = float(NormalDist().inv_cdf(1.0 - alpha / n_baselines))
    z_power = float(NormalDist().inv_cdf(power))
    planning_effect = float(shrink_effect * mean_delta)
    planning_sd = float(inflate_sd * sd_delta)
    if planning_effect <= margin:
        n_uncapped: int | float = float("inf")
    else:
        ratio = (z_alpha + z_power) * planning_sd / (planning_effect - margin)
        n_uncapped = int(np.ceil(ratio**2))
    exceeds_cap = bool(cap is not None and n_uncapped > cap)
    n_required: int | float = float("inf") if exceeds_cap else n_uncapped
    return {
        "n_required": n_required,
        "n_uncapped": n_uncapped,
        "exceeds_cap": exceeds_cap,
        "planning_effect": planning_effect,
        "planning_sd": planning_sd,
        "z_alpha": z_alpha,
        "z_power": z_power,
        "inputs": {
            "mean_delta": float(mean_delta),
            "sd_delta": float(sd_delta),
            "margin": float(margin),
            "alpha": float(alpha),
            "n_baselines": int(n_baselines),
            "power": float(power),
            "shrink_effect": float(shrink_effect),
            "inflate_sd": float(inflate_sd),
            "cap": None if cap is None else int(cap),
        },
    }


def target_quality_report(probabilities: Any) -> dict[str, Any]:
    """Report collision-target support and the three pre-registered gates.

    The interior-support gate uses the strict pre-registered 0.05--0.95 band;
    the separately reported interior fraction uses strict 0--1 endpoints.
    """
    probability = np.asarray(probabilities, dtype=float)
    if probability.size == 0:
        raise ValueError("probabilities must be non-empty")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probabilities must not contain NaN or infinite values")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    values = probability.ravel()
    zero = values == 0.0
    one = values == 1.0
    interior = (values > 0.0) & (values < 1.0)
    band = (values > 0.05) & (values < 0.95)
    endpoints = zero | one
    count = values.size
    zero_fraction = float(np.mean(zero))
    one_fraction = float(np.mean(one))
    interior_fraction = float(np.mean(interior))
    band_fraction = float(np.mean(band))
    endpoint_fraction = float(np.mean(endpoints))
    variance = float(np.var(values))
    gates = {
        "variance": bool(variance >= 0.02),
        "interior": bool(band_fraction >= 0.20),
        "endpoints": bool(endpoint_fraction <= 0.70),
    }
    gates["all"] = bool(all(gates.values()))
    return {
        "n": int(count),
        "zero_count": int(np.count_nonzero(zero)),
        "one_count": int(np.count_nonzero(one)),
        "interior_count": int(np.count_nonzero(interior)),
        "band_count": int(np.count_nonzero(band)),
        "endpoint_count": int(np.count_nonzero(endpoints)),
        "zero_fraction": zero_fraction,
        "one_fraction": one_fraction,
        "interior_fraction": interior_fraction,
        "band_fraction": band_fraction,
        "endpoint_fraction": endpoint_fraction,
        "variance": variance,
        "gates": gates,
    }


__all__ = [
    "mean_with_bootstrap_ci",
    "midrank",
    "per_seed_spearman",
    "required_cluster_count",
    "spearman",
    "spearman_with_constant_policy",
    "split_half_reliability",
    "target_quality_report",
    "within_seed_deltas",
]
