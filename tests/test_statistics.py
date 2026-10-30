import math
from statistics import NormalDist

import numpy as np
import pytest

from counterworld.statistics import (
    mean_with_bootstrap_ci,
    midrank,
    per_seed_spearman,
    required_cluster_count,
    spearman,
    spearman_with_constant_policy,
    split_half_reliability,
    target_quality_report,
    within_seed_deltas,
)


def test_midrank_uses_average_ranks_for_every_tie():
    values = np.array([10.0, 20.0, 20.0, 30.0, 20.0])
    expected = np.array([1.0, 3.0, 3.0, 5.0, 3.0])
    np.testing.assert_array_equal(midrank(values), expected)
    np.testing.assert_array_equal(midrank(values), midrank(values.copy()))


def test_spearman_is_pearson_of_midranks_and_nan_is_explicit():
    x = np.array([1.0, 2.0, 2.0, 4.0])
    y = np.array([8.0, 7.0, 7.0, 5.0])
    expected = np.corrcoef(midrank(x), midrank(y))[0, 1]
    assert spearman(x, y) == pytest.approx(expected)
    with pytest.raises(ValueError, match="NaN"):
        spearman([1.0, np.nan], [2.0, 3.0])
    assert spearman([1.0, np.nan, 2.0], [2.0, 9.0, 3.0], nan_policy="omit") == 1.0


def test_constant_policy_is_zero_and_flagged_but_mathematical_result_is_nan():
    assert math.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    correlation, constant = spearman_with_constant_policy(
        [1.0, 1.0, 1.0], [1.0, 2.0, 3.0]
    )
    assert correlation == 0.0
    assert constant
    correlation, constant = spearman_with_constant_policy([1, 2, 3], [3, 2, 1])
    assert correlation == -1.0
    assert not constant


def test_per_seed_mean_differs_from_invalid_pooled_correlation():
    scores = np.array([[[0.0, 1.0, 2.0]], [[10.0, 11.0, 12.0]]])
    targets = np.array([[[10.0, 11.0, 12.0]], [[0.0, 1.0, 2.0]]])
    result = per_seed_spearman(scores, targets)
    np.testing.assert_allclose(result["per_seed"], [[1.0], [1.0]])
    np.testing.assert_allclose(result["mean"], [1.0])
    pooled = spearman(scores.ravel(), targets.ravel())
    assert pooled < 0.0
    assert result["mean"][0] != pooled


def test_within_seed_deltas_and_constant_counts():
    scores = np.array([[[0.0, 1.0, 2.0]], [[10.0, 11.0, 12.0]]])
    baselines = np.array([[[12.0, 11.0, 10.0]], [[2.0, 1.0, 0.0]]])
    targets = np.array([[[10.0, 11.0, 12.0]], [[0.0, 1.0, 2.0]]])
    result = within_seed_deltas(scores, baselines, targets)
    np.testing.assert_allclose(result["per_seed"], [[2.0], [2.0]])
    np.testing.assert_allclose(result["mean"], [2.0])

    constant = within_seed_deltas(scores, np.zeros_like(scores), targets)
    np.testing.assert_allclose(constant["per_seed"], [[1.0], [1.0]])
    assert constant["constant_count"] == 2


def test_cluster_bootstrap_resamples_whole_seed_rows():
    deltas = np.array([[0.0, 10.0], [2.0, 20.0], [4.0, 30.0]])
    result = mean_with_bootstrap_ci(
        deltas, confidence=0.9, n_boot=8, seed=17
    )
    rng = np.random.default_rng(17)
    indices = rng.integers(0, 3, size=(8, 3))
    expected_boot = np.mean(deltas[indices], axis=1)
    np.testing.assert_array_equal(result["bootstrap_means"], expected_boot)
    np.testing.assert_allclose(result["mean"], [2.0, 20.0])
    np.testing.assert_allclose(
        result["lower"], np.quantile(expected_boot, 0.1, axis=0)
    )
    assert result["n_clusters"] == 3


def test_split_half_reliability_is_per_seed_and_flags_constant_half():
    prob_a = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    prob_b = np.array([[2.0, 4.0, 6.0], [2.0, 3.0, 4.0]])
    result = split_half_reliability(prob_a, prob_b)
    np.testing.assert_allclose(result["pearson_r"], [1.0, 0.0])
    np.testing.assert_allclose(result["per_seed"], [1.0, 0.0])
    assert result["mean"] == pytest.approx(0.5)
    np.testing.assert_array_equal(result["constant_mask"], [False, True])
    assert result["constant_count"] == 1
    assert math.isnan(result["mathematical_pearson_r"][1])


def test_split_half_reliability_handles_perfect_anticorrelation():
    result = split_half_reliability(
        np.array([0.0, 0.25, 0.75, 1.0]),
        np.array([1.0, 0.75, 0.25, 0.0]),
    )
    assert result["pearson_r"] == -1.0
    assert result["per_seed"] == 0.0
    assert result["mean"] == 0.0
    assert result["constant_count"] == 1


def test_required_cluster_count_formula_cap_and_infinite_margin():
    result = required_cluster_count(mean_delta=0.5, sd_delta=0.1)
    z_alpha = NormalDist().inv_cdf(1.0 - 0.05 / 4)
    z_power = NormalDist().inv_cdf(0.8)
    expected = math.ceil(((z_alpha + z_power) * 1.2 * 0.1 / (0.8 * 0.5 - 0.1)) ** 2)
    assert result["n_required"] == expected
    assert result["n_uncapped"] == expected
    assert result["planning_effect"] == pytest.approx(0.4)
    assert result["planning_sd"] == pytest.approx(0.12)
    assert result["inputs"]["n_baselines"] == 4

    capped = required_cluster_count(mean_delta=0.15, sd_delta=1.0)
    assert capped["n_uncapped"] > 200
    assert math.isinf(capped["n_required"])
    assert capped["exceeds_cap"]

    impossible = required_cluster_count(mean_delta=0.12, sd_delta=0.1)
    assert impossible["planning_effect"] < impossible["inputs"]["margin"]
    assert math.isinf(impossible["n_required"])
    assert math.isinf(impossible["n_uncapped"])


def test_target_quality_fractions_strict_band_and_gates():
    probabilities = np.array([0.0, 1.0, 0.5, 0.1, 0.9, 0.05, 0.95, 0.2])
    report = target_quality_report(probabilities)
    assert report["zero_count"] == 1
    assert report["one_count"] == 1
    assert report["interior_count"] == 6
    assert report["band_count"] == 4
    assert report["endpoint_count"] == 2
    assert report["interior_fraction"] == pytest.approx(6 / 8)
    assert report["band_fraction"] == pytest.approx(4 / 8)
    assert report["variance"] == pytest.approx(np.var(probabilities))
    assert report["gates"] == {
        "variance": True,
        "interior": True,
        "endpoints": True,
        "all": True,
    }

    failed = target_quality_report(np.zeros(10))
    assert failed["gates"] == {
        "variance": False,
        "interior": False,
        "endpoints": False,
        "all": False,
    }
    with pytest.raises(ValueError, match="NaN"):
        target_quality_report([0.0, np.nan])
