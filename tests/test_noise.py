from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from counterworld.noise import NoiseRegistry, covariance_factor


def test_named_streams_and_indices_are_reproducible() -> None:
    first = NoiseRegistry(12345)
    second = NoiseRegistry(12345)

    expected = first.rng("planner", 4, 2).standard_normal((3, 2))
    actual = second.rng("planner", 4, 2).standard_normal((3, 2))
    np.testing.assert_array_equal(actual, expected)

    normal = first.normal("outcome", (7, 1), (4, 3))
    np.testing.assert_array_equal(normal, second.normal("outcome", (7, 1), (4, 3)))
    assert not np.allclose(normal, first.normal("metric", (7, 1), (4, 3)))
    assert not np.allclose(normal, first.normal("outcome", (7, 2), (4, 3)))
    assert not np.allclose(normal, NoiseRegistry(12346).normal("outcome", (7, 1), (4, 3)))


def test_covariance_factor_clips_indefinite_eigenvalues() -> None:
    covariance = np.asarray(
        [
            [2.0, 0.5, 0.0],
            [0.5, -1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    floor = 1e-4
    factor = covariance_factor(covariance, floor=floor)

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    expected = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    np.testing.assert_allclose(factor @ factor.T, expected, atol=1e-12)
    np.testing.assert_allclose(factor, np.tril(factor), atol=1e-15)
    assert np.min(np.linalg.eigvalsh(factor @ factor.T)) >= floor - 1e-12
    assert np.all(np.diag(factor) > 0.0)


def test_covariance_factor_accepts_roundoff_asymmetry() -> None:
    covariance = np.asarray([[1.0, 0.25], [0.25 + 1e-13, 0.5]])
    factor = covariance_factor(covariance)
    symmetric = 0.5 * (covariance + covariance.T)
    np.testing.assert_allclose(factor @ factor.T, symmetric, atol=1e-12)
