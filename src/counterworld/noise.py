"""Deterministic named random streams and explicit covariance mapping."""

from __future__ import annotations

import hashlib

import numpy as np


class NoiseRegistry:
    """Recreate independent NumPy generators from stable stream identifiers."""

    _DOMAIN = b"counterworld.noise.v1"

    def __init__(self, root_seed: int | str | bytes) -> None:
        self.root_seed = root_seed
        self._encoded_root_seed = self._encode_root_seed(root_seed)

    @staticmethod
    def _encode_root_seed(root_seed: int | str | bytes) -> bytes:
        if isinstance(root_seed, (int, np.integer)):
            return b"integer:" + str(int(root_seed)).encode("ascii")
        if isinstance(root_seed, str):
            return b"string:" + root_seed.encode("utf-8")
        if isinstance(root_seed, bytes):
            return b"bytes:" + root_seed
        raise TypeError("root_seed must be an int, str, or bytes")

    @staticmethod
    def _field(value: bytes) -> bytes:
        return len(value).to_bytes(8, "big") + value

    def rng(self, stream: str, *indices: int) -> np.random.Generator:
        """Return a fresh generator determined only by the seed and identifiers."""
        if not isinstance(stream, str):
            raise TypeError("stream must be a str")
        fields = [self._DOMAIN, self._field(self._encoded_root_seed), self._field(stream.encode("utf-8"))]
        for index in indices:
            if not isinstance(index, (int, np.integer)):
                raise TypeError("stream indices must be integers")
            fields.append(self._field(str(int(index)).encode("ascii")))
        digest = hashlib.sha256(b"".join(fields)).digest()
        seed_sequence = np.random.SeedSequence(int.from_bytes(digest, "big"))
        return np.random.default_rng(seed_sequence)

    def normal(
        self,
        stream: str,
        indices: tuple[int, ...] | list[int],
        shape: int | tuple[int, ...],
    ) -> np.ndarray:
        """Draw standard-normal values from a reproducibly identified stream."""
        return self.rng(stream, *indices).standard_normal(shape)


def covariance_factor(covariance: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Return a lower Cholesky factor after clipping symmetric eigenvalues."""
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("covariance must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must contain only finite values")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    if not np.allclose(matrix, matrix.T, rtol=1e-7, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(floor))
    clipped_covariance = (eigenvectors * clipped) @ eigenvectors.T
    clipped_covariance = 0.5 * (clipped_covariance + clipped_covariance.T)
    return np.linalg.cholesky(clipped_covariance)
