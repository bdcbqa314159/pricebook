"""
Random number generation for Monte Carlo simulation.

Provides pseudo-random (numpy) and quasi-random (Sobol) generators
for standard normal variates, with reproducible seed management.

    rng = PseudoRandom(seed=42)
    z = rng.normals(n_paths=10000, n_steps=252)  # shape (10000, 252)

    qrng = QuasiRandom(dimension=252)
    z = qrng.normals(n_paths=10000)               # shape (10000, 252)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc, norm


class PseudoRandom:
    """Pseudo-random standard normal generator with seed management.

    Args:
        seed: random seed for reproducibility. None for non-deterministic.
    """

    def __init__(self, seed: int | None = 42):
        self._rng = np.random.default_rng(seed)

    def normals(self, n_paths: int, n_steps: int = 1) -> np.ndarray:
        """Generate standard normal variates.

        Returns:
            Array of shape (n_paths, n_steps).
        """
        return self._rng.standard_normal((n_paths, n_steps))


class QuasiRandom:
    """Sobol quasi-random standard normal generator.

    Low-discrepancy sequences fill the space more uniformly than
    pseudo-random, giving faster MC convergence (roughly O(1/N) vs O(1/sqrt(N))).

    Args:
        dimension: number of dimensions (= n_steps in a path).
        seed: seed for the scrambled Sobol sequence.
    """

    def __init__(self, dimension: int = 1, seed: int | None = 42):
        if dimension < 1:
            raise ValueError(f"dimension must be >= 1, got {dimension}")
        self._dimension = dimension
        self._seed = seed

    def normals(self, n_paths: int) -> np.ndarray:
        """Generate quasi-random standard normal variates.

        Returns:
            Array of shape (n_paths, dimension).
        """
        sampler = qmc.Sobol(d=self._dimension, scramble=True, seed=self._seed)

        # Sobol requires n = 2^m; take next power of 2 then trim
        m = int(np.ceil(np.log2(max(n_paths, 1))))
        n_pow2 = 2**m
        u = sampler.random(n_pow2)  # shape (n_pow2, dimension), uniform [0,1)

        # Clip to avoid inf at boundaries, then inverse normal CDF
        u = np.clip(u, 1e-10, 1 - 1e-10)
        z = norm.ppf(u)

        return z[:n_paths]
