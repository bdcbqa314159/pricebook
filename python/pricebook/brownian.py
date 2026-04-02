"""
Brownian motion framework.

Standard Wiener process, multi-dimensional correlated BM,
and Brownian bridge. Foundation for all continuous-time stochastic models.

    from pricebook.brownian import WienerProcess, CorrelatedBM, BrownianBridge

    wp = WienerProcess(seed=42)
    paths = wp.sample(T=1.0, n_steps=252, n_paths=10000)

    cbm = CorrelatedBM(corr_matrix=[[1, 0.5], [0.5, 1]], seed=42)
    paths = cbm.sample(T=1.0, n_steps=252, n_paths=10000)
"""

from __future__ import annotations

import math

import numpy as np


class WienerProcess:
    """Standard 1D Brownian motion W(t).

    Properties: W(0)=0, E[W(t)]=0, Var[W(t)]=t, independent increments.
    """

    def __init__(self, seed: int | None = 42):
        self._rng = np.random.default_rng(seed)

    def sample(
        self, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Simulate Wiener paths.

        Returns:
            Array of shape (n_paths, n_steps + 1). Column 0 is W(0)=0.
        """
        dt = T / n_steps
        dW = math.sqrt(dt) * self._rng.standard_normal((n_paths, n_steps))
        W = np.zeros((n_paths, n_steps + 1))
        W[:, 1:] = np.cumsum(dW, axis=1)
        return W

    def increments(
        self, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Return just the increments dW. Shape: (n_paths, n_steps)."""
        dt = T / n_steps
        return math.sqrt(dt) * self._rng.standard_normal((n_paths, n_steps))


class CorrelatedBM:
    """Multi-dimensional correlated Brownian motion.

    Given a d×d correlation matrix, generates d correlated Wiener processes
    via Cholesky decomposition: W = L @ Z where L = cholesky(corr).

    Args:
        corr_matrix: d×d correlation matrix (symmetric, positive definite).
        seed: random seed.
    """

    def __init__(self, corr_matrix: list[list[float]] | np.ndarray, seed: int | None = 42):
        self._corr = np.asarray(corr_matrix, dtype=float)
        self._d = self._corr.shape[0]
        self._L = np.linalg.cholesky(self._corr)
        self._rng = np.random.default_rng(seed)

    @property
    def dimension(self) -> int:
        return self._d

    def sample(
        self, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Simulate correlated BM paths.

        Returns:
            Array of shape (n_paths, n_steps + 1, d). [:,0,:] = 0.
        """
        dt = T / n_steps
        Z = self._rng.standard_normal((n_paths, n_steps, self._d))
        # Correlate: dW = sqrt(dt) * Z @ L^T
        dW = math.sqrt(dt) * (Z @ self._L.T)
        W = np.zeros((n_paths, n_steps + 1, self._d))
        W[:, 1:, :] = np.cumsum(dW, axis=1)
        return W

    def increments(
        self, T: float, n_steps: int, n_paths: int,
    ) -> np.ndarray:
        """Correlated increments. Shape: (n_paths, n_steps, d)."""
        dt = T / n_steps
        Z = self._rng.standard_normal((n_paths, n_steps, self._d))
        return math.sqrt(dt) * (Z @ self._L.T)


class BrownianBridge:
    """Brownian bridge: W(t) conditioned on W(0)=a, W(T)=b.

    Used for exact barrier crossing simulation and variance reduction.

    The bridge at time s ∈ [0, T]:
        W(s) = a + (b-a)*s/T + sqrt(s*(T-s)/T) * Z
    """

    def __init__(self, seed: int | None = 42):
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        start: float = 0.0,
        end: float = 0.0,
    ) -> np.ndarray:
        """Simulate bridge paths from start to end.

        Returns:
            Array of shape (n_paths, n_steps + 1).
            Column 0 = start, column -1 = end.
        """
        times = np.linspace(0, T, n_steps + 1)
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = start
        paths[:, -1] = end

        # Fill intermediate points using bridge construction
        # Sequential: condition each point on its neighbours
        for i in range(1, n_steps):
            s = times[i]
            # Bridge mean: linear interpolation between start and end
            mean = start + (end - start) * s / T
            # Bridge variance: s*(T-s)/T
            var = s * (T - s) / T
            paths[:, i] = mean + math.sqrt(max(var, 0.0)) * self._rng.standard_normal(n_paths)

        return paths

    @staticmethod
    def conditional_mean(
        t: float, T: float, start: float, end: float,
    ) -> float:
        """E[W(t) | W(0)=start, W(T)=end]."""
        return start + (end - start) * t / T

    @staticmethod
    def conditional_variance(t: float, T: float) -> float:
        """Var[W(t) | W(0), W(T)]."""
        if T <= 0:
            return 0.0
        return t * (T - t) / T
