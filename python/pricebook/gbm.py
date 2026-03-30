"""
Geometric Brownian Motion path generation.

    dS/S = (r - q) dt + sigma dW

Supports single-step (European payoffs) and multi-step (path-dependent)
simulation, with antithetic variates for variance reduction.

    gen = GBMGenerator(spot=100, rate=0.05, div_yield=0.02, vol=0.20)
    paths = gen.generate(T=1.0, n_steps=252, n_paths=10000, rng=PseudoRandom(42))
    # paths.shape = (10000, 253)  — includes time 0
"""

from __future__ import annotations

import numpy as np

from pricebook.rng import PseudoRandom, QuasiRandom


class GBMGenerator:
    """GBM path generator.

    Args:
        spot: initial price S(0).
        rate: risk-free rate (continuous compounding).
        div_yield: continuous dividend yield (default 0).
        vol: lognormal volatility.
    """

    def __init__(
        self,
        spot: float,
        rate: float,
        vol: float,
        div_yield: float = 0.0,
    ):
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if vol < 0:
            raise ValueError(f"vol must be non-negative, got {vol}")

        self.spot = spot
        self.rate = rate
        self.vol = vol
        self.div_yield = div_yield

    def generate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        rng: PseudoRandom | QuasiRandom | None = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        """Generate GBM paths.

        Args:
            T: time horizon in years.
            n_steps: number of time steps (1 = single-step for European).
            n_paths: number of simulated paths.
            rng: random number generator. Defaults to PseudoRandom(seed=42).
            antithetic: if True, use antithetic variates (doubles effective paths).

        Returns:
            Array of shape (n_effective_paths, n_steps + 1) where column 0 is S(0).
            If antithetic, n_effective_paths = 2 * n_paths.
        """
        if rng is None:
            rng = PseudoRandom(seed=42)

        dt = T / n_steps
        drift = (self.rate - self.div_yield - 0.5 * self.vol**2) * dt
        diffusion = self.vol * np.sqrt(dt)

        # Generate normals
        if isinstance(rng, QuasiRandom):
            z = rng.normals(n_paths)
            # Sobol dimension must match n_steps; reshape if needed
            if z.shape[1] != n_steps:
                # Re-create with correct dimension
                qrng = QuasiRandom(dimension=n_steps, seed=rng._seed)
                z = qrng.normals(n_paths)
        else:
            z = rng.normals(n_paths, n_steps)

        if antithetic:
            z = np.vstack([z, -z])  # shape (2*n_paths, n_steps)

        n_eff = z.shape[0]

        # Build log-returns and cumulate
        log_increments = drift + diffusion * z  # shape (n_eff, n_steps)
        log_paths = np.cumsum(log_increments, axis=1)  # cumulative log-return

        # Prepend log(S0) = 0 (since we'll multiply by S0 at the end)
        paths = np.empty((n_eff, n_steps + 1))
        paths[:, 0] = self.spot
        paths[:, 1:] = self.spot * np.exp(log_paths)

        return paths

    def terminal(
        self,
        T: float,
        n_paths: int,
        rng: PseudoRandom | QuasiRandom | None = None,
        antithetic: bool = False,
    ) -> np.ndarray:
        """Generate terminal values S(T) only (single-step, more efficient).

        Returns:
            1D array of shape (n_effective_paths,).
        """
        if rng is None:
            rng = PseudoRandom(seed=42)

        if isinstance(rng, QuasiRandom):
            qrng = QuasiRandom(dimension=1, seed=rng._seed)
            z = qrng.normals(n_paths).ravel()
        else:
            z = rng.normals(n_paths, 1).ravel()

        if antithetic:
            z = np.concatenate([z, -z])

        drift = (self.rate - self.div_yield - 0.5 * self.vol**2) * T
        diffusion = self.vol * np.sqrt(T)

        return self.spot * np.exp(drift + diffusion * z)
