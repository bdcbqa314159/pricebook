"""Multi-asset correlated GBM Monte Carlo.

Cholesky-based correlated path generation for basket options,
worst-of autocallables, and multi-asset exotics.

    from pricebook.multi_asset_mc import CorrelatedGBM

    gen = CorrelatedGBM(
        spots=[100, 50, 200], vols=[0.20, 0.25, 0.30],
        corr_matrix=[[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]],
        rates=[0.03, 0.03, 0.03])
    paths = gen.generate(T=1.0, n_steps=252, n_paths=100_000)
    # paths.shape = (3, 100_000, 253)

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from pricebook.serialisable import _register


@dataclass
class MultiAssetResult:
    """Result from multi-asset MC pricing."""
    price: float
    std_error: float = 0.0
    n_paths: int = 0
    n_assets: int = 0

    def to_dict(self) -> dict:
        return {"price": self.price, "std_error": self.std_error,
                "n_paths": self.n_paths, "n_assets": self.n_assets}


class CorrelatedGBM:
    """Correlated multi-asset GBM path generator.

    Uses Cholesky decomposition of the correlation matrix to generate
    correlated Brownian increments.

    dS_i/S_i = (r_i - q_i) dt + σ_i dW_i
    where dW_i · dW_j = ρ_ij dt

    Args:
        spots: initial prices per asset.
        vols: volatilities per asset.
        corr_matrix: NxN correlation matrix (must be positive definite).
        rates: risk-free rates per asset (or single rate for all).
        div_yields: dividend yields per asset (default 0).
    """

    def __init__(
        self,
        spots: list[float],
        vols: list[float],
        corr_matrix: list[list[float]] | np.ndarray,
        rates: list[float] | float = 0.03,
        div_yields: list[float] | float = 0.0,
    ):
        self.n_assets = len(spots)
        self.spots = np.array(spots, dtype=float)
        self.vols = np.array(vols, dtype=float)
        self.corr = np.array(corr_matrix, dtype=float)

        if isinstance(rates, (int, float)):
            self.rates = np.full(self.n_assets, float(rates))
        else:
            self.rates = np.array(rates, dtype=float)

        if isinstance(div_yields, (int, float)):
            self.div_yields = np.full(self.n_assets, float(div_yields))
        else:
            self.div_yields = np.array(div_yields, dtype=float)

        # Validate
        if self.corr.shape != (self.n_assets, self.n_assets):
            raise ValueError(f"corr_matrix shape {self.corr.shape} != ({self.n_assets}, {self.n_assets})")
        if len(self.vols) != self.n_assets:
            raise ValueError(f"vols length {len(self.vols)} != n_assets {self.n_assets}")

        # Cholesky decomposition
        try:
            self.cholesky = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError:
            raise ValueError("Correlation matrix is not positive definite")

    def generate(
        self,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate correlated paths.

        Returns:
            Array of shape (n_assets, n_paths, n_steps+1).
        """
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        rng = np.random.default_rng(seed)

        paths = np.zeros((self.n_assets, n_paths, n_steps + 1))
        paths[:, :, 0] = self.spots[:, np.newaxis]

        for step in range(n_steps):
            # Independent normals: (n_assets, n_paths)
            Z = rng.standard_normal((self.n_assets, n_paths))
            # Correlate via Cholesky: W = L @ Z
            W = self.cholesky @ Z  # (n_assets, n_paths)

            for i in range(self.n_assets):
                mu = self.rates[i] - self.div_yields[i]
                S = paths[i, :, step]
                paths[i, :, step + 1] = S * np.exp(
                    (mu - 0.5 * self.vols[i]**2) * dt
                    + self.vols[i] * sqrt_dt * W[i]
                )

        return paths

    def terminal(
        self,
        T: float,
        n_paths: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate only terminal values. Returns (n_assets, n_paths)."""
        paths = self.generate(T, n_steps=1, n_paths=n_paths, seed=seed)
        return paths[:, :, -1]


def basket_option_mc(
    gen: CorrelatedGBM,
    strike: float,
    T: float,
    weights: list[float] | None = None,
    option_type: str = "call",
    n_paths: int = 100_000,
    seed: int = 42,
) -> MultiAssetResult:
    """Price a basket option: payoff = max(Σ w_i S_i(T) - K, 0).

    Args:
        gen: CorrelatedGBM generator.
        strike: basket strike.
        T: time to maturity.
        weights: asset weights (default = equal).
        option_type: "call" or "put".
    """
    terminals = gen.terminal(T, n_paths, seed)  # (n_assets, n_paths)

    if weights is None:
        w = np.ones(gen.n_assets) / gen.n_assets
    else:
        w = np.array(weights)

    basket = (w[:, np.newaxis] * terminals).sum(axis=0)  # (n_paths,)

    rate = float(gen.rates.mean())
    df = math.exp(-rate * T)

    if option_type == "call":
        payoffs = np.maximum(basket - strike, 0.0)
    else:
        payoffs = np.maximum(strike - basket, 0.0)

    discounted = df * payoffs
    price = float(discounted.mean())
    std_err = float(discounted.std(ddof=1) / math.sqrt(n_paths))

    return MultiAssetResult(price=price, std_error=std_err,
                             n_paths=n_paths, n_assets=gen.n_assets)


def worst_of_mc(
    gen: CorrelatedGBM,
    T: float,
    barrier: float = 1.0,
    n_paths: int = 100_000,
    seed: int = 42,
) -> float:
    """Probability that worst performer breaches barrier.

    Returns P(min_i S_i(T)/S_i(0) < barrier).
    """
    terminals = gen.terminal(T, n_paths, seed)
    returns = terminals / gen.spots[:, np.newaxis]
    worst = returns.min(axis=0)
    return float((worst < barrier).mean())
