"""Dividend surface: term structure × dividend yield volatility.

    from pricebook.equity.dividend_surface import (
        DividendSurface, build_dividend_surface, simulate_dividend_surface,
    )

References:
    Bos, Kragt & Bovenberg (2017). Pricing and Hedging Dividend Derivatives.
    Buehler (2010). Stochastic Proportional Dividends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Note: DividendSimResult is defined after DividendSurface below


@dataclass
class DividendSurface:
    """Dividend yield surface across tenors."""
    tenors: np.ndarray            # years
    yield_levels: np.ndarray      # mean dividend yield per tenor
    yield_vols: np.ndarray        # volatility of dividend yield per tenor
    spot_correlation: float       # ρ(spot, div_yield)

    def interpolate(self, T: float) -> tuple[float, float]:
        """Interpolate yield level and vol at a given tenor."""
        level = float(np.interp(T, self.tenors, self.yield_levels))
        vol = float(np.interp(T, self.tenors, self.yield_vols))
        return level, vol

    def to_dict(self) -> dict:
        return {
            "tenors": self.tenors.tolist(),
            "yield_levels": self.yield_levels.tolist(),
            "yield_vols": self.yield_vols.tolist(),
            "spot_correlation": self.spot_correlation,
        }


def build_dividend_surface(
    spot: float,
    div_futures: list[dict],
    div_options: list[dict] | None = None,
    rate: float = 0.05,
) -> DividendSurface:
    """Build dividend surface from futures and options data.

    Args:
        div_futures: list of {"T": float, "price": float} (cumulative div futures).
        div_options: optional list of {"T": float, "iv": float} (dividend option IV).
        rate: risk-free rate.

    Returns:
        DividendSurface with yield levels and vols.
    """
    tenors = np.array([f["T"] for f in div_futures])
    prices = np.array([f["price"] for f in div_futures])

    # Implied yields
    yields = np.where(tenors > 0, prices / (spot * tenors), 0.0)

    # Yield vols: from options if available, else estimate from yield term structure
    if div_options and len(div_options) > 0:
        opt_tenors = np.array([o["T"] for o in div_options])
        opt_vols = np.array([o["iv"] for o in div_options])
        yield_vols = np.interp(tenors, opt_tenors, opt_vols)
    else:
        # Estimate: yield vol ~ 20-30% of yield level (typical)
        yield_vols = yields * 0.25

    # Default correlation: negative (higher spot → lower yield as ratio)
    rho = -0.3

    return DividendSurface(tenors, yields, yield_vols, rho)


@dataclass
class DividendSimResult:
    """Result of dividend surface simulation."""
    spot_paths: np.ndarray
    yield_paths: np.ndarray
    terminal_spot_mean: float
    terminal_yield_mean: float
    terminal_spot_std: float
    terminal_yield_std: float

    def to_dict(self) -> dict:
        return {
            "terminal_spot_mean": self.terminal_spot_mean,
            "terminal_yield_mean": self.terminal_yield_mean,
            "terminal_spot_std": self.terminal_spot_std,
            "terminal_yield_std": self.terminal_yield_std,
            "n_paths": self.spot_paths.shape[0],
            "n_steps": self.spot_paths.shape[1] - 1,
        }


def simulate_dividend_surface(
    surface: DividendSurface,
    spot: float,
    rate: float,
    T: float,
    spot_vol: float = 0.20,
    kappa_q: float = 2.0,
    n_paths: int = 10_000,
    n_steps: int = 100,
    seed: int = 42,
) -> DividendSimResult:
    """Simulate correlated spot and dividend yield paths.

    Spot: GBM dS/S = (r - q(t))dt + σ_S dW_S  (log-Euler scheme)
    Div yield: OU dq = κ(θ - q)dt + ξ dW_q
    dW_S·dW_q = ρ dt

    Args:
        spot_vol: annualised spot volatility.
        kappa_q: mean-reversion speed of dividend yield OU process.

    Returns:
        DividendSimResult with spot and yield paths.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Surface parameters at T
    q0, xi = surface.interpolate(T)
    theta_q = q0
    rho = surface.spot_correlation

    # Validate correlation
    rho = max(-0.999, min(0.999, rho))

    # Cholesky for correlation
    L = np.array([[1.0, 0.0], [rho, math.sqrt(1 - rho**2)]])

    # Log-Euler for spot (prevents negative spot)
    log_S = np.full(n_paths, math.log(spot))
    q = np.full(n_paths, q0)

    spot_paths = np.zeros((n_paths, n_steps + 1))
    yield_paths = np.zeros((n_paths, n_steps + 1))
    spot_paths[:, 0] = spot
    yield_paths[:, 0] = q0

    sqrt_dt = math.sqrt(dt)

    for t in range(n_steps):
        Z = rng.standard_normal((n_paths, 2))
        W = Z @ L.T  # correlated increments

        # Spot (log-Euler): d(log S) = (r - q - 0.5σ²)dt + σ dW
        log_S += (rate - q - 0.5 * spot_vol**2) * dt + spot_vol * sqrt_dt * W[:, 0]

        # Dividend yield (OU, exact discretisation)
        dq = kappa_q * (theta_q - q) * dt + xi * sqrt_dt * W[:, 1]
        q = np.maximum(q + dq, 0.0)

        spot_paths[:, t + 1] = np.exp(log_S)
        yield_paths[:, t + 1] = q

    S_terminal = np.exp(log_S)
    return DividendSimResult(
        spot_paths=spot_paths,
        yield_paths=yield_paths,
        terminal_spot_mean=float(np.mean(S_terminal)),
        terminal_yield_mean=float(np.mean(q)),
        terminal_spot_std=float(np.std(S_terminal)),
        terminal_yield_std=float(np.std(q)),
    )
