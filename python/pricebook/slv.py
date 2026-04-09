"""Stochastic Local Volatility (SLV): mixing Heston with Dupire.

Combines the realistic dynamics of stochastic vol with the exact
calibration of local vol via a leverage function.

    dS/S = (r - q)dt + L(S, t) × √v × dW_s
    dv = κ(θ - v)dt + ξ√v dW_v
    dW_s dW_v = ρ dt

where L(S, t) is calibrated so that the model reprices vanillas.

    from pricebook.slv import SLVModel, slv_mc_european
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.black76 import black76_price, OptionType
from pricebook.local_vol import LocalVolSurface


# ---- SLV Model ----

@dataclass
class HestonParams:
    """Heston model parameters."""
    v0: float      # initial variance
    kappa: float   # mean reversion speed
    theta: float   # long-run variance
    xi: float      # vol of vol
    rho: float     # correlation


class SLVModel:
    """Stochastic Local Vol model.

    The leverage function L(S, t) adjusts the local vol so that:
        effective_vol = L(S, t) × √v

    With mixing fraction α:
        α = 1: pure local vol (L chosen so L×√θ ≈ σ_loc)
        α = 0: pure Heston (L = 1)

    Args:
        local_vol: Dupire local vol surface.
        heston: Heston parameters.
        mixing: mixing fraction (0 = pure Heston, 1 = pure local vol).
    """

    def __init__(
        self,
        local_vol: LocalVolSurface,
        heston: HestonParams,
        mixing: float = 0.5,
    ):
        self.local_vol = local_vol
        self.heston = heston
        self.mixing = max(0.0, min(1.0, mixing))

    def leverage(self, spot: float, t: float, v: float) -> float:
        """Leverage function L(S, t, v).

        L = σ_loc(S, t) / √(E[v | S]) ≈ σ_loc(S, t) / √v (simplified)
        Mixed: L = α × σ_loc / √v + (1 - α) × 1
        """
        sigma_loc = self.local_vol.vol(spot, t)
        sqrt_v = math.sqrt(max(v, 1e-10))

        if self.mixing >= 1.0:
            # Pure local vol
            return sigma_loc / sqrt_v if sqrt_v > 1e-8 else 1.0
        elif self.mixing <= 0.0:
            # Pure Heston
            return 1.0
        else:
            lv_component = sigma_loc / sqrt_v if sqrt_v > 1e-8 else 1.0
            return self.mixing * lv_component + (1 - self.mixing)


# ---- SLV MC simulation ----

def slv_mc(
    spot: float,
    rate: float,
    model: SLVModel,
    T: float,
    n_steps: int = 100,
    n_paths: int = 10_000,
    div_yield: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate SLV paths. Returns terminal spot values."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    h = model.heston
    rho = h.rho

    S = np.full(n_paths, spot, dtype=float)
    v = np.full(n_paths, h.v0, dtype=float)

    for step in range(n_steps):
        t = step * dt
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        Zv = Z1
        Zs = rho * Z1 + math.sqrt(1 - rho ** 2) * Z2

        # Leverage function (vectorised approximation)
        L = np.array([model.leverage(s, t, vi) for s, vi in zip(S, v)])

        sqrt_v = np.sqrt(np.maximum(v, 0))
        S = S * np.exp(
            (rate - div_yield - 0.5 * (L * sqrt_v) ** 2) * dt
            + L * sqrt_v * sqrt_dt * Zs
        )

        # Variance process (QE-like: reflect at zero)
        v = v + h.kappa * (h.theta - v) * dt + h.xi * sqrt_v * sqrt_dt * Zv
        v = np.maximum(v, 0)

    return S


def slv_mc_european(
    spot: float,
    rate: float,
    model: SLVModel,
    strike: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    n_steps: int = 100,
    n_paths: int = 50_000,
    div_yield: float = 0.0,
    seed: int = 42,
) -> float:
    """Price a European option under SLV via MC."""
    S_T = slv_mc(spot, rate, model, T, n_steps, n_paths, div_yield, seed)
    df = math.exp(-rate * T)

    if option_type == OptionType.CALL:
        payoffs = np.maximum(S_T - strike, 0)
    else:
        payoffs = np.maximum(strike - S_T, 0)

    return float(df * payoffs.mean())
