"""SABR convexity adjustment for RFR futures.

Market-standard SABR-based convexity for EUR/GBP RFR futures,
complementing the existing Hull-White analytical adjustment.

* :func:`sabr_convexity_adjustment` — SABR-based convexity for futures.
* :func:`empirical_convexity` — calibrate from futures vs OIS spread.
* :func:`compare_convexity_models` — HW vs SABR comparison.

References:
    Hagan et al., *Managing Smile Risk*, Wilmott, 2002.
    Piterbarg, *Rates Squared*, Risk, 2003.
    Henrard, *Interest Rate Modelling in the Multi-Curve Framework*, Ch. 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ConvexityResult:
    """Futures convexity adjustment result."""
    adjustment_bp: float        # convexity adjustment in basis points
    futures_rate: float         # observed futures rate
    forward_rate: float         # adjusted forward rate
    model: str                  # "sabr", "hw", or "empirical"

    def to_dict(self) -> dict:
        return vars(self)


def sabr_convexity_adjustment(
    futures_rate: float,
    expiry_years: float,
    alpha: float = 0.01,
    beta: float = 0.5,
    rho: float = -0.3,
    nu: float = 0.4,
) -> ConvexityResult:
    """SABR-based convexity adjustment for RFR futures.

    The convexity adjustment arises because futures settle daily
    (linear payoff) while forwards settle at maturity (non-linear).

    For SABR with β < 1:
    adjustment ≈ -½ × α² × T × F^{2β-2} × (1 + higher-order SABR terms)

    Simplified Piterbarg approximation:
    Δ ≈ -½ × σ² × T² × F

    where σ is the SABR ATM vol.

    Args:
        futures_rate: futures-implied rate (decimal).
        expiry_years: time to futures expiry.
        alpha: SABR alpha (ATM vol level).
        beta: SABR beta (backbone).
        rho: SABR rho (skew).
        nu: SABR nu (vol-of-vol).
    """
    F = futures_rate
    T = expiry_years

    if T <= 0 or alpha <= 0:
        return ConvexityResult(0, futures_rate, futures_rate, "sabr")

    # SABR ATM vol
    F_beta = max(F, 1e-6) ** beta
    sigma_atm = alpha / F_beta if F_beta > 0 else alpha

    # Piterbarg convexity: -½ σ² T²
    # This captures the covariance between rate and discount factor
    adjustment = -0.5 * sigma_atm ** 2 * T * T

    # Higher-order SABR correction
    sabr_correction = 1.0 + (
        (2 - 3 * rho ** 2) * nu ** 2 / 24
        + rho * beta * nu * alpha / (4 * F_beta)
        + (beta * (beta - 2)) * alpha ** 2 / (24 * F_beta ** 2)
    ) * T

    adjustment *= sabr_correction

    forward_rate = futures_rate + adjustment
    adj_bp = adjustment * 10_000

    return ConvexityResult(
        adjustment_bp=adj_bp,
        futures_rate=futures_rate,
        forward_rate=forward_rate,
        model="sabr",
    )


def hw_convexity_adjustment(
    futures_rate: float,
    expiry_years: float,
    a: float = 0.03,
    sigma: float = 0.01,
) -> ConvexityResult:
    """Hull-White convexity adjustment (for comparison).

    Δ = -½ × σ² × B(t,T)² × T

    where B(t,T) = (1 - e^{-a(T-t)}) / a.

    Args:
        a: HW mean reversion.
        sigma: HW short rate vol.
    """
    T = expiry_years
    if T <= 0 or sigma <= 0:
        return ConvexityResult(0, futures_rate, futures_rate, "hw")

    B = (1 - math.exp(-a * T)) / a if a > 0 else T
    adjustment = -0.5 * sigma ** 2 * B * B

    return ConvexityResult(
        adjustment_bp=adjustment * 10_000,
        futures_rate=futures_rate,
        forward_rate=futures_rate + adjustment,
        model="hw",
    )


def empirical_convexity(
    futures_rates: list[float],
    ois_forwards: list[float],
    maturities: list[float],
) -> list[ConvexityResult]:
    """Calibrate empirical convexity from futures vs OIS spread.

    The observed spread between futures rates and OIS forwards
    IS the convexity adjustment (plus any residual basis).

    Args:
        futures_rates: observed futures-implied rates.
        ois_forwards: OIS forward rates for same period.
        maturities: time to expiry per contract.

    Returns:
        Empirical convexity adjustment per contract.
    """
    results = []
    for fr, ois, T in zip(futures_rates, ois_forwards, maturities):
        adj = fr - ois  # futures > forward due to convexity
        results.append(ConvexityResult(
            adjustment_bp=adj * 10_000,
            futures_rate=fr,
            forward_rate=ois,
            model="empirical",
        ))
    return results


def compare_convexity_models(
    futures_rate: float,
    expiry_years: float,
    sabr_params: dict | None = None,
    hw_params: dict | None = None,
) -> dict[str, ConvexityResult]:
    """Compare SABR and HW convexity adjustments.

    Args:
        sabr_params: {"alpha", "beta", "rho", "nu"}.
        hw_params: {"a", "sigma"}.
    """
    sabr_p = sabr_params or {"alpha": 0.01, "beta": 0.5, "rho": -0.3, "nu": 0.4}
    hw_p = hw_params or {"a": 0.03, "sigma": 0.01}

    return {
        "sabr": sabr_convexity_adjustment(futures_rate, expiry_years, **sabr_p),
        "hw": hw_convexity_adjustment(futures_rate, expiry_years, **hw_p),
    }
