"""Smith-Wilson curve extrapolation.

Regulatory method for extrapolating yield curves beyond the last
liquid point (LLP) to an ultimate forward rate (UFR).

Used by Solvency II / EIOPA for insurance risk-free curves.

The Smith-Wilson method fits a curve that:
1. Exactly matches market data at liquid maturities
2. Converges to the UFR at long maturities
3. Has smooth forward rates
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve


def _wilson_function(t: float, u: float, alpha: float, ufr: float) -> float:
    """Wilson function W(t, u) = exp(-ufr*(t+u)) * (alpha*min(t,u)
       - 0.5*exp(-alpha*max(t,u)) * (exp(alpha*min(t,u)) - exp(-alpha*min(t,u))))
    """
    min_tu = min(t, u)
    max_tu = max(t, u)
    return math.exp(-ufr * (t + u)) * (
        alpha * min_tu
        - 0.5 * math.exp(-alpha * max_tu) * (
            math.exp(alpha * min_tu) - math.exp(-alpha * min_tu)
        )
    )


def smith_wilson_calibrate(
    maturities: list[float],
    market_dfs: list[float],
    ufr: float,
    alpha: float,
) -> np.ndarray:
    """Calibrate Smith-Wilson coefficients to match market discount factors.

    Args:
        maturities: liquid point maturities (years).
        market_dfs: market discount factors at those maturities.
        ufr: ultimate forward rate (continuously compounded).
        alpha: convergence speed parameter.

    Returns:
        Coefficient vector zeta (one per maturity).
    """
    n = len(maturities)
    W = np.zeros((n, n))
    for i in range(n):
        W[i, i] = _wilson_function(maturities[i], maturities[i], alpha, ufr)
        for j in range(i + 1, n):
            W[i, j] = _wilson_function(maturities[i], maturities[j], alpha, ufr)
            W[j, i] = W[i, j]

    p_ufr = np.array([math.exp(-ufr * t) for t in maturities])
    target = np.array(market_dfs) - p_ufr

    zeta = np.linalg.solve(W, target)
    return zeta


def smith_wilson_df(
    t: float,
    maturities: list[float],
    zeta: np.ndarray,
    ufr: float,
    alpha: float,
) -> float:
    """Smith-Wilson discount factor at time t.

    P(t) = exp(-ufr*t) + sum_j zeta_j * W(t, u_j)
    """
    p = math.exp(-ufr * t)
    for j in range(len(maturities)):
        p += zeta[j] * _wilson_function(t, maturities[j], alpha, ufr)
    return p


def smith_wilson_forward(
    t: float,
    maturities: list[float],
    zeta: np.ndarray,
    ufr: float,
    alpha: float,
    dt: float = 1.0 / 365,
) -> float:
    """Instantaneous forward rate from Smith-Wilson curve."""
    p1 = smith_wilson_df(t, maturities, zeta, ufr, alpha)
    p2 = smith_wilson_df(t + dt, maturities, zeta, ufr, alpha)
    if p1 <= 0 or p2 <= 0:
        return ufr
    return -math.log(p2 / p1) / dt


def smith_wilson_curve(
    reference_date: date,
    maturities: list[float],
    market_dfs: list[float],
    ufr: float = 0.0345,
    alpha: float = 0.1,
    extrapolation_tenors: list[float] | None = None,
) -> DiscountCurve:
    """Build a DiscountCurve using Smith-Wilson extrapolation.

    Args:
        reference_date: valuation date.
        maturities: liquid point maturities (years).
        market_dfs: discount factors at liquid points.
        ufr: ultimate forward rate (EIOPA default: 3.45%).
        alpha: convergence speed (EIOPA default: ~0.1).
        extrapolation_tenors: additional tenors beyond LLP for the curve.

    Returns:
        DiscountCurve that matches market at liquid points and
        converges to UFR at long maturities.
    """
    zeta = smith_wilson_calibrate(maturities, market_dfs, ufr, alpha)

    if extrapolation_tenors is None:
        extrapolation_tenors = [30, 40, 50, 60, 80, 100, 120]

    all_tenors = sorted(set(maturities + extrapolation_tenors))
    all_dfs = [smith_wilson_df(t, maturities, zeta, ufr, alpha) for t in all_tenors]

    dates = [date.fromordinal(reference_date.toordinal() + int(t * 365)) for t in all_tenors]
    return DiscountCurve(reference_date, dates, all_dfs)
