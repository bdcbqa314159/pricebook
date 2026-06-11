"""Curve Jacobian, smooth forward interpolation, and roll-down analysis.

- Curve Jacobian: d(zero_rate_i) / d(market_quote_j)
- Roll-down: expected P&L from curve shape over time
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction, date_from_year_fraction


# ---------------------------------------------------------------------------
# Curve Jacobian
# ---------------------------------------------------------------------------


def curve_jacobian(
    curve: DiscountCurve,
    query_tenors: list[float],
    pillar_tenors: list[float] | None = None,
    bump_size: float = 0.0001,
    pillar_tol: float = 1e-2,
) -> np.ndarray:
    """Compute d(zero_rate at query) / d(zero_rate at pillar) via finite difference.

    Args:
        curve: the base discount curve.
        query_tenors: year fractions where we want zero rate sensitivities.
        pillar_tenors: year fractions of pillar points to bump.
            If None, uses the curve's own pillar times (excluding t=0).
            **Each entry must match (within `pillar_tol`) one of the curve's
            actual pillars** — bumping happens by curve-pillar-index, so the
            tenors are resolved to indices first. Mismatched tenors raise.
        bump_size: parallel bump in zero rate units.
        pillar_tol: matching tolerance (year fractions) when resolving
            `pillar_tenors` to curve indices. Default `1e-2` (~3.6 days)
            accommodates 365.25-vs-day-count rounding drift between
            `date_from_year_fraction` (which constructs the curve dates)
            and the curve's `day_count` (typically ACT/365F).

    Returns:
        Jacobian matrix of shape (n_query, n_pillar).
        J[i, j] = d(zero_rate(query_i)) / d(zero_rate(pillar_j)).

    Notes:
        Fix C.2 B1 (L1 audit): pre-fix the loop enumerated `pillar_tenors`
        and called `bumped_at(j, ...)` with the enumeration index `j`. When
        the user supplied a custom `pillar_tenors` shorter than the curve's
        pillar grid, the column labels were silently mismatched — Jacobian
        columns described the WRONG pillars. Now `pillar_tenors` is
        resolved to actual curve indices first.
    """
    ref = curve.reference_date
    curve_pillar_times = [float(t) for t in curve.pillar_times if t > 0]
    curve_pillar_idx = list(range(len(curve_pillar_times)))   # index into bumped_at

    if pillar_tenors is None:
        pillar_tenors = list(curve_pillar_times)
        resolved_idx = curve_pillar_idx
    else:
        # Resolve each requested tenor to an actual pillar index.
        resolved_idx = []
        for pt in pillar_tenors:
            # Find the curve pillar nearest to `pt` within `pillar_tol`.
            diffs = [(abs(pt - ct), i) for i, ct in enumerate(curve_pillar_times)]
            best = min(diffs)
            if best[0] > pillar_tol:
                raise ValueError(
                    f"pillar_tenors[{len(resolved_idx)}]={pt} does not match any "
                    f"curve pillar (within tol={pillar_tol}). Curve pillars: "
                    f"{curve_pillar_times}. Either pass pillar_tenors that "
                    f"correspond to actual curve pillars, or omit the argument "
                    f"to use the curve's own pillar grid."
                )
            resolved_idx.append(best[1])

    n_query = len(query_tenors)
    n_pillar = len(pillar_tenors)

    # Base zero rates at query points
    base_zeros = np.array([
        curve.zero_rate(date_from_year_fraction(ref, t))
        for t in query_tenors
    ])

    J = np.zeros((n_query, n_pillar))

    for j, pillar_idx in enumerate(resolved_idx):
        # Bump the resolved pillar's zero rate.
        bumped = curve.bumped_at(pillar_idx, bump_size)
        bumped_zeros = np.array([
            bumped.zero_rate(date_from_year_fraction(ref, t))
            for t in query_tenors
        ])
        J[:, j] = (bumped_zeros - base_zeros) / bump_size

    return J


def input_jacobian(
    build_func,
    market_quotes: list[float],
    query_tenors: list[float],
    bump_size: float = 0.0001,
) -> np.ndarray:
    """Compute d(zero_rate at query) / d(market_quote_j).

    Args:
        build_func: callable(quotes: list[float]) → DiscountCurve.
        market_quotes: base market input values.
        query_tenors: year fractions for zero rate output.
        bump_size: bump to each market quote.

    Returns:
        Jacobian of shape (n_query, n_quotes).
    """
    base_curve = build_func(market_quotes)
    ref = base_curve.reference_date

    base_zeros = np.array([
        base_curve.zero_rate(date_from_year_fraction(ref, t))
        for t in query_tenors
    ])

    n_query = len(query_tenors)
    n_quotes = len(market_quotes)
    J = np.zeros((n_query, n_quotes))

    for j in range(n_quotes):
        bumped_quotes = list(market_quotes)
        bumped_quotes[j] += bump_size
        bumped_curve = build_func(bumped_quotes)

        bumped_zeros = np.array([
            bumped_curve.zero_rate(date_from_year_fraction(ref, t))
            for t in query_tenors
        ])
        J[:, j] = (bumped_zeros - base_zeros) / bump_size

    return J


# ---------------------------------------------------------------------------
# Roll-down analysis
# ---------------------------------------------------------------------------


def curve_rolldown(
    curve: DiscountCurve,
    horizon_days: int = 30,
) -> dict[str, list[float]]:
    """Compute roll-down: how zero rates and forwards change by just waiting.

    If nothing changes in the market, a trade "rolls down" the curve
    as time passes. This gives the expected carry from curve shape.

    Args:
        curve: current discount curve.
        horizon_days: number of days to roll forward.

    Returns:
        dict with tenors, current_zeros, rolled_zeros, rolldown (change in zero rate).
    """
    ref = curve.reference_date
    tenors = [float(t) for t in curve.pillar_times if t > 0]
    horizon = horizon_days / 365.0

    current_zeros = []
    rolled_zeros = []

    for t in tenors:
        d = date_from_year_fraction(ref, t)
        current_zeros.append(curve.zero_rate(d))

        # After horizon, the tenor t becomes t - horizon
        t_rolled = t - horizon
        if t_rolled > 0:
            d_rolled = date_from_year_fraction(ref, t_rolled)
            rolled_zeros.append(curve.zero_rate(d_rolled))
        else:
            rolled_zeros.append(current_zeros[-1])

    rolldown = [r - c for r, c in zip(rolled_zeros, current_zeros)]

    return {
        "tenors": tenors,
        "current_zeros": current_zeros,
        "rolled_zeros": rolled_zeros,
        "rolldown": rolldown,
    }


def rolldown_pnl(
    curve: DiscountCurve,
    notional: float,
    maturity_years: float,
    horizon_days: int = 30,
) -> float:
    """Estimate roll-down P&L for a zero-coupon position.

    P&L ≈ notional * (df_rolled - df_current) where rolled uses
    the shorter maturity on the same curve.
    """
    ref = curve.reference_date
    d_mat = date_from_year_fraction(ref, maturity_years)
    df_current = curve.df(d_mat)

    horizon = horizon_days / 365.0
    t_rolled = maturity_years - horizon
    if t_rolled <= 0:
        return notional * (1.0 - df_current)

    d_rolled = date_from_year_fraction(ref, t_rolled)
    df_rolled = curve.df(d_rolled)

    return notional * (df_rolled - df_current)
