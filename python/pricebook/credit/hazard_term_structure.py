"""Hazard rate term structure analytics: proxy curves, liquidity decomposition.

Tools for illiquid names that don't have their own liquid CDS market.
Build proxy survival curves from similar liquid names with adjustments.

    from pricebook.hazard_term_structure import proxy_survival_curve, liquidity_spread

    proxy = proxy_survival_curve(liquid_curve, additive_shift=0.005)
    liq = liquidity_spread(total_spread=0.015, credit_spread=0.010)

References:
    O'Kane, D. (2008). Modelling Single-name and Multi-name Credit Derivatives.
    Wiley, Ch. 6 — CDS curve construction and proxying.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.survival_curve import SurvivalCurve


def proxy_survival_curve(
    liquid_curve: SurvivalCurve,
    additive_shift: float = 0.0,
    multiplicative_scale: float = 1.0,
) -> SurvivalCurve:
    """Build a proxy survival curve from a liquid name's curve.

    Two adjustment methods:
    - Additive: h_proxy(t) = h_liquid(t) + additive_shift
    - Multiplicative: h_proxy(t) = h_liquid(t) × multiplicative_scale

    Both can be combined: h_proxy = h_liquid × scale + shift.

    Args:
        liquid_curve: calibrated survival curve from a liquid name.
        additive_shift: hazard rate shift (e.g. 0.005 = 50bp wider).
        multiplicative_scale: hazard rate multiplier (e.g. 1.5 = 50% riskier).

    Returns:
        New SurvivalCurve with adjusted hazard rates.
    """
    ref = liquid_curve.reference_date
    dates = list(liquid_curve._pillar_dates)
    times = [year_fraction(ref, d, liquid_curve.day_count) for d in dates]

    # Extract segment hazard rates and adjust
    new_survs = []
    q_prev = 1.0
    t_prev = 0.0
    for i, (t, d) in enumerate(zip(times, dates)):
        dt = t - t_prev
        if dt <= 0:
            new_survs.append(q_prev)
            continue
        # Original hazard for this segment
        h = liquid_curve.forward_hazard(
            ref + timedelta(days=int(t_prev * 365)) if t_prev > 0 else ref, d
        )
        # Adjust
        h_adj = max(h * multiplicative_scale + additive_shift, 1e-10)
        q = q_prev * math.exp(-h_adj * dt)
        new_survs.append(q)
        q_prev = q
        t_prev = t

    return SurvivalCurve(ref, dates, new_survs, liquid_curve.day_count)


def liquidity_spread(total_spread: float, credit_spread: float) -> float:
    """Decompose CDS spread into credit + liquidity components.

    total_spread = credit_spread + liquidity_premium

    Credit component: from CDS market or structural model (Merton).
    Liquidity: residual — captures bid-ask, market depth, issuance effects.

    Args:
        total_spread: observed CDS spread (or bond asset swap spread).
        credit_spread: estimated pure credit spread.

    Returns:
        Liquidity premium (total - credit).
    """
    return total_spread - credit_spread


def spread_from_survival(
    survival_curve: SurvivalCurve,
    maturity: date,
    recovery: float = 0.4,
) -> float:
    """Implied par CDS spread from a survival curve.

    Approximate: spread ≈ (1-R) × h̄  where h̄ is the average hazard rate.
    More precisely: spread = protection_leg_pv / risky_annuity.
    """
    ref = survival_curve.reference_date
    T = year_fraction(ref, maturity, survival_curve.day_count)
    if T <= 0:
        return 0.0
    q_T = survival_curve.survival(maturity)
    if q_T >= 1.0:
        return 0.0
    avg_hazard = -math.log(max(q_T, 1e-15)) / T
    return (1 - recovery) * avg_hazard


def compare_curves(
    curve_a: SurvivalCurve,
    curve_b: SurvivalCurve,
    label_a: str = "A",
    label_b: str = "B",
) -> list[dict]:
    """Compare two survival curves at their pillar dates.

    Returns list of {date, survival_a, survival_b, hazard_a, hazard_b, spread_diff_bp}.
    """
    # Use union of pillar dates
    all_dates = sorted(set(curve_a._pillar_dates) | set(curve_b._pillar_dates))
    result = []
    for d in all_dates:
        sa = curve_a.survival(d)
        sb = curve_b.survival(d)
        ha = curve_a.hazard_rate(d)
        hb = curve_b.hazard_rate(d)
        # Implied spread difference (approximate)
        spread_diff = (ha - hb) * 0.6 * 10_000  # (1-R) × Δh in bp
        result.append({
            "date": d.isoformat(),
            f"survival_{label_a}": sa,
            f"survival_{label_b}": sb,
            f"hazard_{label_a}": ha,
            f"hazard_{label_b}": hb,
            "spread_diff_bp": spread_diff,
        })
    return result
