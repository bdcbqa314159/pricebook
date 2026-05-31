"""Dividend strip analytics: per-period decomposition, carry, growth.

    from pricebook.equity.dividend_strip import (
        decompose_strip, strip_carry, dividend_growth_rate,
    )

    strips = decompose_strip(curve, n_periods=4)

References:
    van Bezooijen (2016). Dividend Strip Trading, JPM.
    Binsbergen, Brandt & Koijen (2012). On the Timing and Pricing of Dividends, AER.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.equity.dividend_advanced import DividendCurve


@dataclass
class DividendStrip:
    """Single strip: one period's forward dividend."""
    period_start: float     # years
    period_end: float       # years
    forward_div: float      # forward value of dividends in this period
    pv: float               # present value
    implied_yield: float    # annualised yield for this period
    weight: float           # fraction of annual total

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class StripCarryResult:
    """Carry analytics for a dividend strip."""
    strip: DividendStrip
    carry: float            # carry = yield - funding cost
    roll_down: float        # change in PV from 1-day time decay
    total: float            # carry + roll

    def to_dict(self) -> dict:
        return {"period_end": self.strip.period_end,
                "carry": self.carry, "roll_down": self.roll_down,
                "total": self.total}


def decompose_strip(
    curve: DividendCurve,
    n_periods: int | None = None,
    custom_breaks: list[float] | None = None,
) -> list[DividendStrip]:
    """Decompose a DividendCurve into per-period strips.

    Args:
        curve: calibrated dividend curve.
        n_periods: number of equal periods (default: len(curve.tenors)).
        custom_breaks: custom period boundaries in years.

    Returns:
        List of DividendStrip, one per period. Forward divs sum to total.
    """
    if custom_breaks is not None:
        breaks = sorted(custom_breaks)
    elif n_periods is not None:
        t_max = float(curve.tenors[-1])
        breaks = [t_max * i / n_periods for i in range(1, n_periods + 1)]
    else:
        breaks = list(curve.tenors)

    # Interpolate cumulative dividends at break points
    cum_at = np.interp(breaks, curve.tenors, curve.cumulative_dividends, left=0.0)
    total = float(cum_at[-1]) if len(cum_at) > 0 else 1.0

    strips = []
    prev_cum = 0.0
    prev_t = 0.0

    for i, t in enumerate(breaks):
        fwd_div = float(cum_at[i] - prev_cum)
        dt = t - prev_t
        pv = fwd_div  # already in PV terms from curve
        impl_yield = fwd_div / dt if dt > 0 else 0.0
        weight = fwd_div / total if total > 0 else 0.0

        strips.append(DividendStrip(prev_t, t, fwd_div, pv, impl_yield, weight))
        prev_cum = float(cum_at[i])
        prev_t = t

    return strips


def strip_carry(
    strips: list[DividendStrip],
    funding_rate: float,
    spot: float = 100.0,
) -> list[StripCarryResult]:
    """Compute carry-and-roll for each dividend strip.

    Carry = implied dividend yield - funding cost.
    Roll-down = change in PV from time passing (1 day).

    Args:
        strips: list of DividendStrip.
        funding_rate: annualised funding cost (repo rate).
        spot: spot for normalisation.
    """
    results = []
    for s in strips:
        dt = s.period_end - s.period_start
        if dt <= 0:
            results.append(StripCarryResult(s, 0.0, 0.0, 0.0))
            continue

        # Carry: earn implied yield, pay funding
        carry = (s.implied_yield - funding_rate * spot) * (1.0 / 365)  # per day

        # Roll-down: as time passes, strip gets closer, PV changes
        # Approximate: dPV/dt ≈ PV × funding_rate (discount unwind)
        roll = s.pv * funding_rate * (1.0 / 365)

        results.append(StripCarryResult(s, carry, roll, carry + roll))

    return results


def dividend_growth_rate(
    strips: list[DividendStrip],
) -> float:
    """Implied dividend growth rate from forward strip term structure.

    Fits exponential growth: D(t) = D_0 · exp(g·t) to the strip forward divs.

    Returns:
        Annualised growth rate g.
    """
    if len(strips) < 2:
        return 0.0

    # Use midpoint of each period
    mids = [(s.period_start + s.period_end) / 2 for s in strips]
    divs = [s.forward_div for s in strips]

    # Filter positive
    valid = [(m, d) for m, d in zip(mids, divs) if d > 0]
    if len(valid) < 2:
        return 0.0

    mids_v, divs_v = zip(*valid)

    # Log-linear regression: log(D) = log(D_0) + g·t
    log_divs = np.log(divs_v)
    t_arr = np.array(mids_v)

    # Simple OLS: g = cov(t, log_d) / var(t)
    t_mean = np.mean(t_arr)
    ld_mean = np.mean(log_divs)
    cov = np.mean((t_arr - t_mean) * (log_divs - ld_mean))
    var_t = np.mean((t_arr - t_mean) ** 2)

    if var_t < 1e-10:
        return 0.0

    return float(cov / var_t)
