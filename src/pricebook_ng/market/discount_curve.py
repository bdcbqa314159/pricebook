"""Bootstrapped single-curve discount curve (L1).

Built FROM market quotes (deposits + par swaps), reached through the same
`CurveHandle` capability as the Slice 0 flat curve. Deposits give the short-end
discount factors in closed form; each par swap extends the curve by one pillar
via a closed-form sequential solve (single-curve: discount = projection, so the
float leg telescopes to `1 - DF(maturity)`).

Interpolation is log-linear in the discount factor (linear in ln DF on an
ACT/365F time axis) — the standard, arbitrage-reasonable choice. Off-range
queries extrapolate at the nearest segment's constant zero rate.

Scope (CLAUDE.md 6b): annual fixed legs with coupons landing on prior pillars,
unadjusted dates. Business-day-adjusted pillars, multi-curve (OIS discount /
IBOR projection), and non-pillar coupons arrive with the slice that needs them.

Provenance:
  quarry: python/pricebook/core/discount_curve.py
  source: Hull, Options Futures & Other Derivatives ch.4; single-curve bootstrap
  oracle: inputs reprice to par (self-consistency) + closed-form deposit DFs < 1e-12
  slice:  S03; bootstrapped-dv01 (parallel-zero `bumped` for curve greeks)
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention, year_fraction

_CURVE_DC = DayCountConvention.ACT_365_FIXED  # internal interpolation time axis


@dataclass(frozen=True)
class DepositQuote:
    """A money-market deposit: simple-compounded rate to `maturity`."""

    maturity: date
    rate: float
    day_count: DayCountConvention


@dataclass(frozen=True)
class ParSwapQuote:
    """A par (fixed-vs-float) interest-rate swap quote."""

    maturity: date
    rate: float
    fixed_frequency: Frequency
    day_count: DayCountConvention


@dataclass(frozen=True)
class DiscountCurve:
    """Log-linear interpolated discount curve behind the `CurveHandle` protocol.

    `pillars` are (date, DF) pairs including the anchor (valuation_date, 1.0),
    sorted by date.
    """

    valuation_date: date
    pillars: tuple[tuple[date, float], ...]

    def _t(self, d: date) -> float:
        return year_fraction(self.valuation_date, d, _CURVE_DC)

    def df(self, d: date) -> float:
        ts = [self._t(pd) for pd, _ in self.pillars]
        lns = [math.log(df) for _, df in self.pillars]
        t = self._t(d)
        i = bisect_left(ts, t)
        if i == 0:
            lo, hi = 0, 1                        # at/before anchor, or before pillar 1
        elif i >= len(ts):
            lo, hi = len(ts) - 2, len(ts) - 1    # flat-zero extrapolation past last pillar
        else:
            lo, hi = i - 1, i
        slope = (lns[hi] - lns[lo]) / (ts[hi] - ts[lo])
        return math.exp(lns[lo] + slope * (t - ts[lo]))

    def bumped(self, shift: float) -> "DiscountCurve":
        """Parallel zero-rate shift, as a new curve (for generic curve greeks /
        dv01 on a bootstrapped curve): every pillar DF scales by exp(-shift*t), so
        each continuously-compounded zero moves by `shift`. Log-linear interpolation
        keeps the shift uniform between pillars (ln DF(t) -> ln DF(t) - shift*t)."""
        pillars = tuple((d, df * math.exp(-shift * self._t(d))) for d, df in self.pillars)
        return DiscountCurve(self.valuation_date, pillars)


def bootstrap_discount_curve(
    valuation_date: date,
    deposits: list[DepositQuote],
    swaps: list[ParSwapQuote],
) -> DiscountCurve:
    """Bootstrap a discount curve from deposits then par swaps, short end first."""
    if not deposits and not swaps:
        raise ValueError("bootstrap requires at least one deposit or swap quote")

    pillars: list[tuple[date, float]] = [(valuation_date, 1.0)]

    for dep in sorted(deposits, key=lambda q: q.maturity):
        tau = year_fraction(valuation_date, dep.maturity, dep.day_count)
        pillars.append((dep.maturity, 1.0 / (1.0 + dep.rate * tau)))

    for sw in sorted(swaps, key=lambda q: q.maturity):
        curve = DiscountCurve(valuation_date, tuple(pillars))
        sched = generate_schedule(valuation_date, sw.maturity, sw.fixed_frequency)
        # par: rate * sum(tau_i * DF(t_i)) = 1 - DF(t_n).  DF(t_i<n) are known
        # pillars; solve the linear equation for the final DF(t_n).
        annuity_known = 0.0
        tau_last = 0.0
        for i in range(1, len(sched)):
            tau_i = year_fraction(sched[i - 1], sched[i], sw.day_count)
            if i < len(sched) - 1:
                annuity_known += tau_i * curve.df(sched[i])
            else:
                tau_last = tau_i
        df_n = (1.0 - sw.rate * annuity_known) / (1.0 + sw.rate * tau_last)
        pillars.append((sw.maturity, df_n))

    return DiscountCurve(valuation_date, tuple(pillars))
