"""Single-curve discount curve + rate/swap quote types (L1).

The `DiscountCurve` (log-linear in ln DF on an ACT/365F axis) reached through the
`CurveHandle` capability, plus the `DepositQuote`/`ParSwapQuote` market
observables it is built from. The bootstrap *solver* lives at the L3 calibration
front (`calibration.discount_curve`); this module holds the curve type it produces
and the greek `bumped`/`bump_pillar` operations. Off-range queries extrapolate at
the nearest segment's constant zero rate.

Scope (CLAUDE.md 6b): annual fixed legs with coupons landing on prior pillars,
unadjusted dates. Business-day-adjusted pillars, multi-curve (OIS discount /
IBOR projection), and non-pillar coupons arrive with the slice that needs them.

Provenance:
  quarry: python/pricebook/core/discount_curve.py
  source: Hull, Options Futures & Other Derivatives ch.4; single-curve discounting
  oracle: closed-form log-linear df + parallel/pillar `bumped` (greek FD) < 1e-12
  slice:  S03; bootstrapped-dv01 (parallel-zero `bumped`); key-rate-buckets (`bump_pillar`);
          general-curve-rates (zero_rate + instantaneous_forward — CP-2b, toward core/discount_curve)
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.schedule import Frequency
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

    def _bracket_slope(self, t: float) -> tuple[float, float, float]:
        """The log-linear segment covering time `t`: `(ln DF_lo, t_lo, slope)` where
        `slope = d(ln DF)/dt` on that segment. Shared by `df` and the forward rate."""
        ts = [self._t(pd) for pd, _ in self.pillars]
        lns = [math.log(df) for _, df in self.pillars]
        i = bisect_left(ts, t)
        if i == 0:
            lo, hi = 0, 1                        # at/before anchor, or before pillar 1
        elif i >= len(ts):
            lo, hi = len(ts) - 2, len(ts) - 1    # flat-zero extrapolation past last pillar
        else:
            lo, hi = i - 1, i
        slope = (lns[hi] - lns[lo]) / (ts[hi] - ts[lo])
        return lns[lo], ts[lo], slope

    def df(self, d: date) -> float:
        t = self._t(d)
        ln_lo, t_lo, slope = self._bracket_slope(t)
        return math.exp(ln_lo + slope * (t - t_lo))

    def instantaneous_forward(self, d: date) -> float:
        """`f(0,t) = -d/dt ln P(0,t)`. Log-linear-in-DF interpolation makes the forward
        piecewise CONSTANT per segment (`= -slope`), which integrates back to `-ln df`."""
        return -self._bracket_slope(self._t(d))[2]

    def zero_rate(self, d: date) -> float:
        """Continuously-compounded zero rate `-ln P(0,t)/t`; the short rate `f(0,0)` at t=0."""
        t = self._t(d)
        if t <= 0.0:
            return self.instantaneous_forward(d)
        return -math.log(self.df(d)) / t

    def bumped(self, shift: float) -> "DiscountCurve":
        """Parallel zero-rate shift, as a new curve (for generic curve greeks /
        dv01 on a bootstrapped curve): every pillar DF scales by exp(-shift*t), so
        each continuously-compounded zero moves by `shift`. Log-linear interpolation
        keeps the shift uniform between pillars (ln DF(t) -> ln DF(t) - shift*t)."""
        pillars = tuple((d, df * math.exp(-shift * self._t(d))) for d, df in self.pillars)
        return DiscountCurve(self.valuation_date, pillars)

    def bump_pillar(self, i: int, shift: float) -> "DiscountCurve":
        """Shift the zero at pillar `i` only (DF_i -> DF_i*exp(-shift*t_i)), the rest
        fixed — one key-rate bucket. Log-linear interpolation tents the bump between
        neighbours; bumping every pillar reproduces the parallel `bumped` (so the
        buckets partition dv01)."""
        d, df = self.pillars[i]
        bumped = (d, df * math.exp(-shift * self._t(d)))
        pillars = (*self.pillars[:i], bumped, *self.pillars[i + 1:])
        return DiscountCurve(self.valuation_date, pillars)
