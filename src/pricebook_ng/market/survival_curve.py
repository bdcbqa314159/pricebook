"""Hazard/survival curve, CDS leg math, and the hazard bootstrap (L1).

The credit analogue of the S03 discount-curve bootstrap. A `SurvivalCurve` gives
the survival probability `Q(t)` (no default by `t`) from a piecewise-constant
hazard — log-linear in `ln Q` on an ACT/365F axis, exactly like the discount
curve is log-linear in `ln DF`. `bootstrap_survival_curve` solves the curve so
each CDS reprices to zero at its par spread.

CDS legs (protection buyer, unit notional): the premium leg pays `spread` on an
annual ACT/360 schedule while alive; the protection leg pays `(1 - R)` on default.
Discretised on the premium grid:
    RPV01       = sum_i  tau_i * DF(t_i) * Q(t_i)
    protection  = (1-R) * sum_i DF(t_i) * (Q(t_{i-1}) - Q(t_i))
    par spread  = protection / RPV01
    value(buyer)= protection - spread * RPV01

Scope (CLAUDE.md 6b): annual premiums, no accrual-on-default, protection at the
interval end. Quarterly schedules, accrual-on-default, and a finer protection
integral are refinements for a later slice — the reprice-to-zero oracle is exact
on whatever discretisation the bootstrap and pricing share.

Provenance:
  quarry: python/pricebook/core/survival_curve.py
  source: standard single-name CDS; ISDA hazard-rate bootstrap
  oracle: each input CDS reprices to zero par spread (self-consistency) < 1e-10
  slice:  credit-hazard-bootstrap
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.solvers import bisect_root
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.snapshot import CurveHandle, MarketSnapshot

_CURVE_DC = DayCountConvention.ACT_365_FIXED   # interpolation time axis
_ACCRUAL_DC = DayCountConvention.ACT_360       # CDS premium accrual convention


@dataclass(frozen=True)
class CDSQuote:
    """A par CDS: the fair running `par_spread` for protection to `maturity`."""

    maturity: date
    par_spread: float


@dataclass(frozen=True)
class SurvivalCurve:
    """Log-linear (in ln Q) survival curve. `pillars` are (date, Q) including the
    anchor (valuation_date, 1.0), sorted by date."""

    valuation_date: date
    pillars: tuple[tuple[date, float], ...]

    def _t(self, d: date) -> float:
        return year_fraction(self.valuation_date, d, _CURVE_DC)

    def survival(self, d: date) -> float:
        ts = [self._t(pd) for pd, _ in self.pillars]
        lns = [math.log(q) for _, q in self.pillars]
        t = self._t(d)
        i = bisect_left(ts, t)
        if i == 0:
            lo, hi = 0, 1
        elif i >= len(ts):
            lo, hi = len(ts) - 2, len(ts) - 1     # flat-hazard extrapolation
        else:
            lo, hi = i - 1, i
        slope = (lns[hi] - lns[lo]) / (ts[hi] - ts[lo])
        return math.exp(lns[lo] + slope * (t - ts[lo]))

    def df(self, d: date) -> float:
        """Survival probability as a `CurveHandle` factor (A5): a hazard curve is
        the credit-risky discount-factor curve, so it lives in `curves`."""
        return self.survival(d)


def _rpv01(discount: CurveHandle, survival: SurvivalCurve, schedule: list[date]) -> float:
    return sum(
        year_fraction(schedule[i - 1], schedule[i], _ACCRUAL_DC)
        * discount.df(schedule[i]) * survival.survival(schedule[i])
        for i in range(1, len(schedule))
    )


def _protection_pv(
    discount: CurveHandle, survival: SurvivalCurve, schedule: list[date], recovery: float
) -> float:
    return (1.0 - recovery) * sum(
        discount.df(schedule[i])
        * (survival.survival(schedule[i - 1]) - survival.survival(schedule[i]))
        for i in range(1, len(schedule))
    )


def cds_par_spread(
    discount: CurveHandle, survival: SurvivalCurve, schedule: list[date], recovery: float
) -> float:
    """Fair running spread: protection PV / risky annuity (RPV01)."""
    return _protection_pv(discount, survival, schedule, recovery) / _rpv01(
        discount, survival, schedule
    )


def cds_pv(
    discount: CurveHandle,
    survival: SurvivalCurve,
    schedule: list[date],
    spread: float,
    recovery: float,
) -> float:
    """Value to the protection buyer, per unit notional: protection - spread*RPV01."""
    protection = _protection_pv(discount, survival, schedule, recovery)
    return protection - spread * _rpv01(discount, survival, schedule)


def bootstrap_survival_curve(
    market: MarketSnapshot, quotes: list[CDSQuote], recovery: float
) -> SurvivalCurve:
    """Bootstrap the survival curve from CDS par spreads, short end first: each
    quote adds a pillar whose Q makes that CDS reprice to zero."""
    if not quotes:
        raise ValueError("bootstrap requires at least one CDS quote")

    valuation = market.valuation_date
    discount = market.discount_curve
    pillars: list[tuple[date, float]] = [(valuation, 1.0)]

    for q in sorted(quotes, key=lambda x: x.maturity):
        schedule = generate_schedule(valuation, q.maturity, Frequency.ANNUAL)
        prev_q = pillars[-1][1]

        def repriced(q_mat: float, q=q, schedule=schedule) -> float:
            trial = SurvivalCurve(valuation, (*pillars, (q.maturity, q_mat)))
            return cds_pv(discount, trial, schedule, q.par_spread, recovery)

        # value is monotonic in Q(maturity); bracket (0, prev_Q]
        q_mat = bisect_root(repriced, 1e-12, prev_q)
        pillars.append((q.maturity, q_mat))

    return SurvivalCurve(valuation, tuple(pillars))
