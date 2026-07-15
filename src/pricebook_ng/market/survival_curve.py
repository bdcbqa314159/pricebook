"""Hazard/survival curve + CDS leg math (L1).

A `SurvivalCurve` gives the survival probability `Q(t)` (no default by `t`) from a
piecewise-constant hazard — log-linear in `ln Q` on an ACT/365F axis, exactly like
the discount curve is log-linear in `ln DF`. The hazard *bootstrap* that solves the
curve to reprice each CDS to zero lives at the L3 calibration front
(`calibration.survival_curve`); this module holds the curve type it produces and the
CDS leg math (`cds_pv`/`cds_par_spread`) that the bootstrap reprices against.

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
  source: standard single-name CDS; ISDA hazard model
  oracle: closed-form log-linear Q + CDS RPV01/protection/par-spread leg math
  slice:  credit-hazard-bootstrap
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.snapshot import CurveHandle

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

    def bumped(self, shift: float) -> "SurvivalCurve":
        """Parallel hazard shift: scale each survival pillar by exp(-shift*t), as a
        new curve — the credit analogue of a rate shift (for generic curve greeks)."""
        pillars = tuple(
            (d, q * math.exp(-shift * year_fraction(self.valuation_date, d, _CURVE_DC)))
            for d, q in self.pillars
        )
        return SurvivalCurve(self.valuation_date, pillars)


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
