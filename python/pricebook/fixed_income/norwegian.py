"""Norwegian fixed income derivatives.

NOWA swaps and NGB sovereign bonds.

    from pricebook.fixed_income.norwegian import (
        NOWASwap, NGBBond, build_nok_curve, synthetic_nowa_strip,
    )

Conventions:
- Day count: ACT/360 for NOWA swaps, ACT/ACT ICMA for NGB
- NOWA: Norwegian Overnight Weighted Average (Norges Bank)
- NGB: Norwegian government bonds, annual, ACT/ACT ICMA, T+2

References:
    Norges Bank (2024). NOWA methodology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def synthetic_nowa_strip(reference_date: date, rate: float = 0.045,
                                 n: int = 10, slope_bp: float = 5.0) -> list[dict]:
    """Synthetic NOWA OIS strip. Norwegian: ~4.5% base rate."""
    cal = get_calendar("NOK")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_nok_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap NOK discount curve. ACT/360, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_360, InterpolationMethod.LOG_LINEAR)


@dataclass
class NOWASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class NOWASwap:
    """Norwegian NOWA overnight swap. Annual fixed, ACT/360."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 10_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> NOWASwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        dc = DayCountConvention.ACT_360
        schedule = generate_schedule(self.start, self.end, Frequency.ANNUAL)

        fixed_pv = sum(self.fixed_rate * year_fraction(schedule[i-1], schedule[i], dc)
                       * curve.df(schedule[i]) for i in range(1, len(schedule)))
        float_pv = curve.df(self.start) - curve.df(self.end)
        pv = self.direction * self.notional * (fixed_pv - float_pv)

        annuity = sum(year_fraction(schedule[i-1], schedule[i], dc)
                      * curve.df(schedule[i]) for i in range(1, len(schedule)))
        par = float_pv / annuity if annuity > 0 else 0
        pv_up = self.direction * self.notional * (
            (self.fixed_rate + 0.0001) * annuity - float_pv)
        return NOWASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "nowa_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


class NGBBond:
    """Norwegian government bond (NGB). Annual coupon, ACT/ACT ICMA, T+2."""

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        pv = sum(self.face * self.coupon * year_fraction(schedule[i-1], schedule[i],
                 DayCountConvention.ACT_ACT_ICMA, ref_start=schedule[i-1],
                 ref_end=schedule[i], frequency=1) * curve.df(schedule[i])
                 for i in range(1, len(schedule)))
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "ngb", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon}
