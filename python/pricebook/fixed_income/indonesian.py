"""Indonesian fixed income derivatives.

INDONIA swaps, INDOGB bonds.

    from pricebook.fixed_income.indonesian import (
        INDONIASwap, INDOGBBond,
        build_idr_curve, synthetic_indonia_strip,
    )

Conventions:
- Day count: ACT/360 for INDONIA swaps
- INDONIA: Indonesia Overnight Index Average (BI)
- INDOGB: semi-annual ACT/ACT ICMA, T+2

References:
    BI (2024). Bank Indonesia -- INDONIA.
    MOF (2024). Ministry of Finance -- INDOGB conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


_DC = DayCountConvention.ACT_360
_DC_ICMA = DayCountConvention.ACT_ACT_ICMA


def synthetic_indonia_strip(reference_date: date, rate: float = 0.06,
                            n: int = 10, slope_bp: float = -3.0) -> list[dict]:
    """Synthetic INDONIA OIS strip. Indonesia: ~6% base rate."""
    cal = get_calendar("IDR")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_idr_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap IDR discount curve. ACT/360, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class INDONIASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class INDONIASwap:
    """Indonesian INDONIA overnight swap. Annual fixed, ACT/360."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> INDONIASwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.start, self.end, Frequency.ANNUAL)
        fixed_pv = sum(self.fixed_rate * year_fraction(schedule[i-1], schedule[i], _DC)
                       * curve.df(schedule[i]) for i in range(1, len(schedule)))
        float_pv = curve.df(self.start) - curve.df(self.end)
        pv = self.direction * self.notional * (fixed_pv - float_pv)
        annuity = sum(year_fraction(schedule[i-1], schedule[i], _DC)
                      * curve.df(schedule[i]) for i in range(1, len(schedule)))
        par = float_pv / annuity if annuity > 0 else 0
        pv_up = self.direction * self.notional * (
            (self.fixed_rate + 0.0001) * annuity - float_pv)
        return INDONIASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "indonia_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ===================================================================
# INDOGB (Indonesian Government Bond) -- semi-annual ACT/ACT ICMA, T+2
# ===================================================================

class INDOGBBond:
    """Indonesian Government Bond. Semi-annual ACT/ACT ICMA, T+2."""

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= curve.reference_date:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], _DC_ICMA,
                                ref_start=schedule[i-1], ref_end=schedule[i], frequency=2)
            pv += self.face * self.coupon * tau * curve.df(schedule[i])
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "indogb", "maturity": self.maturity.isoformat(), "coupon": self.coupon}
