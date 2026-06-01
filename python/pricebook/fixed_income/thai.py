"""Thai fixed income derivatives.

THOR swaps, Thai government bonds.

    from pricebook.fixed_income.thai import (
        THORSwap, THAIGBBond,
        build_thb_curve, synthetic_thor_strip,
    )

Conventions:
- Day count: ACT/365F for THOR swaps
- THOR: Thai Overnight Repurchase Rate (BOT)
- THAIGB: semi-annual ACT/ACT ICMA, T+2

References:
    BOT (2024). Bank of Thailand — THOR.
    PDMO (2024). Public Debt Management Office — Thai government bond conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


_DC = DayCountConvention.ACT_365_FIXED
_DC_ICMA = DayCountConvention.ACT_ACT_ICMA


def synthetic_thor_strip(reference_date: date, rate: float = 0.025,
                          n: int = 10, slope_bp: float = 3.0) -> list[dict]:
    """Synthetic THOR OIS strip. Thailand: ~2.5% base rate."""
    cal = get_calendar("THB")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_thb_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap THB discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class THORSwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class THORSwap:
    """Thai THOR overnight swap. Annual fixed, ACT/365F."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> THORSwapResult:
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
        return THORSwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "thor_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# THAIGB (Thai Government Bond) — semi-annual ACT/ACT ICMA, T+2
# ═══════════════════════════════════════════════════════════════

class THAIGBBond:
    """Thai Government Bond. Semi-annual ACT/ACT ICMA, T+2."""

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
        return {"type": "thaigb", "maturity": self.maturity.isoformat(), "coupon": self.coupon}
