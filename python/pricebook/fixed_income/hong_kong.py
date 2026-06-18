"""Hong Kong fixed income derivatives.

HONIA swaps, HKGB bonds.

    from pricebook.fixed_income.hong_kong import (
        HONIASwap, HKGBBond,
        build_hkd_curve, synthetic_honia_strip,
    )

Conventions:
- Day count: ACT/365F for HONIA swaps
- HONIA: Hong Kong Dollar Overnight Index Average (HKMA)
- HKGB: semi-annual ACT/ACT ICMA, T+2
- HKD is pegged to USD (~4% tracks Fed funds)

References:
    HKMA (2024). Hong Kong Monetary Authority — HONIA.
    HKMA (2024). Hong Kong Government Bond conventions.
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


def synthetic_honia_strip(reference_date: date, rate: float = 0.04,
                           n: int = 10, slope_bp: float = 3.0) -> list[dict]:
    """Synthetic HONIA OIS strip. Hong Kong: ~4% (pegged to USD)."""
    cal = get_calendar("HKD")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_hkd_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap HKD discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class HONIASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class HONIASwap:
    """Hong Kong HONIA overnight swap. Annual fixed, ACT/365F."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> HONIASwapResult:
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
        return HONIASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "honia_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# HKGB (Hong Kong Government Bond) — semi-annual ACT/ACT ICMA, T+2
# ═══════════════════════════════════════════════════════════════

class HKGBBond:
    """Hong Kong Government Bond. Semi-annual ACT/ACT ICMA, T+2."""

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
        return {"type": "hkgb", "maturity": self.maturity.isoformat(), "coupon": self.coupon}
