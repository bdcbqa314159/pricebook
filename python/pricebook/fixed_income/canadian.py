"""Canadian fixed income derivatives.

CORRA swaps, Real Return Bonds (RRBs), CGB pricing.

    from pricebook.fixed_income.canadian import (
        CORRASwap, RRBBond, build_corra_curve, synthetic_corra_strip,
    )

Conventions:
- Day count: ACT/365 for CAD instruments
- CORRA: Canadian Overnight Repo Rate Average (BOC)
- CGB: Canadian Government Bond, semi-annual, ACT/365
- RRB: Real Return Bond (CPI_CA-linked, 3-month lag)

References:
    BOC (2024). Bank of Canada — CORRA.
    CSA (2024). Canadian Securities Administrators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def synthetic_corra_strip(reference_date: date, corra: float = 0.0425,
                           n: int = 10, slope_bp: float = 5.0) -> list[dict]:
    """Synthetic CORRA swap strip. Canada: ~4.25% overnight rate."""
    cal = get_calendar("CAD")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = corra + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_corra_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_365_FIXED, InterpolationMethod.LOG_LINEAR)


@dataclass
class CORRASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class CORRASwap:
    """Canadian CORRA overnight swap. ACT/365."""
    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 10_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> CORRASwapResult:
        tau = year_fraction(self.start, self.end, DayCountConvention.ACT_365_FIXED)
        df_s, df_e = curve.df(self.start), curve.df(self.end)
        float_pv = df_s - df_e
        fixed_pv = self.fixed_rate * tau * df_e
        pv = self.direction * self.notional * (fixed_pv - float_pv)
        par = float_pv / (tau * df_e) if tau * df_e > 0 else 0
        pv_up = self.direction * self.notional * ((self.fixed_rate + 0.0001) * tau * df_e - float_pv)
        return CORRASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "corra_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


@dataclass
class RRBResult:
    real_price: float; nominal_price: float; cpi_ratio: float; real_yield: float
    def to_dict(self) -> dict: return vars(self)


class RRBBond:
    """Canadian Real Return Bond (RRB). CPI_CA-linked, 3-month lag.

    Like US TIPS: principal indexed to CPI. Semi-annual real coupon.
    Deflation floor: principal protected at par.
    """
    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100.0):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_cpi: float) -> RRBResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        cpi_ratio = max(current_cpi / self.base_cpi, 1.0)  # deflation floor

        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i],
                  DayCountConvention.ACT_365_FIXED) * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return RRBResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "rrb", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}
