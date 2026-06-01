"""Turkish fixed income derivatives.

TLREF OIS, TURKGB sovereign bonds, and CPI-linked bonds.

    from pricebook.fixed_income.turkish import (
        TLREFSwap, TURKGBBond, TurkishCPILinker,
        build_try_curve, synthetic_tlref_strip,
    )

Conventions:
- Day count: ACT/365F for all instruments
- TLREF: Turkish Lira Reference Rate (overnight, CBRT)
- TURKGB: Turkish government bonds, semi-annual, ACT/365F, T+0 settlement
- CPI-linked: indexed to CPI_TR, 2-month lag, no deflation floor
- Extreme rate environment (~45% policy rate)

References:
    CBRT (2024). Central Bank of the Republic of Turkey — Market Conventions.
    Borsa Istanbul (2024). Fixed Income Specifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def _try_yf(start: date, end: date) -> float:
    """ACT/365F year fraction."""
    return year_fraction(start, end, DayCountConvention.ACT_365_FIXED)


def _next_bday(d: date, cal) -> date:
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


# ═══════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════

def synthetic_tlref_strip(reference_date: date, rate: float = 0.45,
                          n: int = 10, slope_bp: float = -50.0) -> list[dict]:
    """Synthetic TLREF swap strip. Turkey: ~45% policy rate (extreme)."""
    cal = get_calendar("TRY")
    tenors = [1, 3, 6, 9, 12, 18, 24, 36, 60, 120][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        mat = _next_bday(mat, cal)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


# ═══════════════════════════════════════════════════════════════
# Curve construction
# ═══════════════════════════════════════════════════════════════

def build_try_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap TRY discount curve from TLREF strip. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_365_FIXED, InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# TLREF Swap (overnight OIS)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TLREFSwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class TLREFSwap:
    """Turkish TLREF overnight swap. Annual fixed, ACT/365F."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1e9, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> TLREFSwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        dc = DayCountConvention.ACT_365_FIXED
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
        return TLREFSwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "tlref_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# TURKGB Bond (semi-annual ACT/365F, T+0 settlement)
# ═══════════════════════════════════════════════════════════════

class TURKGBBond:
    """Turkish government bond. Semi-annual coupon, ACT/365F, T+0 settlement.

    Unique: T+0 settlement (same-day), unlike most markets (T+1 or T+2).
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        dc = DayCountConvention.ACT_365_FIXED
        pv = sum(self.face * self.coupon * year_fraction(schedule[i-1], schedule[i], dc)
                 * curve.df(schedule[i]) for i in range(1, len(schedule)))
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "turkgb", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon}


# ═══════════════════════════════════════════════════════════════
# Turkish CPI Linker (CPI_TR indexed, 2-month lag, no floor)
# ═══════════════════════════════════════════════════════════════

@dataclass
class TurkishCPILinkerResult:
    real_price: float; nominal_price: float; cpi_ratio: float; real_yield: float
    def to_dict(self) -> dict: return vars(self)


class TurkishCPILinker:
    """Turkish CPI-linked government bond. CPI_TR indexed, 2-month lag, no floor.

    Semi-annual coupon, ACT/365F. No deflation floor on principal.

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: annual real coupon rate.
        base_cpi: CPI_TR at issue (lagged 2 months).
        face: face value.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100.0):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve,
              current_cpi: float) -> TurkishCPILinkerResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        cpi_ratio = current_cpi / self.base_cpi  # no deflation floor

        rpv = sum(self.face * self.real_coupon * _try_yf(schedule[i-1], schedule[i])
                  * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = _try_yf(ref, self.maturity)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return TurkishCPILinkerResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "turkish_cpi_linker", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}
