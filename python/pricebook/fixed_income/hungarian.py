"""Hungarian fixed income derivatives.

BUBOR swaps, HUFONIA OIS, HGB sovereign bonds, and CPI-linked bonds.

    from pricebook.fixed_income.hungarian import (
        BUBORSwap, HUFONIASwap, HGBBond, HGBLinker,
        build_huf_curve, synthetic_bubor_strip,
    )

Conventions:
- Day count: ACT/360 for BUBOR/HUFONIA swaps
- BUBOR: 3M Budapest Interbank Offered Rate (MNB)
- HUFONIA: Hungarian Forint Overnight Index Average (MNB)
- HGB: Hungarian government bonds, annual, ACT/365F (unique among CEE)
- CPI-linked: indexed to CPI_HU, 3-month lag, no deflation floor

References:
    MNB (2024). Magyar Nemzeti Bank — Market Conventions.
    BSE (2024). Budapest Stock Exchange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def _huf_swap_yf(start: date, end: date) -> float:
    """ACT/360 year fraction for swaps."""
    return year_fraction(start, end, DayCountConvention.ACT_360)


def _huf_bond_yf(start: date, end: date) -> float:
    """ACT/365F year fraction for bonds."""
    return year_fraction(start, end, DayCountConvention.ACT_365_FIXED)


def _next_bday(d: date, cal) -> date:
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


# ═══════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════

def synthetic_bubor_strip(reference_date: date, rate: float = 0.065,
                          n: int = 10, slope_bp: float = -10.0) -> list[dict]:
    """Synthetic BUBOR swap strip. Hungary: ~6.5% policy rate."""
    cal = get_calendar("HUF")
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

def build_huf_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap HUF discount curve from BUBOR strip. ACT/360, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_360, InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# BUBOR Swap (3M BUBOR floating, annual fixed)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BUBORSwapResult:
    pv: float; fixed_rate: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class BUBORSwap:
    """Hungarian BUBOR 3M interest rate swap.

    Fixed leg: annual, ACT/360.
    Floating leg: 3M BUBOR, quarterly reset (telescoping valuation).

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: annual fixed rate.
        notional: notional in HUF.
        direction: +1 = pay fixed, -1 = receive fixed.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1e9, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> BUBORSwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        dc = DayCountConvention.ACT_360
        schedule = generate_schedule(self.start, self.end, Frequency.ANNUAL)

        # Fixed leg: Σ fixed_rate × τ × df(t_i)
        fixed_pv = sum(self.fixed_rate * year_fraction(schedule[i-1], schedule[i], dc)
                       * curve.df(schedule[i]) for i in range(1, len(schedule)))

        # Floating (3M BUBOR): telescoping = df(start) - df(end)
        float_pv = curve.df(self.start) - curve.df(self.end)

        pv = self.direction * self.notional * (fixed_pv - float_pv)

        # Par rate
        annuity = sum(year_fraction(schedule[i-1], schedule[i], dc)
                      * curve.df(schedule[i]) for i in range(1, len(schedule)))
        par = float_pv / annuity if annuity > 0 else 0

        # DV01
        pv_up = self.direction * self.notional * (
            (self.fixed_rate + 0.0001) * annuity - float_pv)
        dv01 = abs(pv_up - pv)

        return BUBORSwapResult(pv, self.fixed_rate, par, dv01, self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "bubor_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# HUFONIA Swap (overnight OIS)
# ═══════════════════════════════════════════════════════════════

@dataclass
class HUFONIASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class HUFONIASwap:
    """Hungarian HUFONIA overnight swap. Annual fixed, ACT/360."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1e9, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> HUFONIASwapResult:
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
        return HUFONIASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "hufonia_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# HGB Bond (annual ACT/365F — unique among CEE)
# ═══════════════════════════════════════════════════════════════

class HGBBond:
    """Hungarian government bond. Annual coupon, ACT/365F.

    Unlike other CEE bonds (which use ACT/ACT ICMA), HGBs use ACT/365F.
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        dc = DayCountConvention.ACT_365_FIXED
        pv = sum(self.face * self.coupon * year_fraction(schedule[i-1], schedule[i], dc)
                 * curve.df(schedule[i]) for i in range(1, len(schedule)))
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "hgb", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon}


# ═══════════════════════════════════════════════════════════════
# HGB Linker (CPI_HU indexed, 3-month lag, no floor)
# ═══════════════════════════════════════════════════════════════

@dataclass
class HGBLinkerResult:
    real_price: float; nominal_price: float; cpi_ratio: float; real_yield: float
    def to_dict(self) -> dict: return vars(self)


class HGBLinker:
    """Hungarian CPI-linked government bond. CPI_HU indexed, 3-month lag, no floor.

    Annual coupon, ACT/365F. No deflation floor on principal.

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: annual real coupon rate.
        base_cpi: CPI_HU at issue (lagged 3 months).
        face: face value.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100.0):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve,
              current_cpi: float) -> HGBLinkerResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        cpi_ratio = current_cpi / self.base_cpi  # no deflation floor

        dc = DayCountConvention.ACT_365_FIXED
        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i], dc)
                  * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, dc)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return HGBLinkerResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "hgb_linker", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}
