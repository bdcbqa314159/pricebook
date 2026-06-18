"""Czech fixed income derivatives.

PRIBOR swaps, CZEONIA OIS, CZGB sovereign bonds, and CPI-linked bonds.

    from pricebook.fixed_income.czech import (
        PRIBORSwap, CZEONIASwap, CZGBBond, CZGBLinker,
        build_czk_curve, synthetic_pribor_strip,
    )

Conventions:
- Day count: ACT/360 for PRIBOR/CZEONIA swaps
- PRIBOR: 3M Prague Interbank Offered Rate (CNB)
- CZEONIA: Czech Overnight Index Average (CNB)
- CZGB: Czech government bonds, annual, ACT/ACT ICMA
- CPI-linked: indexed to CPI_CZ, 3-month lag, no deflation floor

References:
    CNB (2024). Czech National Bank — Market Conventions.
    PSE (2024). Prague Stock Exchange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def _czk_yf(start: date, end: date) -> float:
    """ACT/360 year fraction."""
    return year_fraction(start, end, DayCountConvention.ACT_360)


def _next_bday(d: date, cal) -> date:
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


# ═══════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════

def synthetic_pribor_strip(reference_date: date, rate: float = 0.045,
                           n: int = 10, slope_bp: float = -5.0) -> list[dict]:
    """Synthetic PRIBOR swap strip. Czech Republic: ~4.5% policy rate."""
    cal = get_calendar("CZK")
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

def build_czk_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap CZK discount curve from PRIBOR strip. ACT/360, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_360, InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# PRIBOR Swap (3M PRIBOR floating, annual fixed)
# ═══════════════════════════════════════════════════════════════

@dataclass
class PRIBORSwapResult:
    pv: float; fixed_rate: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class PRIBORSwap:
    """Czech PRIBOR 3M interest rate swap.

    Fixed leg: annual, ACT/360.
    Floating leg: 3M PRIBOR, quarterly reset (telescoping valuation).

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: annual fixed rate.
        notional: notional in CZK.
        direction: +1 = pay fixed, -1 = receive fixed.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1e9, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> PRIBORSwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        dc = DayCountConvention.ACT_360
        schedule = generate_schedule(self.start, self.end, Frequency.ANNUAL)

        # Fixed leg: Σ fixed_rate × τ × df(t_i)
        fixed_pv = sum(self.fixed_rate * year_fraction(schedule[i-1], schedule[i], dc)
                       * curve.df(schedule[i]) for i in range(1, len(schedule)))

        # Floating (3M PRIBOR): telescoping = df(start) - df(end)
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

        return PRIBORSwapResult(pv, self.fixed_rate, par, dv01, self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "pribor_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# CZEONIA Swap (overnight OIS)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CZEONIASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class CZEONIASwap:
    """Czech CZEONIA overnight swap. Annual fixed, ACT/360."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1e9, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> CZEONIASwapResult:
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
        return CZEONIASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "czeonia_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# CZGB Bond (annual ACT/ACT ICMA)
# ═══════════════════════════════════════════════════════════════

class CZGBBond:
    """Czech government bond. Annual coupon, ACT/ACT ICMA."""

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
        return {"type": "czgb", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon}


# ═══════════════════════════════════════════════════════════════
# CZGB Linker (CPI_CZ indexed, 3-month lag, no floor)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CZGBLinkerResult:
    real_price: float; nominal_price: float; cpi_ratio: float; real_yield: float
    def to_dict(self) -> dict: return dict(vars(self))


class CZGBLinker:
    """Czech CPI-linked government bond. CPI_CZ indexed, 3-month lag, no floor.

    Annual coupon, ACT/365F. No deflation floor on principal.

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: annual real coupon rate.
        base_cpi: CPI_CZ at issue (lagged 3 months).
        face: face value.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100.0):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve,
              current_cpi: float) -> CZGBLinkerResult:
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
        return CZGBLinkerResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "czgb_linker", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


def breakeven_inflation_cz(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """CZK BEI from CZK nominal vs CPI_CZ real curves."""
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [2, 5, 10, 20]
    ref = reference_date or nominal_curve.reference_date
    results = []
    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom, df_real = nominal_curve.df(mat), real_curve.df(mat)
        if df_nom > 0 and df_real > 0 and T > 0:
            nom, real = -math.log(df_nom) / T, -math.log(df_real) / T
            bei = nom - real
        else:
            nom = real = bei = 0.0
        results.append({"years": T, "nominal_rate": nom, "real_rate": real, "bei": bei})
    return results
