"""Indian fixed income derivatives.

MIBOR swaps, GSEC bonds, IIB inflation-indexed bonds, breakeven inflation.

    from pricebook.fixed_income.indian import (
        MIBORSwap, GSECBond, IIBBond,
        build_inr_curve, breakeven_inflation_in,
    )

Conventions:
- Day count: ACT/365F for MIBOR swaps
- MIBOR: Mumbai Interbank Offered Rate (RBI)
- GSEC: semi-annual 30/360, T+1 (unique: only sovereign globally using 30/360)
- IIB: CPI_IN indexed, 3-month lag, semi-annual 30/360, deflation floor

References:
    RBI (2024). Reserve Bank of India -- MIBOR.
    MOF (2024). Ministry of Finance -- GSEC conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


_DC = DayCountConvention.ACT_365_FIXED
_DC_30360 = DayCountConvention.THIRTY_360


def synthetic_mibor_strip(reference_date: date, rate: float = 0.065,
                          n: int = 10, slope_bp: float = -5.0) -> list[dict]:
    """Synthetic MIBOR OIS strip. India: ~6.5% base rate."""
    cal = get_calendar("INR")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_inr_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap INR discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class MIBORSwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class MIBORSwap:
    """Indian MIBOR overnight swap. Annual fixed, ACT/365F."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> MIBORSwapResult:
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
        return MIBORSwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "mibor_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ===================================================================
# GSEC (Government Securities) -- semi-annual 30/360, T+1
# Only sovereign bond globally using 30/360 day count convention.
# ===================================================================

class GSECBond:
    """Indian Government Security. Semi-annual 30/360, T+1.

    Unique among sovereign bonds: GSEC is the only sovereign globally
    that uses the 30/360 day count convention.
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= curve.reference_date:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], _DC_30360)
            pv += self.face * self.coupon * tau * curve.df(schedule[i])
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "gsec", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


# ===================================================================
# IIB (India Inflation-Indexed Bond) -- CPI_IN, 3-month lag, 30/360
# ===================================================================

@dataclass
class IIBResult:
    real_price: float; nominal_price: float; cpi_ratio: float; real_yield: float
    def to_dict(self) -> dict: return vars(self)


class IIBBond:
    """Indian Inflation-Indexed Bond.

    CPI_IN indexed, 3-month lag, semi-annual 30/360 (matching GSEC),
    deflation floor on principal.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_cpi: float) -> IIBResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        cpi_ratio = max(current_cpi / self.base_cpi, 1.0)  # deflation floor

        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i], _DC_30360)
                  * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, _DC_30360)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return IIBResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "iib", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


# ===================================================================
# Breakeven Inflation
# ===================================================================

def breakeven_inflation_in(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """Indian breakeven inflation from MIBOR nominal vs IIB real curves.

    India BEI typically ~4-6% (RBI targets 4% +/- 2%).
    """
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [2, 5, 10, 20, 30]
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
