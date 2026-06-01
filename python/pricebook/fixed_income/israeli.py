"""Israeli fixed income derivatives.

Telbor swaps, Shahar bonds, Galil inflation-linked bonds, breakeven inflation.

    from pricebook.fixed_income.israeli import (
        TelborSwap, ShaharBond, GalilBond,
        build_ils_curve, breakeven_inflation_il,
    )

Conventions:
- Day count: ACT/365F for Telbor swaps and Shahar bonds
- Telbor: Tel Aviv Interbank Offered Rate (BOI)
- Shahar: annual ACT/365F, T+1
- Galil: CPI_IL indexed, 1-month lag, annual ACT/365F, no deflation floor

References:
    BOI (2024). Bank of Israel — Telbor.
    MOF (2024). Ministry of Finance Israel — Shahar/Galil conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


_DC = DayCountConvention.ACT_365_FIXED


def synthetic_telbor_strip(reference_date: date, telbor: float = 0.045,
                            n: int = 10, slope_bp: float = 5.0) -> list[dict]:
    """Synthetic Telbor strip. Israel: ~4.5% base rate."""
    cal = get_calendar("ILS")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = telbor + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_ils_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap Telbor discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class TelborSwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class TelborSwap:
    """Israeli Telbor swap. Annual fixed, ACT/365F."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> TelborSwapResult:
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
        return TelborSwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "telbor_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# Shahar (Israeli nominal government bond) — annual ACT/365F, T+1
# ═══════════════════════════════════════════════════════════════

class ShaharBond:
    """Israeli Shahar government bond. Annual ACT/365F, T+1."""

    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= curve.reference_date:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], _DC)
            pv += self.face * self.coupon * tau * curve.df(schedule[i])
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "shahar", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


# ═══════════════════════════════════════════════════════════════
# Galil (Israeli inflation-linked bond) — CPI_IL, 1-month lag, no deflation floor
# ═══════════════════════════════════════════════════════════════

@dataclass
class GalilResult:
    real_price: float
    nominal_price: float
    cpi_ratio: float
    real_yield: float

    def to_dict(self) -> dict: return vars(self)


class GalilBond:
    """Galil — Israeli inflation-linked government bond.

    CPI_IL indexed, 1-month lag (shortest lag globally), ACT/365F, annual coupon,
    no deflation floor. Unlike TIPS (3-month lag, deflation floor) and UK ILGs
    (8-month lag, no floor), Galil uses a 1-month CPI lag for faster inflation
    pass-through.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_cpi: float) -> GalilResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        cpi_ratio = current_cpi / self.base_cpi  # no deflation floor

        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i], _DC)
                  * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, _DC)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return GalilResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "galil", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


# ═══════════════════════════════════════════════════════════════
# Breakeven inflation
# ═══════════════════════════════════════════════════════════════

def breakeven_inflation_il(
    ils_curve: DiscountCurve,
    galil_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """Israeli breakeven inflation from Telbor nominal vs Galil real curves.

    Israel BEI typically ~2-3% (inflation-targeting regime since 1990s).
    """
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [2, 5, 10, 20, 30]
    ref = reference_date or ils_curve.reference_date
    results = []
    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom, df_real = ils_curve.df(mat), galil_curve.df(mat)
        if df_nom > 0 and df_real > 0 and T > 0:
            nom, real = -math.log(df_nom) / T, -math.log(df_real) / T
            bei = nom - real
        else:
            nom = real = bei = 0.0
        results.append({"years": T, "nominal_rate": nom, "real_rate": real, "bei": bei})
    return results
