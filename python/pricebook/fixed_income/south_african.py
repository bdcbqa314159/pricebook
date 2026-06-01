"""South African fixed income derivatives.

JIBAR swaps, SAGB bonds, SAILB inflation-linked bonds, breakeven inflation.

    from pricebook.fixed_income.south_african import (
        JIBARSwap, SAGBBond, SAILBBond,
        build_zar_curve, breakeven_inflation_za,
    )

Conventions:
- Day count: ACT/365F for JIBAR swaps and SAGBs
- JIBAR: Johannesburg Interbank Average Rate (SARB)
- SAGB: semi-annual ACT/365F, T+3 (unique to South Africa)
- SAILB: CPI_ZA indexed, 3-month lag, semi-annual ACT/365F, no deflation floor

References:
    SARB (2024). South African Reserve Bank — JIBAR.
    National Treasury (2024). Republic of South Africa — SAGB conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


_DC = DayCountConvention.ACT_365_FIXED


def synthetic_jibar_strip(reference_date: date, jibar: float = 0.0825,
                           n: int = 10, slope_bp: float = -5.0) -> list[dict]:
    """Synthetic JIBAR strip. South Africa: ~8.25% base rate."""
    cal = get_calendar("ZAR")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = jibar + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_zar_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap JIBAR discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class JIBARSwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return vars(self)


class JIBARSwap:
    """South African JIBAR swap. Quarterly fixed, ACT/365F.

    ZAR IRS convention is quarterly fixed — unique among major markets.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> JIBARSwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.start, self.end, Frequency.QUARTERLY)
        fixed_pv = sum(self.fixed_rate * year_fraction(schedule[i-1], schedule[i], _DC)
                       * curve.df(schedule[i]) for i in range(1, len(schedule)))
        float_pv = curve.df(self.start) - curve.df(self.end)
        pv = self.direction * self.notional * (fixed_pv - float_pv)
        annuity = sum(year_fraction(schedule[i-1], schedule[i], _DC)
                      * curve.df(schedule[i]) for i in range(1, len(schedule)))
        par = float_pv / annuity if annuity > 0 else 0
        pv_up = self.direction * self.notional * (
            (self.fixed_rate + 0.0001) * annuity - float_pv)
        return JIBARSwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "jibar_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# SAGB (South African Government Bond) — semi-annual ACT/365F, T+3
# ═══════════════════════════════════════════════════════════════

class SAGBBond:
    """South African Government Bond. Semi-annual ACT/365F, T+3.

    South Africa uses T+3 settlement — unique among major sovereign markets.
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
            tau = year_fraction(schedule[i-1], schedule[i], _DC)
            pv += self.face * self.coupon * tau * curve.df(schedule[i])
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "sagb", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


# ═══════════════════════════════════════════════════════════════
# SAILB (SA Inflation-Linked Bond) — CPI_ZA, 3-month lag, no deflation floor
# ═══════════════════════════════════════════════════════════════

@dataclass
class SAILBResult:
    real_price: float
    nominal_price: float
    cpi_ratio: float
    real_yield: float

    def to_dict(self) -> dict: return vars(self)


class SAILBBond:
    """SAILB — South African Inflation-Linked Bond.

    CPI_ZA indexed, 3-month lag, ACT/365F, semi-annual, no deflation floor.
    Unlike TIPS (deflation floor on principal), SAILB investors bear full deflation risk.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_cpi: float) -> SAILBResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        cpi_ratio = current_cpi / self.base_cpi  # no deflation floor

        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i], _DC)
                  * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, _DC)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return SAILBResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "sailb", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


# ═══════════════════════════════════════════════════════════════
# Breakeven inflation
# ═══════════════════════════════════════════════════════════════

def breakeven_inflation_za(
    zar_curve: DiscountCurve,
    ilb_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """South African breakeven inflation from JIBAR nominal vs SAILB real curves.

    SA BEI typically ~5-7% (higher inflation environment).
    """
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [2, 5, 10, 20, 30]
    ref = reference_date or zar_curve.reference_date
    results = []
    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom, df_real = zar_curve.df(mat), ilb_curve.df(mat)
        if df_nom > 0 and df_real > 0 and T > 0:
            nom, real = -math.log(df_nom) / T, -math.log(df_real) / T
            bei = nom - real
        else:
            nom = real = bei = 0.0
        results.append({"years": T, "nominal_rate": nom, "real_rate": real, "bei": bei})
    return results
