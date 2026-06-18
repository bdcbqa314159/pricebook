"""Australian fixed income derivatives.

AONIA swaps, ACGB bonds, TIB inflation-linked bonds, breakeven inflation.

    from pricebook.fixed_income.australian import (
        AONIASwap, ACGBBond, TIBBond,
        build_aud_curve, breakeven_inflation_au,
    )

Conventions:
- Day count: ACT/365F for AONIA swaps and ACGBs
- AONIA: AUD Overnight Index Average (RBA)
- ACGB: semi-annual ACT/ACT ICMA, T+2
- TIB: CPI_AU indexed, quarterly coupon, ACT/ACT ICMA, no deflation floor

References:
    RBA (2024). Reserve Bank of Australia — AONIA.
    AOFM (2024). Australian Office of Financial Management — ACGB conventions.
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


def synthetic_aonia_strip(reference_date: date, aonia: float = 0.0435,
                          n: int = 10, slope_bp: float = 5.0) -> list[dict]:
    """Synthetic AONIA OIS strip. Australia: ~4.35% base rate."""
    cal = get_calendar("AUD")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = aonia + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_aud_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap AONIA discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs, _DC, InterpolationMethod.LOG_LINEAR)


@dataclass
class AONIASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class AONIASwap:
    """Australian AONIA overnight swap. Annual fixed, ACT/365F."""

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> AONIASwapResult:
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
        return AONIASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "aonia_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# ACGB (Australian Commonwealth Government Bond) — semi-annual ACT/ACT ICMA
# ═══════════════════════════════════════════════════════════════

class ACGBBond:
    """Australian Commonwealth Government Bond. Semi-annual ACT/ACT ICMA, T+2."""

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
        return {"type": "acgb", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


def synthetic_acgb_strip(reference_date: date) -> list[dict]:
    """Synthetic ACGB benchmark quotes (realistic 2024)."""
    from dateutil.relativedelta import relativedelta
    return [
        {"tenor": "2Y", "maturity": reference_date + relativedelta(years=2),
         "coupon": 0.0425, "price": 99.2},
        {"tenor": "5Y", "maturity": reference_date + relativedelta(years=5),
         "coupon": 0.04, "price": 96.5},
        {"tenor": "10Y", "maturity": reference_date + relativedelta(years=10),
         "coupon": 0.0375, "price": 92.0},
        {"tenor": "30Y", "maturity": reference_date + relativedelta(years=30),
         "coupon": 0.035, "price": 80.0},
    ]


# ═══════════════════════════════════════════════════════════════
# TIB (Treasury Indexed Bond) — CPI_AU, quarterly coupon, no deflation floor
# ═══════════════════════════════════════════════════════════════

@dataclass
class TIBResult:
    real_price: float
    nominal_price: float
    cpi_ratio: float
    real_yield: float

    def to_dict(self) -> dict: return dict(vars(self))


class TIBBond:
    """TIB — Australian Treasury Indexed Bond.

    CPI_AU indexed, quarterly coupon, ACT/ACT ICMA, no deflation floor.
    The ONLY quarterly sovereign linker globally — unique to Australia.
    Unlike TIPS (semi-annual, deflation floor) and UK ILGs (semi-annual, 8M lag),
    TIBs pay quarterly coupons with no principal floor.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_cpi: float = 100.0, face: float = 100):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_cpi, self.face = real_coupon, base_cpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_cpi: float) -> TIBResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.QUARTERLY)
        cpi_ratio = current_cpi / self.base_cpi  # no deflation floor

        rpv = sum(self.face * self.real_coupon * year_fraction(
                      schedule[i-1], schedule[i], _DC_ICMA,
                      ref_start=schedule[i-1], ref_end=schedule[i], frequency=4)
                  * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * cpi_ratio

        T = year_fraction(ref, self.maturity, _DC)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return TIBResult(rpv, nominal, cpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "tib", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


# ═══════════════════════════════════════════════════════════════
# Breakeven inflation
# ═══════════════════════════════════════════════════════════════

def breakeven_inflation_au(
    aud_curve: DiscountCurve,
    tib_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """Australian breakeven inflation from AONIA nominal vs TIB real curves.

    Australia BEI typically ~2.5-3.5% (inflation-targeting regime).
    """
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [2, 5, 10, 20, 30]
    ref = reference_date or aud_curve.reference_date
    results = []
    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom, df_real = aud_curve.df(mat), tib_curve.df(mat)
        if df_nom > 0 and df_real > 0 and T > 0:
            nom, real = -math.log(df_nom) / T, -math.log(df_real) / T
            bei = nom - real
        else:
            nom = real = bei = 0.0
        results.append({"years": T, "nominal_rate": nom, "real_rate": real, "bei": bei})
    return results
