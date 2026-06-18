"""Peruvian fixed income derivatives.

BTP Peru sovereign bonds, VAC inflation-linked bonds, PEN curve.

    from pricebook.fixed_income.peruvian import (
        BTPPeru, VACBond, build_pen_curve, synthetic_pen_strip,
    )

Conventions:
- Day count: ACT/360 for swaps, ACT/365 for bonds
- TIPM: overnight reference rate (BCRP)
- BTP: Bonos del Tesoro Público, semi-annual, ACT/365
- VAC: Valor Adquisitivo Constante — inflation-linked

References:
    BCRP (2024). Banco Central de Reserva del Perú.
    MEF (2024). Ministerio de Economía y Finanzas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def synthetic_pen_strip(reference_date: date, rate: float = 0.0525,
                         n: int = 8, slope_bp: float = 15.0) -> list[dict]:
    """Synthetic PEN swap strip. Peru: ~5.25% policy rate."""
    cal = get_calendar("PEN")
    tenors = [3, 6, 12, 24, 36, 60, 84, 120][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_pen_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_360, InterpolationMethod.LOG_LINEAR)


class BTPPeru:
    """Peruvian sovereign bond (Bono del Tesoro Público). Semi-annual, ACT/365."""
    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        pv = sum(self.face * self.coupon * year_fraction(schedule[i-1], schedule[i],
                 DayCountConvention.ACT_365_FIXED) * curve.df(schedule[i])
                 for i in range(1, len(schedule)))
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "btp_peru", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


@dataclass
class VACResult:
    real_price: float; nominal_price: float; ipc_value: float
    def to_dict(self) -> dict: return dict(vars(self))


class VACBond:
    """VAC bond — Peruvian inflation-linked sovereign."""
    def __init__(self, issue_date: date, maturity: date, real_coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.real_coupon, self.face = issue_date, maturity, real_coupon, face

    def price(self, ref: date, real_curve: DiscountCurve, ipc: float) -> VACResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)
        rpv = sum(self.face * self.real_coupon * year_fraction(schedule[i-1], schedule[i],
                  DayCountConvention.ACT_365_FIXED) * real_curve.df(schedule[i])
                  for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face * real_curve.df(self.maturity)
        return VACResult(rpv, rpv * ipc, ipc)

    def to_dict(self) -> dict:
        return {"type": "vac_bond", "maturity": self.maturity.isoformat()}


def breakeven_inflation_pe(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """PEN BEI from PEN nominal vs IPC real curves."""
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [1, 2, 5, 10, 20]
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
