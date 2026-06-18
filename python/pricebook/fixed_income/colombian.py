"""Colombian fixed income derivatives.

IBR swaps, TES fixed-rate bonds, TES UVR (inflation-linked).

    from pricebook.fixed_income.colombian import (
        IBRSwap, TESBond, TESUVRBond,
        build_ibr_curve, synthetic_ibr_strip,
    )

Conventions:
- Day count: ACT/360 for swaps, ACT/365 for bonds
- IBR: overnight rate (Indicador Bancario de Referencia), BanRep
- TES: Títulos de Tesorería, annual coupon, ACT/365
- UVR: Unidad de Valor Real, daily inflation unit (BanRep)

References:
    BanRep (2024). Banco de la República — Market Conventions.
    BVC (2024). Bolsa de Valores de Colombia.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def synthetic_ibr_strip(reference_date: date, ibr: float = 0.0975, n: int = 10,
                         slope_bp: float = -10.0) -> list[dict]:
    """Synthetic IBR swap strip. Colombia: ~9.75% policy rate."""
    cal = get_calendar("COP")
    tenors = [1, 3, 6, 9, 12, 18, 24, 36, 60, 120][:n]
    return [{"maturity": _next_bday(reference_date + timedelta(days=m*30), cal),
             "rate": ibr + slope_bp/10_000 * m/12, "months": m, "years": m/12}
            for m in tenors]


def _next_bday(d: date, cal) -> date:
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


def build_ibr_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_360, InterpolationMethod.LOG_LINEAR)


@dataclass
class IBRSwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class IBRSwap:
    """Colombian IBR overnight swap. ACT/360."""
    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1e9, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> IBRSwapResult:
        tau = year_fraction(self.start, self.end, DayCountConvention.ACT_360)
        df_s, df_e = curve.df(self.start), curve.df(self.end)
        float_pv = df_s - df_e
        fixed_pv = self.fixed_rate * tau * df_e
        pv = self.direction * self.notional * (fixed_pv - float_pv)
        par = float_pv / (tau * df_e) if tau * df_e > 0 else 0
        pv_up = self.direction * self.notional * ((self.fixed_rate + 0.0001) * tau * df_e - float_pv)
        return IBRSwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "ibr_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


class TESBond:
    """TES — Colombian government bond. Annual coupon, ACT/365."""
    def __init__(self, issue_date: date, maturity: date, coupon: float, face: float = 100):
        self.issue_date, self.maturity, self.coupon, self.face = issue_date, maturity, coupon, face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        pv = sum(self.face * self.coupon * year_fraction(schedule[i-1], schedule[i], DayCountConvention.ACT_365_FIXED)
                 * curve.df(schedule[i]) for i in range(1, len(schedule)))
        return pv + self.face * curve.df(self.maturity)

    def to_dict(self) -> dict:
        return {"type": "tes_bond", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


@dataclass
class TESUVRResult:
    real_price: float; nominal_price: float; uvr_value: float; real_yield: float
    def to_dict(self) -> dict: return dict(vars(self))


class TESUVRBond:
    """TES UVR — Colombian inflation-linked bond (UVR-denominated)."""
    def __init__(self, issue_date: date, maturity: date, real_coupon: float, face_uvr: float = 100):
        self.issue_date, self.maturity, self.real_coupon, self.face_uvr = issue_date, maturity, real_coupon, face_uvr

    def price(self, ref: date, uvr_curve: DiscountCurve, current_uvr: float) -> TESUVRResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.ANNUAL)
        rpv = sum(self.face_uvr * self.real_coupon * year_fraction(schedule[i-1], schedule[i], DayCountConvention.ACT_365_FIXED)
                  * uvr_curve.df(schedule[i]) for i in range(1, len(schedule)) if schedule[i] > ref)
        rpv += self.face_uvr * uvr_curve.df(self.maturity)
        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        ry = -math.log(max(rpv/self.face_uvr, 1e-10)) / max(T, 1e-10) if T > 0 else 0
        return TESUVRResult(rpv, rpv * current_uvr, current_uvr, ry)

    def to_dict(self) -> dict:
        return {"type": "tes_uvr", "maturity": self.maturity.isoformat(), "real_coupon": self.real_coupon}


def breakeven_inflation_co(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """COP BEI from IBR nominal vs UVR real curves."""
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [1, 2, 3, 5, 10, 20]
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
