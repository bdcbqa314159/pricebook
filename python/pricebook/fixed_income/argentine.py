"""Argentine fixed income derivatives.

Lecap (zero-coupon), Lecer (CER-linked), Bonares (ARS sovereign),
Globales (USD sovereign). Handles extreme rates (50-120%+).

    from pricebook.fixed_income.argentine import (
        LecapBond, LecerBond, BONARBond,
        build_ars_curve, synthetic_ars_strip,
    )

Conventions:
- Day count: ACT/365 for most instruments
- BADLAR: 30-day bank deposit rate (legacy), BCRA policy rate ~40-120%
- CER: daily inflation coefficient (Coeficiente de Estabilización de Referencia)
- Lecap: zero-coupon capitalisation bond (very short-dated, <1Y)
- Lecer: CER-adjusted (inflation-linked, daily accrual)
- Bonares: ARS-denominated sovereign
- Globales: USD-denominated sovereign (NY law)

References:
    BCRA (2024). Banco Central de la República Argentina.
    CNV (2024). Comisión Nacional de Valores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def synthetic_ars_strip(reference_date: date, policy_rate: float = 0.40,
                         n: int = 8, slope_bp: float = -200.0) -> list[dict]:
    """Synthetic ARS curve strip. Argentina: extreme rates, inverted.

    Policy rate ~40% (2024), curve heavily inverted on rate cut expectations.
    """
    cal = get_calendar("ARS")
    tenors = [1, 3, 6, 9, 12, 18, 24, 36][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = policy_rate + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_ars_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Build ARS discount curve. Handles extreme rates (>100%) without overflow."""
    from pricebook.core.interpolation import InterpolationMethod

    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    # Use exp(-r×T) — safe even for r=1.0 (100%) since r×T < ~3
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_365_FIXED, InterpolationMethod.LOG_LINEAR)


@dataclass
class LecapResult:
    price: float       # per 1,000 face
    rate: float        # capitalisation rate
    days: int

    def to_dict(self) -> dict:
        return dict(vars(self))


class LecapBond:
    """Lecap — Letra de Capitalización del Tesoro.

    Zero-coupon, short-dated (<1Y). Face R$ 1,000.
    Price = Face / (1 + rate)^(days/365).

    At extreme Argentine rates (40%+), even short Lecaps
    trade at significant discounts.
    """

    def __init__(self, maturity: date, rate: float, face: float = 1000.0):
        self.maturity = maturity
        self.rate = rate
        self.face = face

    def price(self, reference_date: date) -> LecapResult:
        days = (self.maturity - reference_date).days
        tau = days / 365
        px = self.face / (1 + self.rate) ** tau
        return LecapResult(px, self.rate, days)

    def to_dict(self) -> dict:
        return {"type": "lecap", "maturity": self.maturity.isoformat(), "rate": self.rate}


@dataclass
class LecerResult:
    real_price: float
    nominal_price: float
    cer_value: float
    real_yield: float

    def to_dict(self) -> dict:
        return dict(vars(self))


class LecerBond:
    """Lecer — Letra del Tesoro ajustada por CER.

    CER-adjusted (inflation-linked). Face accrues daily at CER.
    Similar to Brazil's NTN-B but zero-coupon and short-dated.
    """

    def __init__(self, issue_date: date, maturity: date, face: float = 1000.0):
        self.issue_date = issue_date
        self.maturity = maturity
        self.face = face

    def price(self, reference_date: date, real_curve: DiscountCurve,
              current_cer: float, cer_at_issue: float) -> LecerResult:
        # CER-adjusted face
        cer_ratio = current_cer / cer_at_issue
        adjusted_face = self.face * cer_ratio

        # Price = adjusted_face × df_real(maturity)
        df = real_curve.df(self.maturity)
        real_px = self.face * df  # in real terms
        nominal_px = adjusted_face * df

        T = year_fraction(reference_date, self.maturity, DayCountConvention.ACT_365_FIXED)
        ry = -math.log(max(df, 1e-15)) / max(T, 1e-10) if T > 0 else 0

        return LecerResult(real_px, nominal_px, current_cer, ry)

    def to_dict(self) -> dict:
        return {"type": "lecer", "maturity": self.maturity.isoformat()}


class BONARBond:
    """Bonares — ARS-denominated Argentine sovereign bond.

    Semi-annual coupon, ACT/365. At extreme ARS rates, trades
    at deep discounts.
    """

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
        return {"type": "bonar", "maturity": self.maturity.isoformat(), "coupon": self.coupon}


def breakeven_inflation_ar(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """ARS BEI from ARS nominal vs CER real curves. Extreme values expected (~30%+)."""
    from dateutil.relativedelta import relativedelta
    if maturities_years is None:
        maturities_years = [1, 2, 3, 5]
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
