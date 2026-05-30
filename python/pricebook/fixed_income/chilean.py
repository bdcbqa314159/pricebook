"""Chilean fixed income derivatives.

UF (Unidad de Fomento) dual-currency pricing, Cámara swaps (TPM-based),
BCP (nominal) and BCU (UF-linked) sovereign bonds.

    from pricebook.fixed_income.chilean import (
        CamaraSwap, BCPBond, BCUBond,
        build_clp_curve, build_uf_curve, synthetic_clp_strip,
    )

Conventions:
- Day count: ACT/365F (Chile uses 365-day year, not 360)
- UF: daily-published inflation unit (~CLP 37,000), updated by BCCh
- BCP: nominal CLP bond, semi-annual, ACT/365
- BCU: UF-denominated bond (real rate), semi-annual
- Cámara: overnight TPM-based swap (like SOFR swap)

References:
    BCCh (2024). Banco Central de Chile — Market Conventions.
    Santiago Exchange (2024). Fixed Income Specifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


def _clp_yf(start: date, end: date) -> float:
    return year_fraction(start, end, DayCountConvention.ACT_365_FIXED)


# ═══════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════

def synthetic_clp_strip(
    reference_date: date,
    tpm: float = 0.0575,
    n_contracts: int = 10,
    slope_bp: float = 15.0,
) -> list[dict]:
    """Generate realistic CLP Cámara swap strip.

    Chile: TPM at 5.75%, slightly upward sloping.
    """
    cal = get_calendar("CLP")
    contracts = []
    tenors = [1, 3, 6, 9, 12, 18, 24, 36, 60, 120][:n_contracts]

    for months in tenors:
        mat = reference_date + timedelta(days=months * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        years = months / 12
        rate = tpm + slope_bp / 10_000 * years
        contracts.append({"maturity": mat, "rate": rate, "months": months, "years": years})

    return contracts


def synthetic_uf_strip(
    reference_date: date,
    real_rate: float = 0.02,
    n_contracts: int = 8,
    slope_bp: float = 10.0,
) -> list[dict]:
    """Generate realistic UF (real rate) swap strip."""
    cal = get_calendar("CLP")
    contracts = []
    tenors = [6, 12, 24, 36, 60, 84, 120, 240][:n_contracts]

    for months in tenors:
        mat = reference_date + timedelta(days=months * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        years = months / 12
        rate = real_rate + slope_bp / 10_000 * years
        contracts.append({"maturity": mat, "rate": rate, "months": months, "years": years})

    return contracts


# ═══════════════════════════════════════════════════════════════
# Curve construction
# ═══════════════════════════════════════════════════════════════

def build_clp_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Build CLP nominal discount curve from Cámara swap strip."""
    from pricebook.core.interpolation import InterpolationMethod

    pillar_dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    pillar_dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]

    return DiscountCurve(reference_date, pillar_dates, pillar_dfs,
                         day_count=DayCountConvention.ACT_365_FIXED,
                         interpolation=InterpolationMethod.LOG_LINEAR)


def build_uf_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Build UF (real) discount curve from UF swap strip."""
    from pricebook.core.interpolation import InterpolationMethod

    pillar_dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    pillar_dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]

    return DiscountCurve(reference_date, pillar_dates, pillar_dfs,
                         day_count=DayCountConvention.ACT_365_FIXED,
                         interpolation=InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# Cámara Swap (TPM overnight swap)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CamaraSwapResult:
    pv: float
    par_rate: float
    dv01: float
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


class CamaraSwap:
    """Chilean Cámara swap — fixed vs TPM overnight compounded.

    Similar to SOFR swap but based on BCCh's monetary policy rate (TPM).
    ACT/365 day count.

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: annual fixed rate.
        notional: in CLP.
        direction: +1 = pay fixed.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 1_000_000_000.0, direction: int = 1):
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.direction = direction

    def price(self, clp_curve: DiscountCurve) -> CamaraSwapResult:
        tau = _clp_yf(self.start, self.end)
        df_start = clp_curve.df(self.start)
        df_end = clp_curve.df(self.end)

        float_pv = df_start - df_end
        fixed_pv = self.fixed_rate * tau * df_end
        pv = self.direction * self.notional * (fixed_pv - float_pv)

        par_rate = float_pv / (tau * df_end) if tau * df_end > 0 else 0
        pv_up = self.direction * self.notional * ((self.fixed_rate + 0.0001) * tau * df_end - float_pv)
        dv01 = abs(pv_up - pv)

        return CamaraSwapResult(pv, par_rate, dv01, self.notional)

    def pv_ctx(self, ctx) -> float:
        return self.price(ctx.discount_curve).pv

    def to_dict(self) -> dict:
        return {"type": "camara_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# BCP Bond (nominal CLP sovereign)
# ═══════════════════════════════════════════════════════════════

class BCPBond:
    """BCP — Bono del Banco Central en Pesos.

    Nominal CLP government bond, semi-annual coupon, ACT/365.
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float,
                 face: float = 100.0):
        self.issue_date = issue_date
        self.maturity = maturity
        self.coupon = coupon
        self.face = face

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)

        pv = 0.0
        for i in range(1, len(schedule)):
            tau = _clp_yf(schedule[i-1], schedule[i])
            df = curve.df(schedule[i])
            pv += self.face * self.coupon * tau * df

        pv += self.face * curve.df(self.maturity)
        return pv

    def to_dict(self) -> dict:
        return {"type": "bcp_bond", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon}


# ═══════════════════════════════════════════════════════════════
# BCU Bond (UF-linked sovereign)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BCUResult:
    real_price: float      # in UF
    nominal_price: float   # in CLP
    uf_value: float
    real_yield: float

    def to_dict(self) -> dict:
        return vars(self)


class BCUBond:
    """BCU — Bono del Banco Central en UF.

    UF-denominated sovereign bond. Coupon and principal in UF.
    Nominal value = UF amount × UF/CLP exchange rate.

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: annual real coupon (e.g. 0.03 = 3% real).
        face_uf: face in UF (standard 1,000 UF).
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 face_uf: float = 1000.0):
        self.issue_date = issue_date
        self.maturity = maturity
        self.real_coupon = real_coupon
        self.face_uf = face_uf

    def price(self, reference_date: date, uf_curve: DiscountCurve,
              current_uf: float) -> BCUResult:
        """Price BCU in both UF and CLP terms.

        Args:
            uf_curve: real (UF) discount curve.
            current_uf: current UF value in CLP.
        """
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)

        real_pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= reference_date:
                continue
            tau = _clp_yf(schedule[i-1], schedule[i])
            df = uf_curve.df(schedule[i])
            real_pv += self.face_uf * self.real_coupon * tau * df

        real_pv += self.face_uf * uf_curve.df(self.maturity)
        nominal_pv = real_pv * current_uf

        T = _clp_yf(reference_date, self.maturity)
        real_yield = -math.log(max(real_pv / self.face_uf, 1e-10)) / max(T, 1e-10) if T > 0 else 0

        return BCUResult(real_pv, nominal_pv, current_uf, real_yield)

    def pv_ctx(self, ctx) -> float:
        curve = ctx.discount_curve
        uf = 37_000  # approximate
        return self.price(ctx.valuation_date, curve, uf).nominal_price

    def to_dict(self) -> dict:
        return {"type": "bcu_bond", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon, "face_uf": self.face_uf}


# ═══════════════════════════════════════════════════════════════
# Breakeven inflation from nominal + real curves
# ═══════════════════════════════════════════════════════════════

def breakeven_inflation(
    clp_curve: DiscountCurve,
    uf_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """Compute breakeven inflation from CLP nominal vs UF real curves.

    BEI = nominal_rate - real_rate (approximately).
    More precisely: (1+nominal) = (1+real)(1+BEI).
    """
    from dateutil.relativedelta import relativedelta

    if maturities_years is None:
        maturities_years = [1, 2, 3, 5, 7, 10, 20]

    ref = reference_date or clp_curve.reference_date
    results = []

    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom = clp_curve.df(mat)
        df_real = uf_curve.df(mat)

        if df_nom > 0 and df_real > 0 and T > 0:
            nom_rate = -math.log(df_nom) / T
            real_rate = -math.log(df_real) / T
            bei = nom_rate - real_rate
        else:
            bei = 0.0

        results.append({
            "maturity_years": T,
            "nominal_rate": nom_rate if df_nom > 0 else 0,
            "real_rate": real_rate if df_real > 0 else 0,
            "breakeven_inflation": bei,
            "bei_pct": bei * 100,
        })

    return results
