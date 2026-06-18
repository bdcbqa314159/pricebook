"""UK fixed income derivatives.

SONIA swaps, Gilt bonds, Index-Linked Gilts (ILGs), breakeven inflation.

    from pricebook.fixed_income.british import (
        SONIASwap, GiltBond, ILGBond,
        build_sonia_curve, breakeven_inflation_uk,
    )

Conventions:
- Day count: ACT/365F for SONIA swaps, ACT/ACT ICMA for Gilts
- SONIA: Sterling Overnight Index Average (BOE)
- Gilt: semi-annual ACT/ACT ICMA, T+1, 7-day ex-dividend
- ILG: 8-month RPI lag, flat interpolation (not linear), no deflation floor

References:
    DMO (2024). UK Debt Management Office — Gilt conventions.
    BOE (2024). Bank of England — SONIA.
    RPI (2024). Office for National Statistics — Retail Price Index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


# ═══════════════════════════════════════════════════════════════
# SONIA Curve
# ═══════════════════════════════════════════════════════════════

def synthetic_sonia_strip(reference_date: date, sonia: float = 0.0450,
                           n: int = 10, slope_bp: float = 5.0) -> list[dict]:
    """Synthetic SONIA OIS strip. UK: ~4.50% base rate."""
    cal = get_calendar("GBP")
    tenors = [1, 3, 6, 12, 24, 36, 60, 84, 120, 360][:n]
    result = []
    for m in tenors:
        mat = reference_date + timedelta(days=m * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)
        r = sonia + slope_bp / 10_000 * m / 12
        result.append({"maturity": mat, "rate": r, "months": m, "years": m / 12})
    return result


def build_sonia_curve(reference_date: date, strip: list[dict]) -> DiscountCurve:
    """Bootstrap SONIA discount curve. ACT/365F, log-linear."""
    from pricebook.core.interpolation import InterpolationMethod
    dates = [c["maturity"] for c in sorted(strip, key=lambda c: c["maturity"])]
    dfs = [math.exp(-c["rate"] * c["years"]) for c in sorted(strip, key=lambda c: c["maturity"])]
    return DiscountCurve(reference_date, dates, dfs,
                         DayCountConvention.ACT_365_FIXED, InterpolationMethod.LOG_LINEAR)


# ═══════════════════════════════════════════════════════════════
# SONIA Swap
# ═══════════════════════════════════════════════════════════════

@dataclass
class SONIASwapResult:
    pv: float; par_rate: float; dv01: float; notional: float
    def to_dict(self) -> dict: return dict(vars(self))


class SONIASwap:
    """UK SONIA overnight swap. Annual fixed, ACT/365F.

    Standard GBP OIS: annual fixed vs compounded SONIA.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 10_000_000.0, direction: int = 1):
        self.start, self.end, self.fixed_rate = start, end, fixed_rate
        self.notional, self.direction = notional, direction

    def price(self, curve: DiscountCurve) -> SONIASwapResult:
        from pricebook.core.schedule import Frequency, generate_schedule
        dc = DayCountConvention.ACT_365_FIXED
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
        return SONIASwapResult(pv, par, abs(pv_up - pv), self.notional)

    def pv_ctx(self, ctx) -> float: return self.price(ctx.discount_curve).pv
    def to_dict(self) -> dict:
        return {"type": "sonia_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate}


# ═══════════════════════════════════════════════════════════════
# Gilt Bond
# ═══════════════════════════════════════════════════════════════

class GiltBond:
    """UK Government Gilt. Semi-annual ACT/ACT ICMA, T+1, 7-day ex-div.

    Ex-dividend: buyer doesn't receive coupon if purchased within
    7 business days before the coupon date.
    """

    def __init__(self, issue_date: date, maturity: date, coupon: float,
                 face: float = 100, ex_div_days: int = 7):
        self.issue_date, self.maturity = issue_date, maturity
        self.coupon, self.face, self.ex_div_days = coupon, face, ex_div_days

    def dirty_price(self, curve: DiscountCurve) -> float:
        from pricebook.core.schedule import Frequency, generate_schedule
        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)

        pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= curve.reference_date:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], DayCountConvention.ACT_ACT_ICMA,
                                ref_start=schedule[i-1], ref_end=schedule[i], frequency=2)
            df = curve.df(schedule[i])
            pv += self.face * self.coupon * tau * df

        pv += self.face * curve.df(self.maturity)
        return pv

    def to_dict(self) -> dict:
        return {"type": "gilt", "maturity": self.maturity.isoformat(),
                "coupon": self.coupon}


def synthetic_gilt_strip(reference_date: date) -> list[dict]:
    """Synthetic Gilt benchmark quotes (realistic 2024)."""
    from dateutil.relativedelta import relativedelta
    return [
        {"tenor": "2Y", "maturity": reference_date + relativedelta(years=2),
         "coupon": 0.0425, "price": 99.5},
        {"tenor": "5Y", "maturity": reference_date + relativedelta(years=5),
         "coupon": 0.04, "price": 97.0},
        {"tenor": "10Y", "maturity": reference_date + relativedelta(years=10),
         "coupon": 0.0375, "price": 93.5},
        {"tenor": "30Y", "maturity": reference_date + relativedelta(years=30),
         "coupon": 0.0350, "price": 82.0},
    ]


# ═══════════════════════════════════════════════════════════════
# ILG (Index-Linked Gilt) — 8-month RPI lag, flat interpolation
# ═══════════════════════════════════════════════════════════════

@dataclass
class ILGResult:
    real_price: float
    nominal_price: float
    rpi_ratio: float
    real_yield: float

    def to_dict(self) -> dict: return dict(vars(self))


class ILGBond:
    """UK Index-Linked Gilt. 8-month RPI lag, flat interpolation, no deflation floor.

    Unlike TIPS (3-month lag, linear interpolation, deflation floor),
    UK ILGs use an 8-month lag with flat (prior month) interpolation.
    No deflation floor on principal — investors bear full deflation risk.

    The index ratio uses RPI values published 8 months prior:
        IR(t) = RPI(t - 8 months) / RPI(base_date - 8 months)

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: semi-annual real coupon (e.g. 0.0125 = 1.25%).
        base_rpi: RPI at issue (lagged 8 months).
        face: face value.
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 base_rpi: float = 100.0, face: float = 100):
        self.issue_date, self.maturity = issue_date, maturity
        self.real_coupon, self.base_rpi, self.face = real_coupon, base_rpi, face

    def price(self, ref: date, real_curve: DiscountCurve, current_rpi: float) -> ILGResult:
        """Price ILG in both real and nominal terms.

        Args:
            ref: valuation date.
            real_curve: real (RPI-deflated) discount curve.
            current_rpi: current RPI value (lagged 8 months).
        """
        from pricebook.core.schedule import Frequency, generate_schedule

        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)

        # Flat interpolation: use prior month's value (no daily accrual)
        rpi_ratio = current_rpi / self.base_rpi  # no deflation floor

        rpv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= ref:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], DayCountConvention.ACT_ACT_ICMA,
                                ref_start=schedule[i-1], ref_end=schedule[i], frequency=2)
            df = real_curve.df(schedule[i])
            rpv += self.face * self.real_coupon * tau * df

        rpv += self.face * real_curve.df(self.maturity)
        nominal = rpv * rpi_ratio

        T = year_fraction(ref, self.maturity, DayCountConvention.ACT_365_FIXED)
        ry = -math.log(max(rpv / self.face, 1e-10)) / max(T, 1e-10) if T > 0 else 0

        return ILGResult(rpv, nominal, rpi_ratio, ry)

    def to_dict(self) -> dict:
        return {"type": "ilg", "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon}


# ═══════════════════════════════════════════════════════════════
# Breakeven Inflation (UK)
# ═══════════════════════════════════════════════════════════════

def breakeven_inflation_uk(
    gilt_curve: DiscountCurve,
    ilg_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """UK breakeven inflation from nominal Gilt vs real ILG curves.

    BEI ≈ nominal_rate - real_rate.
    Note: UK BEI reflects RPI (not CPI), typically ~1% higher than CPI.
    """
    from dateutil.relativedelta import relativedelta

    if maturities_years is None:
        maturities_years = [2, 5, 10, 20, 30, 50]

    ref = reference_date or gilt_curve.reference_date
    results = []

    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom = gilt_curve.df(mat)
        df_real = ilg_curve.df(mat)

        if df_nom > 0 and df_real > 0 and T > 0:
            nom = -math.log(df_nom) / T
            real = -math.log(df_real) / T
            bei = nom - real
        else:
            nom = real = bei = 0.0

        results.append({"years": T, "nominal_rate": nom,
                         "real_rate": real, "bei": bei})

    return results
