"""Mexican fixed income derivatives.

TIIE 28D swaps, CETES discount bills, UDI-linked bonds (Udibonos),
and MBONO pricing via TIIE curve.

    from pricebook.fixed_income.mexican import (
        TIIESwap, CETESBill, UDIBond,
        build_tiie_curve, synthetic_tiie_strip,
    )

Conventions:
- Day count: ACT/360 for all instruments
- TIIE: 28-day tenor (unique — not overnight, not quarterly)
- CETES: zero-coupon discount bills (like T-Bills, ACT/360)
- Udibonos: UDI-linked (daily inflation unit, Banxico)
- MBONO: fixed-rate, semi-annual, ACT/360

References:
    Banxico (2024). Market Conventions.
    BMV (2024). TIIE Swap Specifications.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.calendar import get_calendar


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _mxn_year_fraction(start: date, end: date) -> float:
    """ACT/360 year fraction."""
    return year_fraction(start, end, DayCountConvention.ACT_360)


# ═══════════════════════════════════════════════════════════════
# Synthetic data
# ═══════════════════════════════════════════════════════════════

def synthetic_tiie_strip(
    reference_date: date,
    tiie_28d: float = 0.1125,
    n_contracts: int = 12,
    slope_bp_per_year: float = -20.0,
) -> list[dict]:
    """Generate realistic TIIE 28D swap strip.

    Mexico typically has inverted or flat curve near Banxico target.

    Args:
        tiie_28d: current TIIE 28D rate (~11.25% as of late 2024).
        n_contracts: number of swap tenors.
        slope_bp_per_year: term premium (negative = inverted).
    """
    cal = get_calendar("MXN")
    contracts = []

    tenors_months = [1, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84, 120][:n_contracts]
    for months in tenors_months:
        mat = reference_date + timedelta(days=months * 30)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)

        years = months / 12
        rate = tiie_28d + slope_bp_per_year / 10_000 * years
        tau = _mxn_year_fraction(reference_date, mat)

        contracts.append({
            "maturity": mat,
            "rate": rate,
            "tenor_months": months,
            "years": years,
            "tau": tau,
        })

    return contracts


def synthetic_cetes_quotes(
    reference_date: date,
    base_rate: float = 0.1100,
) -> list[dict]:
    """Generate realistic CETES quotes (28D, 91D, 182D, 364D)."""
    cal = get_calendar("MXN")
    tenors = [28, 91, 182, 364]
    quotes = []

    for days in tenors:
        mat = reference_date + timedelta(days=days)
        while not cal.is_business_day(mat):
            mat += timedelta(days=1)

        rate = base_rate + (days - 28) * 0.00005  # slight slope
        tau = _mxn_year_fraction(reference_date, mat)
        price = 10.0 / (1 + rate * tau)  # per MXN 10 face

        quotes.append({
            "maturity": mat,
            "days": days,
            "rate": rate,
            "price": price,
            "tau": tau,
        })

    return quotes


# ═══════════════════════════════════════════════════════════════
# TIIE Curve
# ═══════════════════════════════════════════════════════════════

def build_tiie_curve(
    reference_date: date,
    tiie_strip: list[dict],
) -> DiscountCurve:
    """Build TIIE discount curve from swap strip.

    Uses standard bootstrap: df(T) = 1 / (1 + rate × tau).
    """
    from pricebook.core.interpolation import InterpolationMethod

    pillar_dates = []
    pillar_dfs = []

    for contract in sorted(tiie_strip, key=lambda c: c["maturity"]):
        mat = contract["maturity"]
        rate = contract["rate"]
        tau = contract.get("tau", _mxn_year_fraction(reference_date, mat))
        df = 1.0 / (1 + rate * tau)

        pillar_dates.append(mat)
        pillar_dfs.append(df)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=DayCountConvention.ACT_360,
        interpolation=InterpolationMethod.LOG_LINEAR,
    )


# ═══════════════════════════════════════════════════════════════
# TIIE Swap
# ═══════════════════════════════════════════════════════════════

@dataclass
class TIIESwapResult:
    """TIIE 28D swap pricing result."""
    pv: float
    fixed_rate: float
    par_rate: float
    dv01: float
    n_periods: int
    notional: float

    def to_dict(self) -> dict:
        return vars(self)


class TIIESwap:
    """TIIE 28-day interest rate swap.

    Unique structure: periods are 28 calendar days (not business days).
    Fixed leg pays rate × 28/360 per period.
    Floating leg resets to TIIE 28D each period.

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: annual fixed rate.
        notional: swap notional in MXN.
        direction: +1 = pay fixed, -1 = receive fixed.
    """

    def __init__(self, start: date, end: date, fixed_rate: float,
                 notional: float = 100_000_000.0, direction: int = 1):
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.notional = notional
        self.direction = direction

    def price(self, tiie_curve: DiscountCurve) -> TIIESwapResult:
        """Price the TIIE swap."""
        # Number of 28-day periods
        total_days = (self.end - self.start).days
        n_periods = max(total_days // 28, 1)
        tau_period = 28 / 360  # ACT/360

        # Fixed leg PV: sum of fixed_rate × tau × df(t_i)
        fixed_pv = 0.0
        for i in range(1, n_periods + 1):
            t_i = self.start + timedelta(days=28 * i)
            if t_i > self.end:
                t_i = self.end
            df = tiie_curve.df(t_i)
            fixed_pv += self.fixed_rate * tau_period * df

        # Floating leg PV: telescoping (par - df(T))
        df_start = tiie_curve.df(self.start)
        df_end = tiie_curve.df(self.end)
        float_pv = df_start - df_end

        # PV = direction × notional × (fixed - float)
        pv = self.direction * self.notional * (fixed_pv - float_pv)

        # Par rate
        annuity = sum(tau_period * tiie_curve.df(self.start + timedelta(days=28 * i))
                       for i in range(1, n_periods + 1))
        par_rate = float_pv / annuity if annuity > 0 else 0

        # DV01
        pv_up = self.direction * self.notional * (
            (self.fixed_rate + 0.0001) * annuity - float_pv)
        dv01 = abs(pv_up - pv)

        return TIIESwapResult(pv, self.fixed_rate, par_rate, dv01, n_periods, self.notional)

    def pv_ctx(self, ctx) -> float:
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        return self.price(curve).pv

    def to_dict(self) -> dict:
        return {"type": "tiie_swap", "start": self.start.isoformat(),
                "end": self.end.isoformat(), "fixed_rate": self.fixed_rate,
                "notional": self.notional, "direction": self.direction}


# ═══════════════════════════════════════════════════════════════
# CETES (Mexican T-Bills)
# ═══════════════════════════════════════════════════════════════

@dataclass
class CETESResult:
    """CETES pricing result."""
    price: float           # per MXN 10 face
    discount_rate: float   # discount yield
    yield_rate: float      # equivalent yield
    days: int
    face: float

    def to_dict(self) -> dict:
        return vars(self)


class CETESBill:
    """CETES — Certificados de la Tesorería de la Federación.

    Zero-coupon discount bills, ACT/360. Face value MXN 10.
    Quoted as discount rate: Price = Face / (1 + rate × days/360).

    Standard tenors: 28D, 91D, 182D, 364D.
    """

    def __init__(self, maturity: date, rate: float, face: float = 10.0):
        self.maturity = maturity
        self.rate = rate
        self.face = face

    def price(self, reference_date: date) -> CETESResult:
        days = (self.maturity - reference_date).days
        tau = days / 360
        px = self.face / (1 + self.rate * tau)
        # Equivalent yield
        eq_yield = (self.face / px - 1) / tau if tau > 0 else 0

        return CETESResult(px, self.rate, eq_yield, days, self.face)

    def pv_ctx(self, ctx) -> float:
        return self.price(ctx.valuation_date).price

    def to_dict(self) -> dict:
        return {"type": "cetes", "maturity": self.maturity.isoformat(),
                "rate": self.rate, "face": self.face}


# ═══════════════════════════════════════════════════════════════
# UDI Bond (Udibono)
# ═══════════════════════════════════════════════════════════════

@dataclass
class UDIBondResult:
    """Udibono pricing result."""
    real_price: float      # price in UDI terms
    nominal_price: float   # price in MXN (real × UDI value)
    udi_value: float       # current UDI value
    real_yield: float      # real yield (UDI-denominated)

    def to_dict(self) -> dict:
        return vars(self)


class UDIBond:
    """Udibono — UDI-linked Mexican government bond.

    Coupon and principal denominated in UDI (Unidades de Inversión).
    UDI is a daily-published inflation unit (Banxico).
    Nominal cashflow = real cashflow × UDI_value(payment_date).

    Args:
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: annual real coupon rate (e.g. 0.04 = 4% real).
        udi_at_issue: UDI value at issue date.
        face_udi: face value in UDI (standard 100).
    """

    def __init__(self, issue_date: date, maturity: date, real_coupon: float,
                 udi_at_issue: float = 7.50, face_udi: float = 100.0):
        self.issue_date = issue_date
        self.maturity = maturity
        self.real_coupon = real_coupon
        self.udi_at_issue = udi_at_issue
        self.face_udi = face_udi

    def price(self, reference_date: date, real_curve: DiscountCurve,
              current_udi: float) -> UDIBondResult:
        """Price Udibono.

        Args:
            real_curve: real (UDI-denominated) discount curve.
            current_udi: current UDI value (MXN per UDI).
        """
        from pricebook.core.schedule import Frequency, generate_schedule

        schedule = generate_schedule(self.issue_date, self.maturity, Frequency.SEMI_ANNUAL)

        # Real price (in UDI)
        real_pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= reference_date:
                continue
            tau = _mxn_year_fraction(schedule[i-1], schedule[i])
            df = real_curve.df(schedule[i])
            real_pv += self.face_udi * self.real_coupon * tau * df

        # Principal at maturity
        real_pv += self.face_udi * real_curve.df(self.maturity)

        # Nominal price
        nominal_pv = real_pv * current_udi

        # Real yield (approximate)
        T = _mxn_year_fraction(reference_date, self.maturity)
        real_yield = -math.log(max(real_pv / self.face_udi, 1e-10)) / max(T, 1e-10) if T > 0 else 0

        return UDIBondResult(real_pv, nominal_pv, current_udi, real_yield)

    def pv_ctx(self, ctx) -> float:
        curve = ctx.discount_curve
        if curve is None:
            raise ValueError("No discount curve in context")
        # Assume UDI value from context or default
        udi = 7.80  # approximate current
        return self.price(ctx.valuation_date, curve, udi).nominal_price

    def to_dict(self) -> dict:
        return {"type": "udi_bond", "issue_date": self.issue_date.isoformat(),
                "maturity": self.maturity.isoformat(), "real_coupon": self.real_coupon,
                "udi_at_issue": self.udi_at_issue, "face_udi": self.face_udi}
