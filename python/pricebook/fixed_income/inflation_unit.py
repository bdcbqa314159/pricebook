"""Unified framework for daily inflation units of account.

Several LatAm countries publish daily inflation-adjusted units:
- UDI (Mexico) — Unidades de Inversión, Banxico
- UF  (Chile)  — Unidad de Fomento, BCCh
- UVR (Colombia) — Unidad de Valor Real, BanRep
- CER (Argentina) — Coeficiente de Estabilización de Referencia, BCRA

These are structurally identical: a daily-published number that converts
between nominal currency and "real" (inflation-adjusted) terms. Bonds
denominated in these units have real coupons / principal.

This module provides:
- `InflationUnit` — definition of a unit (name, currency, properties).
- `InflationUnitBond` — generic bond denominated in any inflation unit.
- `dual_curve_breakeven()` — BEI from nominal + real curves.
- `compare_units()` — cross-country comparison of inflation unit properties.

    from pricebook.fixed_income.inflation_unit import (
        get_inflation_unit, InflationUnitBond, dual_curve_breakeven,
    )

    unit = get_inflation_unit("UDI")
    bond = InflationUnitBond("UDI", issue, maturity, 0.04, current_unit=8.2)
    result = bond.price(ref, real_curve)

References:
    Banxico (2024). UDI methodology.
    BCCh (2024). Unidad de Fomento calculation.
    BanRep (2024). UVR methodology.
    BCRA (2024). CER calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve


# ═══════════════════════════════════════════════════════════════
# Inflation unit definitions
# ═══════════════════════════════════════════════════════════════

class UnitPublisher(Enum):
    """Central bank or entity publishing the unit."""
    BANXICO = "Banxico"
    BCCH = "BCCh"
    BANREP = "BanRep"
    BCRA = "BCRA"


@dataclass(frozen=True)
class InflationUnit:
    """Definition of a daily inflation unit of account."""
    name: str               # "UDI", "UF", "UVR", "CER"
    currency: str           # nominal currency (MXN, CLP, COP, ARS)
    publisher: UnitPublisher
    day_count: DayCountConvention
    coupon_frequency_months: int  # 6 = semi-annual, 12 = annual
    base_year: int          # year unit was introduced
    typical_value: float    # approximate current value for sanity checks
    description: str


_UNITS: dict[str, InflationUnit] = {}


def _reg(u: InflationUnit) -> None:
    _UNITS[u.name] = u


_reg(InflationUnit(
    "UDI", "MXN", UnitPublisher.BANXICO,
    DayCountConvention.ACT_360, 6, 2002, 8.2,
    "Unidades de Inversión. Daily inflation index (Banxico). "
    "Base April 4, 1995 = 1.0. Used by Udibonos."))

_reg(InflationUnit(
    "UF", "CLP", UnitPublisher.BCCH,
    DayCountConvention.ACT_365_FIXED, 6, 1967, 37_000.0,
    "Unidad de Fomento. Daily inflation unit (BCCh). "
    "Calculated from prior month CPI. Used by BCU bonds."))

_reg(InflationUnit(
    "UVR", "COP", UnitPublisher.BANREP,
    DayCountConvention.ACT_365_FIXED, 12, 2000, 350.0,
    "Unidad de Valor Real. Daily inflation unit (BanRep). "
    "Base January 15, 2000 = 100. Used by TES UVR."))

_reg(InflationUnit(
    "CER", "ARS", UnitPublisher.BCRA,
    DayCountConvention.ACT_365_FIXED, 6, 2002, 1_200.0,
    "Coeficiente de Estabilización de Referencia. Daily coefficient (BCRA). "
    "Base February 2, 2002 = 1.0. Used by Lecer, Boncer."))


def get_inflation_unit(name: str) -> InflationUnit:
    """Look up an inflation unit by name."""
    key = name.upper()
    u = _UNITS.get(key)
    if u is None:
        raise ValueError(f"Unknown inflation unit {name!r}. "
                         f"Available: {sorted(_UNITS.keys())}")
    return u


def list_inflation_units() -> list[str]:
    """Return sorted list of inflation unit names."""
    return sorted(_UNITS.keys())


# ═══════════════════════════════════════════════════════════════
# Inflation unit bond
# ═══════════════════════════════════════════════════════════════

@dataclass
class InflationUnitBondResult:
    """Result of pricing a bond denominated in an inflation unit."""
    real_price: float       # price in unit terms (UDI, UF, etc.)
    nominal_price: float    # price in local currency
    unit_value: float       # current unit value used
    unit_name: str          # "UDI", "UF", etc.
    real_yield: float       # continuously compounded real yield
    indexation_ratio: float # current_unit / base_unit (if provided)
    currency: str

    def to_dict(self) -> dict:
        return vars(self)


class InflationUnitBond:
    """Bond denominated in a daily inflation unit (UDI, UF, UVR, CER).

    This is a generic instrument that works for any of the four LatAm
    daily inflation units. The bond pays real coupons in unit terms;
    nominal value = real value × current unit value.

    Args:
        unit_name: "UDI", "UF", "UVR", or "CER".
        issue_date: bond issue date.
        maturity: maturity date.
        real_coupon: annual real coupon rate (e.g. 0.04 = 4%).
        face_units: face value in inflation units (default 100).
        base_unit_value: unit value at issue (for indexation ratio).
    """

    def __init__(self, unit_name: str, issue_date: date, maturity: date,
                 real_coupon: float, face_units: float = 100.0,
                 base_unit_value: float | None = None):
        self.unit = get_inflation_unit(unit_name)
        self.issue_date = issue_date
        self.maturity = maturity
        self.real_coupon = real_coupon
        self.face_units = face_units
        self.base_unit_value = base_unit_value

    def price(self, reference_date: date, real_curve: DiscountCurve,
              current_unit_value: float) -> InflationUnitBondResult:
        """Price the bond in both real (unit) and nominal terms.

        Args:
            reference_date: valuation date.
            real_curve: discount curve in real (unit) terms.
            current_unit_value: current value of the inflation unit.
        """
        from pricebook.core.schedule import Frequency, generate_schedule

        freq_months = self.unit.coupon_frequency_months
        freq = {6: Frequency.SEMI_ANNUAL, 12: Frequency.ANNUAL,
                3: Frequency.QUARTERLY}.get(freq_months, Frequency.SEMI_ANNUAL)

        schedule = generate_schedule(self.issue_date, self.maturity, freq)
        dc = self.unit.day_count

        # Real PV: discount real coupon flows
        real_pv = 0.0
        for i in range(1, len(schedule)):
            if schedule[i] <= reference_date:
                continue
            tau = year_fraction(schedule[i-1], schedule[i], dc)
            df = real_curve.df(schedule[i])
            real_pv += self.face_units * self.real_coupon * tau * df

        # Principal at maturity
        real_pv += self.face_units * real_curve.df(self.maturity)

        # Nominal conversion
        nominal_pv = real_pv * current_unit_value

        # Real yield
        T = year_fraction(reference_date, self.maturity, dc)
        if T > 0 and real_pv > 0:
            real_yield = -math.log(real_pv / self.face_units) / T
        else:
            real_yield = 0.0

        # Indexation ratio
        if self.base_unit_value and self.base_unit_value > 0:
            idx_ratio = current_unit_value / self.base_unit_value
        else:
            idx_ratio = 1.0

        return InflationUnitBondResult(
            real_pv, nominal_pv, current_unit_value, self.unit.name,
            real_yield, idx_ratio, self.unit.currency)

    def pv_ctx(self, ctx) -> float:
        curve = ctx.discount_curve
        return self.price(ctx.valuation_date, curve,
                          self.unit.typical_value).nominal_price

    def to_dict(self) -> dict:
        return {"type": "inflation_unit_bond", "unit": self.unit.name,
                "maturity": self.maturity.isoformat(),
                "real_coupon": self.real_coupon, "face_units": self.face_units}


# ═══════════════════════════════════════════════════════════════
# Dual-curve breakeven inflation
# ═══════════════════════════════════════════════════════════════

def dual_curve_breakeven(
    nominal_curve: DiscountCurve,
    real_curve: DiscountCurve,
    maturities_years: list[float] | None = None,
    reference_date: date | None = None,
) -> list[dict]:
    """Compute breakeven inflation from nominal vs real discount curves.

    BEI ≈ nominal_rate - real_rate.
    Exact: (1 + nominal) = (1 + real)(1 + BEI).

    Works for any pair of nominal/real curves (CLP/UF, MXN/UDI, etc.).

    Returns:
        List of dicts with keys: years, nominal_rate, real_rate, bei.
    """
    from dateutil.relativedelta import relativedelta

    if maturities_years is None:
        maturities_years = [1, 2, 3, 5, 7, 10, 15, 20, 30]

    ref = reference_date or nominal_curve.reference_date
    results = []

    for T in maturities_years:
        mat = ref + relativedelta(years=int(T))
        df_nom = nominal_curve.df(mat)
        df_real = real_curve.df(mat)

        if df_nom > 0 and df_real > 0 and T > 0:
            nom_rate = -math.log(df_nom) / T
            real_rate = -math.log(df_real) / T
            bei = nom_rate - real_rate
        else:
            nom_rate = real_rate = bei = 0.0

        results.append({
            "years": T, "nominal_rate": nom_rate,
            "real_rate": real_rate, "bei": bei,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# Cross-country comparison
# ═══════════════════════════════════════════════════════════════

def compare_units() -> list[dict]:
    """Compare all inflation units side by side.

    Returns a list of dicts with unit properties for easy tabulation.
    """
    return [
        {"name": u.name, "currency": u.currency,
         "publisher": u.publisher.value,
         "day_count": u.day_count.value,
         "frequency": f"{u.coupon_frequency_months}M",
         "base_year": u.base_year,
         "typical_value": u.typical_value,
         }
        for u in sorted(_UNITS.values(), key=lambda u: u.name)
    ]
