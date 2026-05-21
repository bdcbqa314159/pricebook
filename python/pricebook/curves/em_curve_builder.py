"""EM local currency curve builders.

Pre-configured curve builders for major EM currencies with correct
market conventions (day count, frequency, calendar, interpolation).

    from pricebook.curves.em_curve_builder import (
        build_cdi_curve, build_tiie_curve, build_shibor_curve,
        build_em_curve, list_em_curve_currencies,
    )

    # Brazil: CDI curve from DI futures
    brl_curve = build_cdi_curve(ref, di_futures)

    # Mexico: TIIE curve from TIIE swaps
    mxn_curve = build_tiie_curve(ref, deposits, swaps)

    # Generic: any supported EM currency
    curve = build_em_curve("ZAR", ref, deposits, swaps)

References:
    B3 (2024). DI Futures Contract Specifications.
    Banxico (2024). TIIE Swap Market Conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.schedule import Frequency
from pricebook.core.calendar import get_calendar
from pricebook.curves.bootstrap import bootstrap


@dataclass(frozen=True)
class EMCurveConventions:
    """Swap/curve conventions for an EM currency."""
    currency: str
    deposit_day_count: DayCountConvention
    fixed_day_count: DayCountConvention
    float_day_count: DayCountConvention
    fixed_frequency: Frequency
    float_frequency: Frequency
    interpolation: InterpolationMethod
    calendar_currency: str

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "deposit_day_count": self.deposit_day_count.value,
            "fixed_day_count": self.fixed_day_count.value,
            "float_day_count": self.float_day_count.value,
            "fixed_frequency": self.fixed_frequency.value,
            "float_frequency": self.float_frequency.value,
        }


# ═══════════════════════════════════════════════════════════════
# Convention registry
# ═══════════════════════════════════════════════════════════════

_CONVENTIONS: dict[str, EMCurveConventions] = {}


def _reg(c: EMCurveConventions) -> None:
    _CONVENTIONS[c.currency] = c


# Brazil — CDI curve (DI futures)
_reg(EMCurveConventions(
    "BRL", DayCountConvention.BUS_252, DayCountConvention.BUS_252,
    DayCountConvention.BUS_252, Frequency.ANNUAL, Frequency.ANNUAL,
    InterpolationMethod.LOG_LINEAR, "BRL"))

# Mexico — TIIE 28d swaps
_reg(EMCurveConventions(
    "MXN", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
    DayCountConvention.ACT_360, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "MXN"))

# China — SHIBOR/FR007 swaps
_reg(EMCurveConventions(
    "CNY", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "CNY"))

# South Korea — KOFR swaps
_reg(EMCurveConventions(
    "KRW", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "KRW"))

# South Africa — JIBAR swaps
_reg(EMCurveConventions(
    "ZAR", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "ZAR"))

# India — MIBOR OIS
_reg(EMCurveConventions(
    "INR", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "INR"))

# Singapore — SORA swaps
_reg(EMCurveConventions(
    "SGD", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "SGD"))

# Hong Kong — HONIA swaps
_reg(EMCurveConventions(
    "HKD", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "HKD"))

# Thailand — THOR swaps
_reg(EMCurveConventions(
    "THB", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "THB"))

# Poland — WIBOR/WIRON swaps
_reg(EMCurveConventions(
    "PLN", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "PLN"))

# Czech — PRIBOR swaps
_reg(EMCurveConventions(
    "CZK", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
    DayCountConvention.ACT_360, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "CZK"))

# Hungary — BUBOR swaps
_reg(EMCurveConventions(
    "HUF", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
    DayCountConvention.ACT_360, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "HUF"))

# Colombia — IBR swaps
_reg(EMCurveConventions(
    "COP", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
    DayCountConvention.ACT_360, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "COP"))

# Chile — TPM/CLF swaps
_reg(EMCurveConventions(
    "CLP", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
    DayCountConvention.ACT_360, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
    InterpolationMethod.LOG_LINEAR, "CLP"))

# Turkey — TLREF swaps
_reg(EMCurveConventions(
    "TRY", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.ANNUAL, Frequency.ANNUAL,
    InterpolationMethod.LOG_LINEAR, "TRY"))

# Indonesia — JIBOR swaps
_reg(EMCurveConventions(
    "IDR", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
    DayCountConvention.ACT_365_FIXED, Frequency.QUARTERLY, Frequency.QUARTERLY,
    InterpolationMethod.LOG_LINEAR, "IDR"))


# ═══════════════════════════════════════════════════════════════
# Registry API
# ═══════════════════════════════════════════════════════════════


def get_em_curve_conventions(currency: str) -> EMCurveConventions:
    """Get curve-building conventions for an EM currency."""
    ccy = currency.upper()
    conv = _CONVENTIONS.get(ccy)
    if conv is None:
        available = sorted(_CONVENTIONS.keys())
        raise ValueError(f"No EM curve conventions for {ccy!r}. Available: {available}")
    return conv


def list_em_curve_currencies() -> list[str]:
    """Return sorted list of EM currencies with curve conventions."""
    return sorted(_CONVENTIONS.keys())


# ═══════════════════════════════════════════════════════════════
# Generic EM curve builder
# ═══════════════════════════════════════════════════════════════


def build_em_curve(
    currency: str,
    reference_date: date,
    deposits: list[tuple[date, float]],
    swaps: list[tuple[date, float]],
    fras: list[tuple[date, date, float]] | None = None,
    futures: list[tuple[date, date, float]] | None = None,
) -> DiscountCurve:
    """Build a discount curve for an EM currency using correct conventions.

    Args:
        currency: 3-letter ISO currency code (e.g. "BRL", "MXN", "CNY").
        reference_date: curve reference date.
        deposits: [(maturity, rate), ...] for short end.
        swaps: [(maturity, par_rate), ...] for long end.
        fras: optional FRAs.
        futures: optional futures.

    Returns:
        DiscountCurve bootstrapped with correct EM conventions.
    """
    conv = get_em_curve_conventions(currency)
    cal = get_calendar(conv.calendar_currency)

    return bootstrap(
        reference_date=reference_date,
        deposits=deposits,
        swaps=swaps,
        fras=fras,
        futures=futures,
        deposit_day_count=conv.deposit_day_count,
        fixed_day_count=conv.fixed_day_count,
        float_day_count=conv.float_day_count,
        fixed_frequency=conv.fixed_frequency,
        float_frequency=conv.float_frequency,
        interpolation=conv.interpolation,
        calendar=cal,
    )


# ═══════════════════════════════════════════════════════════════
# Currency-specific builders
# ═══════════════════════════════════════════════════════════════


def build_cdi_curve(
    reference_date: date,
    di_futures: list[tuple[date, float]],
) -> DiscountCurve:
    """Build Brazilian CDI curve from DI futures.

    DI futures on B3 quote the accumulated CDI rate to maturity.
    The implied discount factor is:

        df(T) = 1 / (1 + DI_rate)^(bus_days / 252)

    We convert DI rates to discount factors directly.

    Args:
        reference_date: curve date.
        di_futures: [(maturity, annual_rate), ...] e.g. [(date(2025,1,2), 0.1175)].
    """
    from pricebook.core.day_count import business_days_between

    cal = get_calendar("BRL")
    pillar_dates = []
    pillar_dfs = []

    for mat, rate in sorted(di_futures, key=lambda x: x[0]):
        bd = business_days_between(reference_date, mat, cal)
        if bd <= 0:
            continue
        df = 1.0 / (1.0 + rate) ** (bd / 252.0)
        pillar_dates.append(mat)
        pillar_dfs.append(df)

    if not pillar_dates:
        raise ValueError("No valid DI futures")

    return DiscountCurve(reference_date, pillar_dates, pillar_dfs)


def build_tiie_curve(
    reference_date: date,
    deposits: list[tuple[date, float]],
    swaps: list[tuple[date, float]],
) -> DiscountCurve:
    """Build Mexican TIIE curve from deposits and TIIE swaps.

    Uses ACT/360, quarterly fixed and float.
    """
    return build_em_curve("MXN", reference_date, deposits, swaps)


def build_shibor_curve(
    reference_date: date,
    deposits: list[tuple[date, float]],
    swaps: list[tuple[date, float]],
) -> DiscountCurve:
    """Build Chinese SHIBOR/FR007 curve from deposits and swaps.

    Uses ACT/365F, quarterly fixed and float.
    """
    return build_em_curve("CNY", reference_date, deposits, swaps)
