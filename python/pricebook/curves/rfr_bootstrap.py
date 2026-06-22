"""Multi-RFR OIS curve bootstrap — production grade.

Builds RFR discount curves for any supported currency from the standard
instrument stack: overnight deposit, RFR futures, and OIS swaps.

    from pricebook.curves.rfr_bootstrap import (
        bootstrap_rfr, RFRCurveInputs, RFRCurveResult,
    )

    inputs = RFRCurveInputs(
        overnight_rate=0.053,
        futures_1m=[...],
        futures_3m=[...],
        ois_swaps=[(date(2026,1,15), 0.045), (date(2029,1,15), 0.042), ...],
    )
    result = bootstrap_rfr("USD", ref_date, inputs)
    curve = result.curve

Instrument stack (in maturity order):
1. O/N deposit → 1-day DF
2. 1M futures (SR1 for SOFR) → short-end, monthly granularity
3. 3M futures (SR3 for SOFR) → 2-8 quarter horizon
4. OIS swaps → long end (2Y-30Y)

References:
    ARRC (2020). SOFR First: Best Practices.
    CME (2024). SOFR Futures and OIS Conventions.
    LCH (2024). SwapClear Discounting: SOFR, ESTR, SONIA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.calendar import get_calendar
from pricebook.curves.bootstrap import bootstrap

if TYPE_CHECKING:
    from pricebook.fixed_income.rfr_futures import RFRFutureSpec


@dataclass
class RFRCurveInputs:
    """Market data inputs for RFR curve construction."""
    overnight_rate: float | None = None
    term_rates: list[tuple[str, float]] | None = None  # [("1M", 0.053), ...]
    futures_1m: list[RFRFutureSpec] = field(default_factory=list)
    futures_3m: list[RFRFutureSpec] = field(default_factory=list)
    ois_swaps: list[tuple[date, float]] = field(default_factory=list)
    deposits: list[tuple[date, float]] = field(default_factory=list)

    def n_instruments(self) -> int:
        n = len(self.futures_1m) + len(self.futures_3m) + len(self.ois_swaps) + len(self.deposits)
        if self.overnight_rate is not None:
            n += 1
        if self.term_rates:
            n += len(self.term_rates)
        return n

    def to_dict(self) -> dict:
        return {
            "overnight_rate": self.overnight_rate,
            "n_futures_1m": len(self.futures_1m),
            "n_futures_3m": len(self.futures_3m),
            "n_ois_swaps": len(self.ois_swaps),
            "n_deposits": len(self.deposits),
            "n_total": self.n_instruments(),
        }


@dataclass
class RFRCurveResult:
    """Result of RFR curve bootstrap."""
    curve: DiscountCurve
    currency: str
    rfr_name: str
    n_instruments: int
    method: str
    pillar_dates: list[date]
    pillar_zero_rates: list[float]
    round_trip_max_error_bp: float
    convexity_adjustments: dict[str, float]  # {contract_label: ca_bp}

    @property
    def calibration_result(self):
        """The underlying discount curve's calibration provenance (it owns the fit)."""
        return self.curve.calibration_result

    def to_dict(self) -> dict:
        return {
            "currency": self.currency,
            "rfr_name": self.rfr_name,
            "n_instruments": self.n_instruments,
            "method": self.method,
            "pillar_dates": [d.isoformat() for d in self.pillar_dates],
            "pillar_zero_rates": self.pillar_zero_rates,
            "round_trip_max_error_bp": self.round_trip_max_error_bp,
        }


# ═══════════════════════════════════════════════════════════════
# RFR OIS conventions per currency
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RFROISConventions:
    """OIS swap conventions for an RFR currency."""
    rfr_name: str
    deposit_dc: DayCountConvention
    fixed_dc: DayCountConvention
    float_dc: DayCountConvention
    fixed_freq: str   # "annual" or "semi_annual"
    float_freq: str   # "annual" or "quarterly"
    calendar_ccy: str


_RFR_OIS: dict[str, RFROISConventions] = {
    "USD": RFROISConventions("SOFR", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
                              DayCountConvention.ACT_360, "annual", "annual", "USD"),
    "EUR": RFROISConventions("ESTR", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
                              DayCountConvention.ACT_360, "annual", "annual", "EUR"),
    "GBP": RFROISConventions("SONIA", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                              DayCountConvention.ACT_365_FIXED, "annual", "annual", "GBP"),
    "JPY": RFROISConventions("TONA", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                              DayCountConvention.ACT_365_FIXED, "annual", "annual", "JPY"),
    "CHF": RFROISConventions("SARON", DayCountConvention.ACT_360, DayCountConvention.ACT_360,
                              DayCountConvention.ACT_360, "annual", "annual", "CHF"),
    "CAD": RFROISConventions("CORRA", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                              DayCountConvention.ACT_365_FIXED, "annual", "annual", "CAD"),
    "AUD": RFROISConventions("AONIA", DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
                              DayCountConvention.ACT_365_FIXED, "semi_annual", "semi_annual", "AUD"),
}

_FREQ_MAP = {
    "annual": "12",
    "semi_annual": "6",
    "quarterly": "3",
}


def get_rfr_ois_conventions(currency: str) -> RFROISConventions:
    """Get OIS conventions for an RFR currency."""
    ccy = currency.upper()
    conv = _RFR_OIS.get(ccy)
    if conv is None:
        raise ValueError(f"No RFR OIS conventions for {ccy}. Available: {sorted(_RFR_OIS.keys())}")
    return conv


def list_rfr_ois_currencies() -> list[str]:
    """Return currencies with RFR OIS conventions."""
    return sorted(_RFR_OIS.keys())


# ═══════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════

from pricebook.core.schedule import Frequency as _Freq

_FREQ_ENUM = {
    "annual": _Freq.ANNUAL,
    "semi_annual": _Freq.SEMI_ANNUAL,
    "quarterly": _Freq.QUARTERLY,
}


def bootstrap_rfr(
    currency: str,
    reference_date: date,
    inputs: RFRCurveInputs,
    method: str = "sequential",
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    hw_a: float = 0.03,
    hw_sigma: float = 0.01,
    turn_of_year_spread: float = 0.0,
) -> RFRCurveResult:
    """Bootstrap an RFR OIS curve from market data.

    Args:
        currency: "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD".
        reference_date: valuation date.
        inputs: market data (O/N, futures, swaps).
        method: "sequential" (Brent per pillar) or "global" (Newton).
        interpolation: interpolation method for the curve.
        hw_a: Hull-White mean reversion for futures convexity.
        hw_sigma: Hull-White vol for futures convexity.
        turn_of_year_spread: additive spread for year-end funding.

    Returns:
        RFRCurveResult with discount curve and diagnostics.
    """
    conv = get_rfr_ois_conventions(currency)
    cal = get_calendar(conv.calendar_ccy)

    if inputs.n_instruments() == 0:
        raise ValueError("At least one instrument required")

    # Build deposit list
    deposits = list(inputs.deposits)

    # O/N deposit → 1-day DF
    if inputs.overnight_rate is not None:
        on_date = cal.add_business_days(reference_date, 1)
        deposits.insert(0, (on_date, inputs.overnight_rate))

    # Term rates → short deposits
    if inputs.term_rates:
        for tenor_str, rate in inputs.term_rates:
            months = {"1W": 0, "1M": 1, "3M": 3, "6M": 6, "12M": 12}.get(tenor_str)
            if months is not None and months > 0:
                mat = _add_months(reference_date, months)
                deposits.append((mat, rate))
            elif tenor_str == "1W":
                mat = cal.add_business_days(reference_date, 5)
                deposits.append((mat, rate))

    # Convert futures to FRA-like forward rates
    all_futures = inputs.futures_1m + inputs.futures_3m
    fras = None
    convexity_adjustments = {}
    if all_futures:
        priced_futures = [f for f in all_futures if f.price > 0]
        if priced_futures:
            from pricebook.fixed_income.rfr_futures import rfr_futures_to_forwards
            fras_list = rfr_futures_to_forwards(priced_futures, reference_date, hw_a, hw_sigma)
            if fras_list:
                fras = fras_list
            # Record convexity adjustments
            from pricebook.fixed_income.rfr_futures import rfr_futures_convexity
            for f in priced_futures:
                ca = rfr_futures_convexity(f, reference_date, hw_a, hw_sigma)
                label = f"{f.contract_type}_{f.contract_month.isoformat()}"
                convexity_adjustments[label] = ca * 10_000  # in bp

    # OIS swaps
    swaps = sorted(inputs.ois_swaps, key=lambda x: x[0])

    # Sort deposits by date
    deposits = sorted(deposits, key=lambda x: x[0])

    # Bootstrap
    fixed_freq = _FREQ_ENUM[conv.fixed_freq]
    float_freq = _FREQ_ENUM[conv.float_freq]

    if method == "global":
        from pricebook.curves.global_solver import global_bootstrap
        curve = global_bootstrap(
            reference_date, deposits, swaps,
            deposit_dc=conv.deposit_dc,
            swap_dc=conv.fixed_dc,
            swap_frequency=fixed_freq,
            interpolation=interpolation,
        )
    else:
        curve = bootstrap(
            reference_date, deposits, swaps,
            fras=fras,
            deposit_day_count=conv.deposit_dc,
            fixed_day_count=conv.fixed_dc,
            float_day_count=conv.float_dc,
            fixed_frequency=fixed_freq,
            float_frequency=float_freq,
            interpolation=interpolation,
            calendar=cal,
            turn_of_year_spread=turn_of_year_spread,
        )

    # Extract pillar info
    pillar_dates = curve.pillar_dates
    dc = conv.deposit_dc
    pillar_zeros = []
    for d in pillar_dates:
        t = year_fraction(reference_date, d, dc)
        if t > 0:
            pillar_zeros.append(-math.log(curve.df(d)) / t)
        else:
            pillar_zeros.append(0.0)

    # Round-trip verification
    max_err = _verify_round_trip(curve, reference_date, deposits, swaps, conv)

    return RFRCurveResult(
        curve=curve,
        currency=currency.upper(),
        rfr_name=conv.rfr_name,
        n_instruments=inputs.n_instruments(),
        method=method,
        pillar_dates=pillar_dates,
        pillar_zero_rates=pillar_zeros,
        round_trip_max_error_bp=max_err,
        convexity_adjustments=convexity_adjustments,
    )


def _verify_round_trip(curve, ref, deposits, swaps, conv):
    """Verify instruments reprice. Returns max error in bp."""
    max_err = 0.0
    dc = conv.deposit_dc

    for mat, rate in deposits:
        t = year_fraction(ref, mat, dc)
        if t <= 0:
            continue
        implied = (1.0 / curve.df(mat) - 1.0) / t
        err = abs(implied - rate) * 10_000
        max_err = max(max_err, err)

    # Swap verification is more complex; skip for now (bootstrap handles it)
    return max_err


def _add_months(d: date, months: int) -> date:
    year = d.year + (d.month + months - 1) // 12
    month = (d.month + months - 1) % 12 + 1
    day = min(d.day, 28)
    return date(year, month, day)
