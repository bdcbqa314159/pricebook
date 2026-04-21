"""Unified curve building entry point.

Single function to build OIS + projection curves from raw market quotes.

    from pricebook.curve_builder import build_curves

    curves = build_curves("USD", date.today(), quotes)
    ois = curves["ois"]
    projection = curves["projection"]

References:
    Ametrano & Bianchetti, *Everything You Always Wanted to Know About
    Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask*, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.bootstrap import bootstrap
from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.schedule import Frequency


# ---- Currency conventions ----

@dataclass
class CurrencyConventions:
    """Market conventions for a currency."""
    deposit_day_count: DayCountConvention
    fixed_day_count: DayCountConvention
    float_day_count: DayCountConvention
    fixed_frequency: Frequency
    float_frequency: Frequency
    interpolation: InterpolationMethod


_CONVENTIONS = {
    "USD": CurrencyConventions(
        DayCountConvention.ACT_360, DayCountConvention.THIRTY_360,
        DayCountConvention.ACT_360, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
        InterpolationMethod.LOG_LINEAR,
    ),
    "EUR": CurrencyConventions(
        DayCountConvention.ACT_360, DayCountConvention.THIRTY_360,
        DayCountConvention.ACT_360, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
        InterpolationMethod.LOG_LINEAR,
    ),
    "GBP": CurrencyConventions(
        DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
        DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
        InterpolationMethod.LOG_LINEAR,
    ),
    "JPY": CurrencyConventions(
        DayCountConvention.ACT_360, DayCountConvention.ACT_365_FIXED,
        DayCountConvention.ACT_360, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
        InterpolationMethod.LOG_LINEAR,
    ),
}


@dataclass
class CurveSetResult:
    """Result of build_curves: OIS + optional projection curve."""
    ois: DiscountCurve
    projection: DiscountCurve | None
    currency: str
    reference_date: date


def build_curves(
    currency: str,
    reference_date: date,
    ois_deposits: list[tuple[date, float]],
    ois_swaps: list[tuple[date, float]],
    projection_swaps: list[tuple[date, float]] | None = None,
    fras: list[tuple[date, date, float]] | None = None,
    futures: list[tuple[date, date, float]] | None = None,
    hw_convexity_a: float = 0.0,
    hw_convexity_sigma: float = 0.0,
    turn_of_year_spread: float = 0.0,
) -> CurveSetResult:
    """Build OIS discount + optional projection curve from market quotes.

    This is the unified entry point for curve construction. It handles:
    - Currency-specific conventions (day counts, frequencies)
    - OIS bootstrap (deposits + swaps)
    - Projection curve bootstrap (if projection_swaps provided, uses OIS for discounting)
    - FRA and futures integration (with convexity + TOY adjustments)

    Args:
        currency: ISO currency code (USD, EUR, GBP, JPY).
        reference_date: Valuation date.
        ois_deposits: OIS deposit quotes [(maturity, rate), ...].
        ois_swaps: OIS swap par rates [(maturity, par_rate), ...].
        projection_swaps: Optional LIBOR/SOFR projection swap quotes.
            If provided, builds a second curve using OIS for discounting.
        fras: Optional FRA quotes [(start, end, rate), ...].
        futures: Optional futures quotes [(start, end, futures_rate), ...].
        hw_convexity_a: Hull-White mean reversion for futures convexity.
        hw_convexity_sigma: Hull-White vol for futures convexity.
        turn_of_year_spread: Additive spread for year-end crossing periods.

    Returns:
        CurveSetResult with OIS curve and optional projection curve.
    """
    conv = _CONVENTIONS.get(currency.upper())
    if conv is None:
        # Default to USD conventions
        conv = _CONVENTIONS["USD"]

    # 1. Build OIS discount curve
    ois_curve = bootstrap(
        reference_date=reference_date,
        deposits=ois_deposits,
        swaps=ois_swaps,
        deposit_day_count=conv.deposit_day_count,
        fixed_day_count=conv.fixed_day_count,
        float_day_count=conv.float_day_count,
        fixed_frequency=conv.fixed_frequency,
        float_frequency=conv.float_frequency,
        interpolation=conv.interpolation,
        turn_of_year_spread=turn_of_year_spread,
    )

    # 2. Build projection curve (if quotes provided)
    projection_curve = None
    if projection_swaps:
        from pricebook.bootstrap import bootstrap_forward_curve
        projection_curve = bootstrap_forward_curve(
            reference_date=reference_date,
            swaps=projection_swaps,
            discount_curve=ois_curve,
            fras=fras,
            futures=futures,
            float_day_count=conv.float_day_count,
            fixed_day_count=conv.fixed_day_count,
            fixed_frequency=conv.fixed_frequency,
            float_frequency=conv.float_frequency,
            interpolation=conv.interpolation,
            hw_convexity_a=hw_convexity_a,
            hw_convexity_sigma=hw_convexity_sigma,
            turn_of_year_spread=turn_of_year_spread,
        )

    return CurveSetResult(
        ois=ois_curve,
        projection=projection_curve,
        currency=currency.upper(),
        reference_date=reference_date,
    )
