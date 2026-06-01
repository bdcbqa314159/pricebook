"""Unified curve building entry point.

Single function to build OIS + projection curves from raw market quotes.
Supports multiple construction methods:

    from pricebook.curves.curve_builder import build_curves

    # Sequential bootstrap (default, fastest)
    curves = build_curves("USD", ref, deposits, swaps)

    # Global Newton (simultaneous, more stable)
    curves = build_curves("USD", ref, deposits, swaps, method="global_newton")

    # Nelson-Siegel parametric fit
    curves = build_curves("USD", ref, deposits, swaps, method="nelson_siegel")

    # Smith-Wilson (Solvency II extrapolation)
    curves = build_curves("USD", ref, deposits, swaps, method="smith_wilson")

References:
    Ametrano & Bianchetti, *Everything You Always Wanted to Know About
    Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask*, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.core.serialisable import serialisable_convention
from pricebook.curves.bootstrap import bootstrap
from pricebook.core.day_count import DayCountConvention
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.interpolation import InterpolationMethod
from pricebook.core.schedule import Frequency


# ---- Currency conventions ----

@serialisable_convention("currency_conventions")
@dataclass(frozen=True)
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
    "CHF": CurrencyConventions(
        DayCountConvention.ACT_360, DayCountConvention.THIRTY_360,
        DayCountConvention.ACT_360, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
        InterpolationMethod.LOG_LINEAR,
    ),
    "CAD": CurrencyConventions(
        DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
        DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.SEMI_ANNUAL,
        InterpolationMethod.LOG_LINEAR,
    ),
    "AUD": CurrencyConventions(
        DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
        DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
        InterpolationMethod.LOG_LINEAR,
    ),
    "NZD": CurrencyConventions(
        DayCountConvention.ACT_365_FIXED, DayCountConvention.ACT_365_FIXED,
        DayCountConvention.ACT_365_FIXED, Frequency.SEMI_ANNUAL, Frequency.QUARTERLY,
        InterpolationMethod.LOG_LINEAR,
    ),
    "SEK": CurrencyConventions(
        DayCountConvention.ACT_360, DayCountConvention.THIRTY_360,
        DayCountConvention.ACT_360, Frequency.ANNUAL, Frequency.QUARTERLY,
        InterpolationMethod.LOG_LINEAR,
    ),
    "NOK": CurrencyConventions(
        DayCountConvention.ACT_360, DayCountConvention.THIRTY_360,
        DayCountConvention.ACT_360, Frequency.ANNUAL, Frequency.SEMI_ANNUAL,
        InterpolationMethod.LOG_LINEAR,
    ),
}


def get_conventions(currency: str) -> CurrencyConventions:
    """Look up swap/curve conventions by currency.

    First checks G10 registry, then falls through to EM conventions.
    This gives all 33+ currencies access to all 5 curve methods.
    """
    key = currency.upper()
    if key in _CONVENTIONS:
        return _CONVENTIONS[key]

    # Fall through to EM conventions
    try:
        from pricebook.curves.em_curve_builder import get_em_curve_conventions
        em = get_em_curve_conventions(key)
        return CurrencyConventions(
            em.deposit_day_count, em.fixed_day_count, em.float_day_count,
            em.fixed_frequency, em.float_frequency, em.interpolation,
        )
    except (ValueError, ImportError):
        pass

    raise ValueError(f"Unknown currency: {key}. "
                     f"G10: {sorted(_CONVENTIONS)}. "
                     f"Use build_em_curve() for EM or add conventions.")


@dataclass
class CurveSetResult:
    """Result of build_curves: OIS + optional projection curve."""
    ois: DiscountCurve
    projection: DiscountCurve | None
    currency: str
    reference_date: date



    def to_dict(self) -> dict:
        return vars(self)
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
    method: str = "sequential",
) -> CurveSetResult:
    """Build OIS discount + optional projection curve from market quotes.

    Args:
        currency: ISO currency code (USD, EUR, GBP, JPY).
        reference_date: Valuation date.
        ois_deposits: OIS deposit quotes [(maturity, rate), ...].
        ois_swaps: OIS swap par rates [(maturity, par_rate), ...].
        projection_swaps: Optional projection swap quotes.
        fras: Optional FRA quotes [(start, end, rate), ...].
        futures: Optional futures quotes [(start, end, futures_rate), ...].
        hw_convexity_a: Hull-White mean reversion for futures convexity.
        hw_convexity_sigma: Hull-White vol for futures convexity.
        turn_of_year_spread: Additive spread for year-end crossing.
        method: Construction method:
            - ``"sequential"`` (default): sequential bootstrap via Brent.
            - ``"global_newton"``: simultaneous Newton with analytical Jacobian.
            - ``"nelson_siegel"``: parametric NS fit to zero rates.
            - ``"svensson"``: parametric Svensson fit.
            - ``"smith_wilson"``: Smith-Wilson with UFR extrapolation.

    Returns:
        CurveSetResult with OIS curve and optional projection curve.
    """
    conv = get_conventions(currency)  # unified G10 + EM lookup

    # 1. Build OIS discount curve
    if method == "sequential":
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

    elif method == "global_newton":
        from pricebook.curves.global_solver import global_bootstrap
        ois_curve = global_bootstrap(
            reference_date=reference_date,
            deposits=ois_deposits,
            swaps=ois_swaps,
            deposit_dc=conv.deposit_day_count,
            swap_dc=conv.fixed_day_count,
            swap_frequency=conv.fixed_frequency,
            interpolation=conv.interpolation,
        )

    elif method in ("nelson_siegel", "svensson"):
        # First bootstrap to get zero rates, then fit parametric form
        temp_curve = bootstrap(
            reference_date=reference_date,
            deposits=ois_deposits,
            swaps=ois_swaps,
            deposit_day_count=conv.deposit_day_count,
            fixed_day_count=conv.fixed_day_count,
            float_day_count=conv.float_day_count,
            fixed_frequency=conv.fixed_frequency,
            float_frequency=conv.float_frequency,
            interpolation=conv.interpolation,
        )
        from pricebook.core.day_count import year_fraction as _yf
        all_dates = sorted(set([d for d, _ in ois_deposits] + [d for d, _ in ois_swaps]))
        tenors = [_yf(reference_date, d, conv.deposit_day_count) for d in all_dates]
        yields = [temp_curve.zero_rate(d) for d in all_dates]

        from pricebook.curves.nelson_siegel import (
            calibrate_nelson_siegel, calibrate_svensson,
            ns_discount_curve, svensson_discount_curve,
        )
        if method == "nelson_siegel":
            params = calibrate_nelson_siegel(tenors, yields)
            ois_curve = ns_discount_curve(
                reference_date, params["beta0"], params["beta1"],
                params["beta2"], params["tau"],
            )
        else:
            params = calibrate_svensson(tenors, yields)
            ois_curve = svensson_discount_curve(
                reference_date, params["beta0"], params["beta1"],
                params["beta2"], params["tau1"],
                params["beta3"], params["tau2"],
            )

    elif method == "smith_wilson":
        # Bootstrap first, then fit SW for extrapolation
        temp_curve = bootstrap(
            reference_date=reference_date,
            deposits=ois_deposits,
            swaps=ois_swaps,
            deposit_day_count=conv.deposit_day_count,
            fixed_day_count=conv.fixed_day_count,
            float_day_count=conv.float_day_count,
            fixed_frequency=conv.fixed_frequency,
            float_frequency=conv.float_frequency,
            interpolation=conv.interpolation,
        )
        from pricebook.core.day_count import year_fraction as _yf
        all_dates = sorted(set([d for d, _ in ois_deposits] + [d for d, _ in ois_swaps]))
        maturities = [_yf(reference_date, d, conv.deposit_day_count) for d in all_dates]
        market_dfs = [temp_curve.df(d) for d in all_dates]

        from pricebook.curves.smith_wilson import smith_wilson_curve
        ois_curve = smith_wilson_curve(reference_date, maturities, market_dfs)

    else:
        raise ValueError(f"unknown method: {method!r}")

    # 2. Build projection curve (if quotes provided) — always sequential
    projection_curve = None
    if projection_swaps:
        from pricebook.curves.bootstrap import bootstrap_forward_curve
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
