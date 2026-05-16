"""Cross-currency basis: the spread that reconciles market FX forwards with OIS curves."""

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod
from pricebook.solvers import brentq


def implied_basis_spread(
    spot: float,
    maturity: date,
    market_forward: float,
    base_curve: DiscountCurve,
    quote_curve: DiscountCurve,
) -> float:
    """
    Implied cross-currency basis spread.

    CIP says: F_theoretical = S * df_base / df_quote.
    In practice: F_market != F_theoretical due to funding/credit differences.

    The basis spread b is defined so that:
        F_market = S * df_base / df_quote_adjusted

    where df_quote_adjusted uses (quote_zero_rate + b) instead of quote_zero_rate.

    Returns the annualised basis spread in decimal (e.g. -0.002 = -20bp).
    """
    df_base = base_curve.df(maturity)
    f_theoretical = spot * df_base / quote_curve.df(maturity)
    if abs(f_theoretical - market_forward) < 1e-12:
        return 0.0

    t = year_fraction(quote_curve.reference_date, maturity, quote_curve.day_count)
    if t <= 0:
        return 0.0

    # Solve: S * df_base / exp(-(z_quote + b) * t) = F_market
    # => exp(-(z_quote + b) * t) = S * df_base / F_market
    # => z_quote + b = -ln(S * df_base / F_market) / t
    # => b = -ln(S * df_base / F_market) / t - z_quote
    z_quote = quote_curve.zero_rate(maturity)
    implied_z = -math.log(spot * df_base / market_forward) / t
    return implied_z - z_quote


def bootstrap_basis_curve(
    reference_date: date,
    spot: float,
    market_forwards: list[tuple[date, float]],
    base_curve: DiscountCurve,
    quote_curve: DiscountCurve,
    day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
) -> DiscountCurve:
    """
    Bootstrap a basis-adjusted quote currency discount curve from market FX forwards.

    Given market forward rates at various tenors, solve for discount factors in
    the quote currency such that CIP holds exactly:

        F_market(T) = spot * df_base(T) / df_quote_adjusted(T)

    Args:
        reference_date: Curve reference date.
        spot: FX spot rate (quote per base).
        market_forwards: List of (maturity, forward_rate) sorted by maturity.
        base_curve: Discount curve for the base currency.
        quote_curve: Original quote currency OIS curve (used for short-end seed).
        day_count: Day count for the resulting curve.
        interpolation: Interpolation method.

    Returns:
        A basis-adjusted DiscountCurve for the quote currency that reprices
        all market forwards.
    """
    for i in range(1, len(market_forwards)):
        if market_forwards[i][0] <= market_forwards[i - 1][0]:
            raise ValueError("market_forwards must be sorted by maturity")

    pillar_dates: list[date] = []
    pillar_dfs: list[float] = []

    for mat, fwd_market in market_forwards:
        # From CIP: df_quote = spot * df_base / F_market
        df_base = base_curve.df(mat)
        df_quote = spot * df_base / fwd_market
        pillar_dates.append(mat)
        pillar_dfs.append(df_quote)

    return DiscountCurve(
        reference_date, pillar_dates, pillar_dfs,
        day_count=day_count, interpolation=interpolation,
    )
