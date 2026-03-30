"""
Discrete dividend handling for equity option pricing.

Spot-based model: adjust spot for PV of known dividends, then price
with Black-Scholes on the adjusted forward.

    adjusted = dividend_adjusted_forward(
        spot=100, dividends=[Dividend(date(2024,6,15), 2.0)],
        curve=ois_curve, maturity=date(2025,1,15),
    )

Piecewise forward: construct F(t) with jumps at each ex-date.
"""

from __future__ import annotations

import math
from datetime import date
from dataclasses import dataclass

from pricebook.discount_curve import DiscountCurve
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.black76 import OptionType, black76_price


@dataclass
class Dividend:
    """A single discrete dividend payment."""

    ex_date: date
    amount: float


def pv_dividends(
    dividends: list[Dividend],
    curve: DiscountCurve,
    maturity: date,
) -> float:
    """Present value of discrete dividends before maturity."""
    return sum(
        d.amount * curve.df(d.ex_date)
        for d in dividends
        if d.ex_date <= maturity
    )


def dividend_adjusted_forward(
    spot: float,
    dividends: list[Dividend],
    curve: DiscountCurve,
    maturity: date,
) -> float:
    """Forward price adjusted for discrete dividends.

    F = (S - PV(divs)) / df(T)
    """
    pv_divs = pv_dividends(dividends, curve, maturity)
    return (spot - pv_divs) / curve.df(maturity)


def piecewise_forward(
    spot: float,
    dividends: list[Dividend],
    curve: DiscountCurve,
    dates: list[date],
) -> list[float]:
    """Piecewise forward curve with dividend jumps.

    Returns a forward price for each date in `dates`. The forward drops
    by the dividend amount at each ex-date.

    Args:
        spot: current spot price.
        dividends: list of discrete dividends.
        curve: discount curve.
        dates: dates at which to compute the forward.

    Returns:
        List of forward prices, one per date.
    """
    result = []
    for d in dates:
        pv_divs = pv_dividends(dividends, curve, d)
        df_d = curve.df(d)
        result.append((spot - pv_divs) / df_d)
    return result


def equity_option_discrete_divs(
    spot: float,
    strike: float,
    dividends: list[Dividend],
    curve: DiscountCurve,
    vol: float,
    maturity: date,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """European option price with discrete dividends.

    Uses the spot-adjustment model: replace S with S - PV(divs),
    then apply Black-76 on the adjusted forward.

    Args:
        spot: current spot price.
        strike: option strike.
        dividends: list of discrete dividends.
        curve: discount curve.
        vol: lognormal volatility.
        maturity: option expiry date.
        option_type: CALL or PUT.
    """
    fwd = dividend_adjusted_forward(spot, dividends, curve, maturity)
    df = curve.df(maturity)
    T = year_fraction(
        curve.reference_date, maturity, DayCountConvention.ACT_365_FIXED,
    )
    return black76_price(fwd, strike, vol, T, df, option_type)
