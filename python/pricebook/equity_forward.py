"""
Equity forward pricing.

Continuous dividend yield:
    F = S * exp((r - q) * T)

Discrete dividends:
    F = (S - PV(divs)) * exp(r * T)

where PV(divs) = sum of div_amount * df(ex_date) for each dividend
before maturity.
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.dividend_model import Dividend, pv_dividends


class EquityForward:
    """
    Equity forward contract.

    Args:
        spot: current spot price.
        maturity: forward delivery date.
        rate: risk-free rate (continuous compounding) — used when no curve given.
        div_yield: continuous dividend yield (default 0).
        dividends: list of discrete dividends (default empty).
    """

    def __init__(
        self,
        spot: float,
        maturity: date,
        rate: float = 0.0,
        div_yield: float = 0.0,
        dividends: list[Dividend] | None = None,
    ):
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        self.spot = spot
        self.maturity = maturity
        self.rate = rate
        self.div_yield = div_yield
        self.dividends = dividends or []

    def forward_continuous(
        self,
        T: float | None = None,
        valuation_date: date | None = None,
    ) -> float:
        """Forward price using continuous dividend yield.

        F = S * exp((r - q) * T)
        """
        if T is None:
            if valuation_date is None:
                raise ValueError("must provide T or valuation_date")
            T = year_fraction(
                valuation_date, self.maturity, DayCountConvention.ACT_365_FIXED,
            )
        return self.spot * math.exp((self.rate - self.div_yield) * T)

    def forward_discrete(
        self,
        curve: DiscountCurve,
    ) -> float:
        """Forward price with discrete dividends.

        F = (S - PV(divs)) * exp(r * T)

        where PV(divs) uses the discount curve, and r is implied from
        the curve's discount factor to maturity.
        """
        pv_divs = pv_dividends(self.dividends, curve, self.maturity)
        return (self.spot - pv_divs) / curve.df(self.maturity)

    def pv(
        self,
        strike: float,
        curve: DiscountCurve,
        valuation_date: date | None = None,
    ) -> float:
        """PV of a forward contract: df(T) * (F - K).

        Args:
            strike: agreed forward price.
            curve: discount curve.
            valuation_date: unused (curve determines discounting).
        """
        if self.dividends:
            fwd = self.forward_discrete(curve)
        else:
            T = year_fraction(
                curve.reference_date, self.maturity,
                DayCountConvention.ACT_365_FIXED,
            )
            fwd = self.forward_continuous(T=T)

        df_T = curve.df(self.maturity)
        return df_T * (fwd - strike)
