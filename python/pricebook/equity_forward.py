"""
Equity forward pricing.

Continuous dividend yield:
    F = S * exp((r - q + b) * T)

Discrete dividends:
    F = (S - PV(divs)) / df(T)

where PV(divs) = sum of div_amount * df(ex_date) for each dividend
before maturity, and b = borrow cost for short sellers.
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
        valuation_date: date from which to measure time (today).
        div_yield: continuous dividend yield (default 0).
        borrow_cost: annualised stock borrow cost (default 0).
        dividends: list of discrete dividends (default empty).
    """

    def __init__(
        self,
        spot: float,
        maturity: date,
        valuation_date: date,
        div_yield: float = 0.0,
        borrow_cost: float = 0.0,
        dividends: list[Dividend] | None = None,
    ):
        if spot <= 0:
            raise ValueError(f"spot must be positive, got {spot}")
        if maturity <= valuation_date:
            raise ValueError(f"maturity ({maturity}) must be after valuation_date ({valuation_date})")
        self.spot = spot
        self.maturity = maturity
        self.valuation_date = valuation_date
        self.div_yield = div_yield
        self.borrow_cost = borrow_cost
        self.dividends = dividends or []

    def _T(self) -> float:
        return year_fraction(
            self.valuation_date, self.maturity, DayCountConvention.ACT_365_FIXED,
        )

    def forward_price(self, curve: DiscountCurve) -> float:
        """Forward price using the discount curve.

        With discrete dividends: F = (S - PV(divs)) × exp(b×T) / df(T)
        Without: F = S × exp((r - q + b) × T), where r = -ln(df)/T.
        """
        T = self._T()
        df_T = curve.df(self.maturity)

        if self.dividends:
            pv_divs = pv_dividends(self.dividends, curve, self.maturity)
            fwd = (self.spot - pv_divs) / df_T
            if self.borrow_cost != 0:
                fwd *= math.exp(self.borrow_cost * T)
            return fwd
        else:
            r = -math.log(df_T) / T if T > 0 and df_T > 0 else 0.0
            return self.spot * math.exp((r - self.div_yield + self.borrow_cost) * T)

    def pv(self, strike: float, curve: DiscountCurve) -> float:
        """PV of a forward contract: df(T) * (F - K)."""
        fwd = self.forward_price(curve)
        return curve.df(self.maturity) * (fwd - strike)

    def delta(self, curve: DiscountCurve) -> float:
        """dF/dS: forward sensitivity to spot.

        With discrete dividends: delta = exp(b×T) / df(T)
        Without: delta = exp((r - q + b) × T)
        """
        T = self._T()
        df_T = curve.df(self.maturity)

        if self.dividends:
            d = 1.0 / df_T
            if self.borrow_cost != 0:
                d *= math.exp(self.borrow_cost * T)
            return d
        else:
            r = -math.log(df_T) / T if T > 0 and df_T > 0 else 0.0
            return math.exp((r - self.div_yield + self.borrow_cost) * T)

    def forward_dv01(self, curve: DiscountCurve, shift: float = 0.0001) -> float:
        """Forward sensitivity to a 1bp parallel rate shift."""
        fwd_base = self.forward_price(curve)
        fwd_bumped = self.forward_price(curve.bumped(shift))
        return fwd_bumped - fwd_base
