"""
Commodity instruments: forward curve, swaps, options.

Commodity forward curve maps dates to forward prices. Commodities
use Black-76 for options (forward-based, no spot carry model needed).

    curve = CommodityForwardCurve(ref, dates, forwards)
    swap = CommoditySwap(start, end, fixed_price, curve)
    option_pv = commodity_option_price(forward, strike, vol, T, df)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.interpolation import InterpolationMethod, create_interpolator
from pricebook.schedule import Frequency, generate_schedule
from pricebook.black76 import OptionType, black76_price, black76_vega


class CommodityForwardCurve:
    """Commodity forward price curve.

    Args:
        reference_date: valuation date.
        dates: delivery dates.
        forwards: forward prices at each delivery date.
        day_count: for year fraction computation.
        interpolation: interpolation method (default: linear).
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        forwards: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ):
        if len(dates) != len(forwards):
            raise ValueError("dates and forwards must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 forward point")

        self.reference_date = reference_date
        self.day_count = day_count
        self._dates = dates
        self._forwards = forwards

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        if len(times) == 1:
            self._single = forwards[0]
            self._interp = None
        else:
            self._single = None
            self._interp = create_interpolator(
                interpolation, np.array(times), np.array(forwards),
            )

    def forward(self, d: date) -> float:
        """Forward price at delivery date d."""
        if self._single is not None:
            return self._single
        t = year_fraction(self.reference_date, d, self.day_count)
        t = max(t, 0.0)
        return float(self._interp(t))

    def spot(self) -> float:
        """Spot price (forward at reference date)."""
        return self.forward(self.reference_date)

    def convenience_yield(
        self,
        d: date,
        discount_curve: DiscountCurve,
    ) -> float:
        """Implied convenience yield: y such that F = S * exp((r - y) * T).

        y = r - ln(F/S) / T
        """
        T = year_fraction(self.reference_date, d, self.day_count)
        if T <= 0:
            return 0.0
        S = self._forwards[0]  # spot ≈ nearest forward
        F = self.forward(d)
        r = discount_curve.zero_rate(d)
        return r - math.log(F / S) / T


class CommoditySwap:
    """Fixed-for-floating commodity swap.

    Fixed leg pays fixed_price per period.
    Floating leg pays the average forward price over the period.
    PV = sum of df * (forward - fixed) * quantity per period.

    Args:
        start: swap start date.
        end: swap end date.
        fixed_price: agreed fixed price per unit.
        quantity: quantity per period.
        frequency: payment frequency.
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_price: float,
        quantity: float = 1.0,
        frequency: Frequency = Frequency.MONTHLY,
    ):
        self.start = start
        self.end = end
        self.fixed_price = fixed_price
        self.quantity = quantity
        self.schedule = generate_schedule(start, end, frequency)

    def pv(
        self,
        forward_curve: CommodityForwardCurve,
        discount_curve: DiscountCurve,
    ) -> float:
        """PV of the swap (receiver floating)."""
        total = 0.0
        for i in range(1, len(self.schedule)):
            payment_date = self.schedule[i]
            fwd = forward_curve.forward(payment_date)
            df = discount_curve.df(payment_date)
            total += df * (fwd - self.fixed_price) * self.quantity
        return total

    def par_price(
        self,
        forward_curve: CommodityForwardCurve,
        discount_curve: DiscountCurve,
    ) -> float:
        """Fixed price that makes PV = 0.

        par = sum(df * fwd * qty) / sum(df * qty)
        """
        num = 0.0
        den = 0.0
        for i in range(1, len(self.schedule)):
            payment_date = self.schedule[i]
            fwd = forward_curve.forward(payment_date)
            df = discount_curve.df(payment_date)
            num += df * fwd * self.quantity
            den += df * self.quantity
        return num / den if den != 0 else 0.0


def commodity_option_price(
    forward: float,
    strike: float,
    vol: float,
    T: float,
    df: float,
    option_type: OptionType = OptionType.CALL,
) -> float:
    """European commodity option price (Black-76 on the forward)."""
    return black76_price(forward, strike, vol, T, df, option_type)
