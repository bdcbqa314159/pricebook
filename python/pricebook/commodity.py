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
from datetime import date, timedelta

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
        spot: spot price. If None, extrapolates from nearest forward.
        day_count: for year fraction computation.
        interpolation: interpolation method (default: linear).
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        forwards: list[float],
        spot: float | None = None,
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        interpolation: InterpolationMethod = InterpolationMethod.LINEAR,
    ):
        if len(dates) != len(forwards):
            raise ValueError("dates and forwards must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 forward point")

        self.reference_date = reference_date
        self.day_count = day_count
        self._dates = list(dates)
        self._forwards = list(forwards)
        self._spot = spot if spot is not None else forwards[0]

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

    def spot_price(self) -> float:
        """Spot price (explicit or extrapolated from nearest forward)."""
        return self._spot

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
        S = self._spot
        F = self.forward(d)
        if S <= 0 or F <= 0:
            return 0.0
        r = discount_curve.zero_rate(d)
        return r - math.log(F / S) / T

    def bumped(self, shift: float) -> "CommodityForwardCurve":
        """Return a new curve with all forwards shifted by `shift` (additive)."""
        new_fwds = [f + shift for f in self._forwards]
        new_spot = self._spot + shift if self._spot is not None else None
        return CommodityForwardCurve(
            self.reference_date, self._dates, new_fwds,
            spot=new_spot, day_count=self.day_count,
        )

    def bumped_pct(self, pct: float) -> "CommodityForwardCurve":
        """Return a new curve with all forwards shifted by `pct` (multiplicative)."""
        new_fwds = [f * (1 + pct) for f in self._forwards]
        new_spot = self._spot * (1 + pct) if self._spot is not None else None
        return CommodityForwardCurve(
            self.reference_date, self._dates, new_fwds,
            spot=new_spot, day_count=self.day_count,
        )

    def roll_down(self, days: int) -> "CommodityForwardCurve":
        """Rolled curve: what the curve looks like in N days if shape unchanged.

        Each forward moves closer by N days on the time axis.
        """
        new_ref = self.reference_date + timedelta(days=days)
        # Keep same dates but new reference → shorter tenors
        return CommodityForwardCurve(
            new_ref, self._dates, self._forwards,
            spot=self._spot, day_count=self.day_count,
        )


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
        self.frequency = frequency
        self.schedule = generate_schedule(start, end, frequency)

    def _future_periods(self, settlement: date) -> list[tuple[date, date]]:
        """Return (period_start, period_end) pairs where period_end > settlement."""
        return [
            (self.schedule[i - 1], self.schedule[i])
            for i in range(1, len(self.schedule))
            if self.schedule[i] > settlement
        ]

    def _period_average_forward(
        self,
        period_start: date,
        period_end: date,
        forward_curve: CommodityForwardCurve,
        n_samples: int = 5,
    ) -> float:
        """Average forward over a period (approximated by sampling)."""
        if n_samples <= 1:
            return forward_curve.forward(period_end)
        total_days = (period_end - period_start).days
        if total_days <= 0:
            return forward_curve.forward(period_end)
        total = 0.0
        for k in range(n_samples):
            d = period_start + timedelta(days=int(k * total_days / (n_samples - 1)))
            total += forward_curve.forward(d)
        return total / n_samples

    def pv(
        self,
        forward_curve: CommodityForwardCurve,
        discount_curve: DiscountCurve,
        settlement: date | None = None,
        use_average: bool = False,
    ) -> float:
        """PV of the swap (receiver floating). Only future periods.

        Args:
            forward_curve: commodity forward curve.
            discount_curve: risk-free discount curve.
            settlement: only include periods after this date. Defaults to curve ref date.
            use_average: if True, use period-average forward instead of point forward.
        """
        settle = settlement if settlement is not None else discount_curve.reference_date
        total = 0.0
        for p_start, p_end in self._future_periods(settle):
            if use_average:
                fwd = self._period_average_forward(p_start, p_end, forward_curve)
            else:
                fwd = forward_curve.forward(p_end)
            df = discount_curve.df(p_end)
            total += df * (fwd - self.fixed_price) * self.quantity
        return total

    def par_price(
        self,
        forward_curve: CommodityForwardCurve,
        discount_curve: DiscountCurve,
        settlement: date | None = None,
        use_average: bool = False,
    ) -> float:
        """Fixed price that makes PV = 0. Only future periods.

        par = sum(df * fwd * qty) / sum(df * qty)
        """
        settle = settlement if settlement is not None else discount_curve.reference_date
        num = 0.0
        den = 0.0
        for p_start, p_end in self._future_periods(settle):
            if use_average:
                fwd = self._period_average_forward(p_start, p_end, forward_curve)
            else:
                fwd = forward_curve.forward(p_end)
            df = discount_curve.df(p_end)
            num += df * fwd * self.quantity
            den += df * self.quantity
        return num / den if den != 0 else 0.0

    def dv01(
        self,
        forward_curve: CommodityForwardCurve,
        discount_curve: DiscountCurve,
        shift: float = 1.0,
    ) -> float:
        """DV01: PV change for a $1 parallel shift in the forward curve."""
        pv_base = self.pv(forward_curve, discount_curve)
        pv_bumped = self.pv(forward_curve.bumped(shift), discount_curve)
        return pv_bumped - pv_base


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
