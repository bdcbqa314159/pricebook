"""Discount curve built from a set of discount factors at known dates."""

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.interpolation import (
    InterpolationMethod,
    create_interpolator,
    Interpolator,
)


class DiscountCurve:
    """
    A discount curve maps dates to discount factors.

    Internally the curve works in year-fraction space. Discount factors
    are interpolated using the chosen method (default: log-linear, which
    gives piecewise constant forward rates).

    Provides:
        - df(date) -> discount factor
        - zero_rate(date) -> continuously compounded zero rate
        - forward_rate(date1, date2) -> simply compounded forward rate
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        dfs: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    ):
        if len(dates) != len(dfs):
            raise ValueError("dates and dfs must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")

        self.reference_date = reference_date
        self.day_count = day_count

        # Convert dates to year fractions from reference date
        times = [year_fraction(reference_date, d, day_count) for d in dates]

        # Prepend t=0, df=1 if not already present
        if times[0] > 0:
            times = [0.0] + times
            dfs = [1.0] + list(dfs)

        self._times = np.array(times)
        self._dfs = np.array(dfs)
        self._interpolator: Interpolator = create_interpolator(
            interpolation, self._times, self._dfs
        )

    def _time(self, d: date) -> float:
        """Year fraction from reference date to d. Negative if d < reference."""
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def df(self, d: date) -> float:
        """Discount factor at date d. Returns 1.0 for d <= reference_date."""
        t = self._time(d)
        if t <= 0:
            return 1.0
        return self._interpolator(t)

    def zero_rate(self, d: date) -> float:
        """Continuously compounded zero rate to date d: r = -ln(df) / t."""
        t = self._time(d)
        if t <= 0:
            return 0.0
        return -math.log(self.df(d)) / t

    def forward_rate(self, d1: date, d2: date) -> float:
        """
        Simply compounded forward rate between d1 and d2.

        F(t1, t2) = (df(t1) / df(t2) - 1) / tau

        where tau is the year fraction between d1 and d2.
        """
        if d1 >= d2:
            raise ValueError(f"d1 ({d1}) must be before d2 ({d2})")
        df1 = self.df(d1)
        df2 = self.df(d2)
        tau = year_fraction(d1, d2, self.day_count)
        return (df1 / df2 - 1.0) / tau
