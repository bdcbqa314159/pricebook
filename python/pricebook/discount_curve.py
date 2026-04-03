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

    @property
    def pillar_times(self) -> np.ndarray:
        """Year fractions of all pillars (including t=0)."""
        return self._times

    @property
    def pillar_dfs(self) -> np.ndarray:
        """Discount factors at all pillars (including df=1 at t=0)."""
        return self._dfs

    @property
    def pillar_dates(self) -> list[date]:
        """Pillar dates (excluding the t=0 point)."""
        return [
            date.fromordinal(self.reference_date.toordinal() + int(t * 365))
            for t in self._times if t > 0
        ]

    def bumped(self, shift: float) -> "DiscountCurve":
        """New curve with all zero rates shifted by `shift` (e.g. 0.0001 = +1bp)."""
        new_dfs = []
        for t, df_val in zip(self._times, self._dfs):
            if t > 0:
                new_dfs.append(float(df_val * math.exp(-shift * t)))
        return DiscountCurve(
            self.reference_date, self.pillar_dates, new_dfs,
            self.day_count,
        )

    def bumped_at(self, pillar_idx: int, shift: float) -> "DiscountCurve":
        """New curve with one pillar's zero rate shifted by `shift`."""
        pillar_t = [t for t in self._times if t > 0]
        pillar_df = [float(df) for t, df in zip(self._times, self._dfs) if t > 0]
        pillar_df[pillar_idx] = pillar_df[pillar_idx] * math.exp(-shift * pillar_t[pillar_idx])
        return DiscountCurve(
            self.reference_date, self.pillar_dates, pillar_df,
            self.day_count,
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

    def instantaneous_forward(self, t: float) -> float:
        """Instantaneous forward rate: f(t) = -d/dT ln P(T) via finite difference."""
        dt = 1.0 / 365.0
        ref = self.reference_date
        d1 = date.fromordinal(ref.toordinal() + int(t * 365))
        d2 = date.fromordinal(ref.toordinal() + int(t * 365) + 1)
        df1 = self.df(d1)
        df2 = self.df(d2)
        if df1 <= 0 or df2 <= 0:
            return 0.0
        return -math.log(df2 / df1) / dt

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
