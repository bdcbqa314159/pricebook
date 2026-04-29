"""Discount curve built from a set of discount factors at known dates."""

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction, date_from_year_fraction
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
        if any(df <= 0 for df in dfs):
            raise ValueError("all discount factors must be positive")
        for i in range(len(dates)):
            if dates[i] <= reference_date:
                raise ValueError(f"pillar date {dates[i]} must be after reference date {reference_date}")
            if i > 0 and dates[i] <= dates[i - 1]:
                raise ValueError(f"pillar dates must be strictly increasing: {dates[i]} <= {dates[i-1]}")

        self.reference_date = reference_date
        self.day_count = day_count
        self._interpolation = interpolation

        # Store original dates for exact retrieval (avoids int(t*365) drift)
        self._pillar_dates_original = list(dates)

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
        return self._times.copy()

    @property
    def pillar_dfs(self) -> np.ndarray:
        """Discount factors at all pillars (including df=1 at t=0)."""
        return self._dfs.copy()

    @property
    def pillar_dates(self) -> list[date]:
        """Pillar dates (excluding the t=0 point). Uses stored original dates."""
        return list(self._pillar_dates_original)

    @classmethod
    def flat(
        cls,
        reference_date: date,
        rate: float,
        tenors: list[float] | None = None,
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ) -> "DiscountCurve":
        """Build a flat discount curve at a constant continuously compounded rate."""
        if tenors is None:
            tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20]
        dates = [date_from_year_fraction(reference_date, t) for t in tenors]
        # Use actual year fractions (after date rounding) so the curve is truly flat
        actual_times = [year_fraction(reference_date, d, day_count) for d in dates]
        dfs = [math.exp(-rate * t) for t in actual_times]
        return cls(reference_date, dates, dfs, day_count=day_count)

    def bumped(self, shift: float) -> "DiscountCurve":
        """New curve with all zero rates shifted by `shift` (e.g. 0.0001 = +1bp)."""
        new_dfs = []
        for t, df_val in zip(self._times, self._dfs):
            if t > 0:
                new_dfs.append(float(df_val * math.exp(-shift * t)))
        return DiscountCurve(
            self.reference_date, self.pillar_dates, new_dfs,
            self.day_count, self._interpolation,
        )

    def bumped_at(self, pillar_idx: int, shift: float) -> "DiscountCurve":
        """New curve with one pillar's zero rate shifted by `shift`."""
        pillar_t = [t for t in self._times if t > 0]
        pillar_df = [float(df) for t, df in zip(self._times, self._dfs) if t > 0]
        pillar_df[pillar_idx] = pillar_df[pillar_idx] * math.exp(-shift * pillar_t[pillar_idx])
        return DiscountCurve(
            self.reference_date, self.pillar_dates, pillar_df,
            self.day_count, self._interpolation,
        )

    def _time(self, d: date) -> float:
        """Year fraction from reference date to d. Returns 0.0 if d <= reference."""
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
        """Continuously compounded zero rate to date d: r = -ln(df) / t.

        At d == reference_date, returns the instantaneous forward rate at t=0
        (the short-end rate) rather than the indeterminate 0/0 limit.
        """
        t = self._time(d)
        if t <= 0:
            # Return instantaneous forward at t=0 (short-end rate)
            if len(self._times) >= 2 and self._times[1] > 0:
                df1 = float(self._dfs[1])
                if df1 > 0:
                    return -math.log(df1) / float(self._times[1])
            return 0.0
        return -math.log(self.df(d)) / t

    def instantaneous_forward(self, t_or_date) -> float:
        """Instantaneous forward rate: f(t) = -d/dt ln P(t) via finite difference.

        Args:
            t_or_date: either a year fraction (float) or a date.
        """
        if isinstance(t_or_date, date):
            d1 = t_or_date
        else:
            d1 = date_from_year_fraction(self.reference_date, t_or_date)
        d2 = date.fromordinal(d1.toordinal() + 1)
        df1 = self.df(d1)
        df2 = self.df(d2)
        if df1 <= 0 or df2 <= 0:
            return 0.0
        dt = year_fraction(d1, d2, self.day_count)
        if dt <= 0:
            return 0.0
        return -math.log(df2 / df1) / dt

    def forward_rate(self, d1: date, d2: date) -> float:
        """
        Simply compounded forward rate between d1 and d2.

        F(t1, t2) = (df(t1) - df(t2)) / (tau * df(t2))

        Numerically stable form: subtracts DFs directly rather than
        dividing near-equal values and subtracting 1. Critical for
        overnight and short-period forwards.
        """
        if d1 >= d2:
            raise ValueError(f"d1 ({d1}) must be before d2 ({d2})")
        df1 = self.df(d1)
        df2 = self.df(d2)
        tau = year_fraction(d1, d2, self.day_count)
        return (df1 - df2) / (tau * df2)

from pricebook.serialisable import _register, _serialise_atom

DiscountCurve._SERIAL_TYPE = "discount_curve"
DiscountCurve._SERIAL_FIELDS = []  # custom

def _dc_to_dict(self):
    pillar_dfs = [float(df) for t, df in zip(self._times, self._dfs) if t > 0]
    return {"type": "discount_curve", "params": {
        "reference_date": self.reference_date.isoformat(),
        "dates": [d.isoformat() for d in self.pillar_dates],
        "dfs": pillar_dfs, "day_count": self.day_count.value,
    }}

@classmethod
def _dc_from_dict(cls, d):
    from datetime import date as _d
    p = d["params"]
    return cls(reference_date=_d.fromisoformat(p["reference_date"]),
               dates=[_d.fromisoformat(s) for s in p["dates"]], dfs=p["dfs"],
               day_count=DayCountConvention(p["day_count"]))

DiscountCurve.to_dict = _dc_to_dict
DiscountCurve.from_dict = _dc_from_dict
_register(DiscountCurve)
