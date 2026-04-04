"""Survival curve for credit modeling."""

import math
from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.interpolation import (
    InterpolationMethod,
    create_interpolator,
    Interpolator,
)


class SurvivalCurve:
    """
    A survival curve maps dates to survival probabilities.

    Q(t) = probability of no default by time t.
    Under piecewise constant hazard rates: Q(t) = exp(-∫h(s)ds).

    Internally the curve stores survival probabilities at pillar dates and
    interpolates in log-space (piecewise constant hazard rates between pillars).

    Provides:
        - survival(date) -> Q(t), the survival probability
        - hazard_rate(date) -> instantaneous hazard rate at t
        - default_prob(d1, d2) -> probability of default between d1 and d2
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        survival_probs: list[float],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
        interpolation: InterpolationMethod = InterpolationMethod.LOG_LINEAR,
    ):
        if len(dates) != len(survival_probs):
            raise ValueError("dates and survival_probs must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")
        for sp in survival_probs:
            if sp <= 0 or sp > 1:
                raise ValueError(f"survival probabilities must be in (0, 1], got {sp}")

        self.reference_date = reference_date
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        # Prepend t=0, Q=1 if not already present
        if times[0] > 0:
            times = [0.0] + times
            survival_probs = [1.0] + list(survival_probs)

        self._times = np.array(times)
        self._survs = np.array(survival_probs)
        self._interpolator: Interpolator = create_interpolator(
            interpolation, self._times, self._survs
        )

    @classmethod
    def flat(cls, reference_date: date, hazard_rate: float, tenors: list[int] | None = None) -> "SurvivalCurve":
        """Build a flat survival curve at a constant hazard rate."""
        from dateutil.relativedelta import relativedelta
        if tenors is None:
            tenors = [1, 2, 3, 5, 7, 10]
        dates = [reference_date + relativedelta(years=t) for t in tenors]
        survs = [math.exp(-hazard_rate * t) for t in tenors]
        return cls(reference_date, dates, survs)

    def _time(self, d: date) -> float:
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def survival(self, d: date) -> float:
        """Survival probability Q(t). Returns 1.0 for d <= reference_date."""
        t = self._time(d)
        if t <= 0:
            return 1.0
        return self._interpolator(t)

    def hazard_rate(self, d: date) -> float:
        """
        Piecewise constant hazard rate at date d.

        h(t) = -d/dt ln(Q(t)). For piecewise constant hazard between pillars:
        h = -ln(Q(t2)/Q(t1)) / (t2 - t1)
        """
        t = self._time(d)
        if t <= 0:
            return 0.0
        # Find the segment
        idx = int(np.searchsorted(self._times, t)) - 1
        idx = max(0, min(idx, len(self._times) - 2))
        t1, t2 = self._times[idx], self._times[idx + 1]
        q1, q2 = self._survs[idx], self._survs[idx + 1]
        if t2 <= t1 or q2 <= 0 or q1 <= 0:
            return 0.0
        return -math.log(q2 / q1) / (t2 - t1)

    def default_prob(self, d1: date, d2: date) -> float:
        """Probability of default between d1 and d2: Q(d1) - Q(d2)."""
        if d1 >= d2:
            raise ValueError(f"d1 ({d1}) must be before d2 ({d2})")
        return self.survival(d1) - self.survival(d2)
