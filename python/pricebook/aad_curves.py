"""AAD-aware discount and survival curves.

Number-valued pillar DFs/survivals, returning Number from df(t) and
survival(t) so that derivatives flow through pricing to curve inputs.
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.aad import Number
from pricebook.aad_interp import aad_log_linear_interp
from pricebook.day_count import DayCountConvention, year_fraction


class AADDiscountCurve:
    """Discount curve with Number-valued pillar DFs.

    Supports df(date) -> Number, so the adjoint graph links pricing
    output to each pillar DF.
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        dfs: list[Number],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(dates) != len(dfs):
            raise ValueError("dates and dfs must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")

        self.reference_date = reference_date
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        if times[0] > 0:
            times = [0.0] + times
            dfs = [Number(1.0)] + list(dfs)

        self._times = times
        self._dfs = dfs

    def _time(self, d: date) -> float:
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def df(self, d: date) -> Number:
        """Discount factor at date d. Returns Number on tape."""
        t = self._time(d)
        if t <= 0:
            return Number(1.0)
        return aad_log_linear_interp(t, self._times, self._dfs)

    @property
    def pillar_dfs(self) -> list[Number]:
        return self._dfs

    @property
    def pillar_times(self) -> list[float]:
        return self._times


class AADSurvivalCurve:
    """Survival curve with Number-valued pillar survival probabilities.

    Supports survival(date) -> Number for AAD CS01 per pillar.
    """

    def __init__(
        self,
        reference_date: date,
        dates: list[date],
        survivals: list[Number],
        day_count: DayCountConvention = DayCountConvention.ACT_365_FIXED,
    ):
        if len(dates) != len(survivals):
            raise ValueError("dates and survivals must have the same length")
        if len(dates) < 1:
            raise ValueError("need at least 1 pillar point")

        self.reference_date = reference_date
        self.day_count = day_count

        times = [year_fraction(reference_date, d, day_count) for d in dates]

        if times[0] > 0:
            times = [0.0] + times
            survivals = [Number(1.0)] + list(survivals)

        self._times = times
        self._survivals = survivals

    def _time(self, d: date) -> float:
        if d <= self.reference_date:
            return 0.0
        return year_fraction(self.reference_date, d, self.day_count)

    def survival(self, d: date) -> Number:
        """Survival probability at date d. Returns Number on tape."""
        t = self._time(d)
        if t <= 0:
            return Number(1.0)
        return aad_log_linear_interp(t, self._times, self._survivals)

    @property
    def pillar_survivals(self) -> list[Number]:
        return self._survivals

    @property
    def pillar_times(self) -> list[float]:
        return self._times
