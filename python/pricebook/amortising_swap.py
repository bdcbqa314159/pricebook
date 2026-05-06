"""Amortising, accreting, and roller-coaster swaps.

DEPRECATED: Use InterestRateSwap(notional=[...]) instead.

This module provides a compatibility wrapper. InterestRateSwap now
natively supports per-period notional schedules via notional_schedule.

    # New way (preferred):
    from pricebook.swap import InterestRateSwap
    swap = InterestRateSwap(start, end, rate, notional=[1e6, 800e3, 600e3, ...])

    # Old way (deprecated, still works):
    from pricebook.amortising_swap import AmortisingSwap
    swap = AmortisingSwap(start, end, rate, notional_schedule=[1e6, 800e3, ...])
"""

from __future__ import annotations

import warnings
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.swap import InterestRateSwap
from pricebook.calendar import Calendar, BusinessDayConvention


class AmortisingSwap:
    """Interest rate swap with per-period notional schedule.

    DEPRECATED: Use InterestRateSwap(notional=[...]) instead.

    Supports amortising (decreasing), accreting (increasing), and
    roller-coaster (arbitrary) notional profiles.
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_rate: float,
        notional_schedule: list[float],
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        float_day_count: DayCountConvention = DayCountConvention.ACT_360,
        spread: float = 0.0,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
    ):
        warnings.warn(
            "AmortisingSwap is deprecated. Use InterestRateSwap(notional=[...]) instead.",
            DeprecationWarning, stacklevel=2,
        )

        self._swap = InterestRateSwap(
            start, end, fixed_rate,
            notional=notional_schedule,
            fixed_frequency=frequency,
            float_frequency=frequency,
            fixed_day_count=fixed_day_count,
            float_day_count=float_day_count,
            spread=spread,
            calendar=calendar,
            convention=convention,
        )

        # Expose attributes for backward compat
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.spread = spread
        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count
        self.notionals = list(self._swap.notional_schedule)
        self.periods = list(zip(
            [cf.accrual_start for cf in self._swap.fixed_leg.cashflows],
            [cf.accrual_end for cf in self._swap.fixed_leg.cashflows],
        ))

    @classmethod
    def amortising(
        cls,
        start: date,
        end: date,
        fixed_rate: float,
        initial_notional: float,
        **kwargs,
    ) -> AmortisingSwap:
        """Linear amortising swap: notional decreases evenly to zero."""
        freq = kwargs.get("frequency", Frequency.SEMI_ANNUAL)
        schedule = generate_schedule(start, end, freq)
        n = len(schedule) - 1
        notionals = [initial_notional * (1.0 - i / n) for i in range(n)]
        return cls(start, end, fixed_rate, notionals, **kwargs)

    @classmethod
    def accreting(
        cls,
        start: date,
        end: date,
        fixed_rate: float,
        initial_notional: float,
        final_notional: float,
        **kwargs,
    ) -> AmortisingSwap:
        """Linear accreting swap: notional increases from initial to final."""
        freq = kwargs.get("frequency", Frequency.SEMI_ANNUAL)
        schedule = generate_schedule(start, end, freq)
        n = len(schedule) - 1
        notionals = [
            initial_notional + (final_notional - initial_notional) * i / max(n - 1, 1)
            for i in range(n)
        ]
        return cls(start, end, fixed_rate, notionals, **kwargs)

    def pv_fixed(self, curve: DiscountCurve) -> float:
        return self._swap.fixed_leg.pv(curve)

    def pv_float(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        return self._swap.floating_leg.pv(curve, projection)

    def pv(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        return self._swap.pv(curve, projection)

    def par_rate(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        return self._swap.par_rate(curve, projection)

    def dv01(self, curve: DiscountCurve, shift: float = 0.0001) -> float:
        pv_base = self.pv(curve)
        pv_bumped = self.pv(curve.bumped(shift))
        return pv_bumped - pv_base

    @property
    def average_notional(self) -> float:
        return sum(self.notionals) / len(self.notionals)

    @property
    def weighted_average_life(self) -> float:
        """WAL in years from start date."""
        total = 0.0
        for (s, e), notl in zip(self.periods, self.notionals):
            mid = year_fraction(self.start, e, DayCountConvention.ACT_365_FIXED)
            total += notl * mid
        return total / sum(self.notionals)
