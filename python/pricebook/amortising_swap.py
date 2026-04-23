"""Amortising, accreting, and roller-coaster swaps.

Notional varies per period according to a schedule. Each period's
fixed and floating cashflows use that period's notional.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention


class AmortisingSwap:
    """Interest rate swap with per-period notional schedule.

    Supports amortising (decreasing), accreting (increasing), and
    roller-coaster (arbitrary) notional profiles.

    Args:
        start: effective date.
        end: maturity date.
        fixed_rate: fixed coupon rate.
        notional_schedule: list of notionals, one per period.
            If shorter than the number of periods, the last value is repeated.
        frequency: payment frequency (same for fixed and float).
        fixed_day_count: day count for fixed leg.
        float_day_count: day count for floating leg.
        spread: floating leg spread.
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
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.spread = spread
        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count

        schedule = generate_schedule(start, end, frequency, calendar, convention,
                                     StubType.SHORT_FRONT, True)
        self.periods = list(zip(schedule[:-1], schedule[1:]))
        n = len(self.periods)

        # Extend notional schedule if needed
        if len(notional_schedule) < n:
            notional_schedule = list(notional_schedule) + \
                [notional_schedule[-1]] * (n - len(notional_schedule))
        self.notionals = notional_schedule[:n]

    @classmethod
    def amortising(
        cls,
        start: date,
        end: date,
        fixed_rate: float,
        initial_notional: float,
        **kwargs,
    ) -> "AmortisingSwap":
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
    ) -> "AmortisingSwap":
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
        """PV of the fixed leg."""
        pv = 0.0
        for (s, e), notl in zip(self.periods, self.notionals):
            yf = year_fraction(s, e, self.fixed_day_count)
            pv += notl * self.fixed_rate * yf * curve.df(e)
        return pv

    def pv_float(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        """PV of the floating leg."""
        proj = projection or curve
        pv = 0.0
        for (s, e), notl in zip(self.periods, self.notionals):
            yf = year_fraction(s, e, self.float_day_count)
            # Forward rate using float_day_count (not the curve's internal day count)
            df1 = proj.df(s)
            df2 = proj.df(e)
            fwd = (df1 - df2) / (yf * df2)
            pv += notl * (fwd + self.spread) * yf * curve.df(e)
        return pv

    def pv(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        """PV of the swap (payer = receive float - pay fixed)."""
        return self.pv_float(curve, projection) - self.pv_fixed(curve)

    def par_rate(self, curve: DiscountCurve, projection: DiscountCurve | None = None) -> float:
        """Fixed rate that makes PV = 0."""
        annuity = sum(
            notl * year_fraction(s, e, self.fixed_day_count) * curve.df(e)
            for (s, e), notl in zip(self.periods, self.notionals)
        )
        if abs(annuity) < 1e-12:
            return 0.0
        float_pv = self.pv_float(curve, projection)
        return float_pv / annuity

    def dv01(self, curve: DiscountCurve, shift: float = 0.0001) -> float:
        """Parallel DV01."""
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
