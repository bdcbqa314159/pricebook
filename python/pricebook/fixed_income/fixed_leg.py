"""Fixed leg of an interest rate swap."""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.schedule import Frequency, StubType, generate_schedule
from pricebook.core.calendar import Calendar, BusinessDayConvention


@dataclass
class Cashflow:
    """A single fixed cashflow."""

    accrual_start: date
    accrual_end: date
    payment_date: date
    notional: float
    rate: float
    year_frac: float

    @property
    def amount(self) -> float:
        return self.notional * self.rate * self.year_frac



    def to_dict(self) -> dict:
        return vars(self)
from pricebook.core.notional import normalize_notional as _normalize_notional


class FixedLeg:
    """
    A sequence of fixed-rate coupons.

    Each coupon pays: notional * rate * year_fraction(accrual_start, accrual_end).
    Present value is the sum of each coupon discounted to the reference date.

    Attributes:
        notional: face amount (first period notional, always float).
        notional_schedule: per-period notional list.
    """

    def __init__(
        self,
        start: date,
        end: date,
        rate: float,
        frequency: Frequency,
        notional: float | list[float] = 1_000_000.0,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
        payment_delay_days: int = 0,
    ):
        self.start = start
        self.end = end
        self.rate = rate
        self.frequency = frequency
        self.day_count = day_count
        self.payment_delay_days = payment_delay_days

        schedule = generate_schedule(
            start, end, frequency, calendar, convention, stub, eom,
        )
        n_periods = len(schedule) - 1

        self.notional_schedule = _normalize_notional(notional, n_periods)
        self.notional = self.notional_schedule[0]

        self.cashflows = []
        for i in range(1, len(schedule)):
            accrual_start = schedule[i - 1]
            accrual_end = schedule[i]
            if calendar is not None and payment_delay_days > 0:
                payment_date = calendar.add_business_days(accrual_end, payment_delay_days)
            else:
                payment_date = accrual_end + timedelta(days=payment_delay_days)
            yf = year_fraction(accrual_start, accrual_end, day_count)
            self.cashflows.append(Cashflow(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                payment_date=payment_date,
                notional=self.notional_schedule[i - 1],
                rate=rate,
                year_frac=yf,
            ))

    def pv(self, curve: DiscountCurve) -> float:
        """Present value of all cashflows discounted off the given curve."""
        return sum(cf.amount * curve.df(cf.payment_date) for cf in self.cashflows)

    def annuity(self, curve: DiscountCurve) -> float:
        """Per-unit annuity factor: sum of year_frac * df for each period.

        This is the PV of the leg per unit of (rate x notional).
        For variable notional, use weighted_annuity() instead.
        """
        return sum(
            cf.year_frac * curve.df(cf.payment_date) for cf in self.cashflows
        )

    def weighted_annuity(self, curve: DiscountCurve) -> float:
        """Notional-weighted annuity: sum of notional_i * year_frac_i * df_i.

        Correct par-rate denominator for any notional schedule.
        For uniform notional N: equals N * annuity(curve).
        """
        return sum(
            cf.notional * cf.year_frac * curve.df(cf.payment_date)
            for cf in self.cashflows
        )
