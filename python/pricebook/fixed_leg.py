"""Fixed leg of an interest rate swap."""

from __future__ import annotations

from datetime import date, timedelta
from dataclasses import dataclass

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention


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


class FixedLeg:
    """
    A sequence of fixed-rate coupons.

    Each coupon pays: notional * rate * year_fraction(accrual_start, accrual_end).
    Present value is the sum of each coupon discounted to the reference date.
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
        self.notional = notional  # original input (scalar or list)
        self.day_count = day_count
        self.payment_delay_days = payment_delay_days

        schedule = generate_schedule(
            start, end, frequency, calendar, convention, stub, eom,
        )
        n_periods = len(schedule) - 1

        # Normalize notional to per-period list
        if isinstance(notional, (int, float)):
            if notional <= 0:
                raise ValueError(f"notional must be positive, got {notional}")
            notionals = [float(notional)] * n_periods
        else:
            if not notional:
                raise ValueError("notional schedule is empty")
            if any(n <= 0 for n in notional):
                raise ValueError(f"all notionals must be positive, got {notional}")
            notionals = list(notional)
            if len(notionals) < n_periods:
                notionals += [notionals[-1]] * (n_periods - len(notionals))
            notionals = notionals[:n_periods]
        self.notionals = notionals

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
                notional=notionals[i - 1],
                rate=rate,
                year_frac=yf,
            ))

    def pv(self, curve: DiscountCurve) -> float:
        """Present value of all cashflows discounted off the given curve."""
        return sum(cf.amount * curve.df(cf.payment_date) for cf in self.cashflows)

    def annuity(self, curve: DiscountCurve) -> float:
        """
        The DV01-weighted annuity factor: sum of year_frac * df for each period.

        This is the present value of the leg per unit of fixed rate:
            PV = rate * notional * annuity
        """
        return sum(
            cf.year_frac * curve.df(cf.payment_date) for cf in self.cashflows
        )

    def weighted_annuity(self, curve: DiscountCurve) -> float:
        """Notional-weighted annuity: sum of notional_i * year_frac_i * df_i.

        This is the correct denominator for par rate when notional varies
        per period (amortising, accreting, roller-coaster).  For uniform
        notional N it equals N * annuity(curve).
        """
        return sum(
            cf.notional * cf.year_frac * curve.df(cf.payment_date)
            for cf in self.cashflows
        )
