"""Fixed leg of an interest rate swap."""

from datetime import date
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
        notional: float = 1_000_000.0,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")

        self.start = start
        self.end = end
        self.rate = rate
        self.frequency = frequency
        self.notional = notional
        self.day_count = day_count

        schedule = generate_schedule(
            start, end, frequency, calendar, convention, stub, eom,
        )

        self.cashflows = []
        for i in range(1, len(schedule)):
            accrual_start = schedule[i - 1]
            accrual_end = schedule[i]
            yf = year_fraction(accrual_start, accrual_end, day_count)
            self.cashflows.append(Cashflow(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                payment_date=accrual_end,
                notional=notional,
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
