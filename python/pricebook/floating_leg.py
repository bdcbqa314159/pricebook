"""Floating leg of an interest rate swap (single-curve)."""

from datetime import date
from dataclasses import dataclass

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention


@dataclass
class FloatingCashflow:
    """A single floating-rate cashflow."""

    accrual_start: date
    accrual_end: date
    payment_date: date
    notional: float
    year_frac: float
    spread: float

    def forward_rate(self, curve: DiscountCurve) -> float:
        """Implied forward rate for this period using the leg's own day count."""
        df1 = curve.df(self.accrual_start)
        df2 = curve.df(self.accrual_end)
        return (df1 / df2 - 1.0) / self.year_frac

    def amount(self, curve: DiscountCurve) -> float:
        """Projected cashflow: notional * (forward_rate + spread) * year_frac."""
        return self.notional * (self.forward_rate(curve) + self.spread) * self.year_frac


class FloatingLeg:
    """
    A sequence of floating-rate coupons (single-curve pricing).

    Each coupon pays: notional * (forward_rate + spread) * year_fraction.
    Forward rates are implied from the discount curve: F = (df1/df2 - 1) / tau.

    In single-curve pricing, the same curve is used for projecting forwards
    and for discounting. This gives the well-known telescoping result:
        PV(floating, no spread) = df(start) - df(end)
    """

    def __init__(
        self,
        start: date,
        end: date,
        frequency: Frequency,
        notional: float = 1_000_000.0,
        spread: float = 0.0,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")

        self.start = start
        self.end = end
        self.frequency = frequency
        self.notional = notional
        self.spread = spread
        self.day_count = day_count

        schedule = generate_schedule(
            start, end, frequency, calendar, convention, stub, eom,
        )

        self.cashflows = []
        for i in range(1, len(schedule)):
            accrual_start = schedule[i - 1]
            accrual_end = schedule[i]
            yf = year_fraction(accrual_start, accrual_end, day_count)
            self.cashflows.append(FloatingCashflow(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                payment_date=accrual_end,
                notional=notional,
                year_frac=yf,
                spread=spread,
            ))

    def pv(self, curve: DiscountCurve) -> float:
        """Present value: sum of each projected cashflow discounted to reference date."""
        return sum(
            cf.amount(curve) * curve.df(cf.payment_date)
            for cf in self.cashflows
        )
