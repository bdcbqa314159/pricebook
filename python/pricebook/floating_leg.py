"""Floating leg of an interest rate swap (single-curve and dual-curve)."""

from datetime import date, timedelta
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

    def forward_rate(self, projection_curve: DiscountCurve) -> float:
        """Implied forward rate for this period using the leg's own day count."""
        df1 = projection_curve.df(self.accrual_start)
        df2 = projection_curve.df(self.accrual_end)
        return (df1 / df2 - 1.0) / self.year_frac

    def amount(self, projection_curve: DiscountCurve) -> float:
        """Projected cashflow: notional * (forward_rate + spread) * year_frac."""
        return self.notional * (self.forward_rate(projection_curve) + self.spread) * self.year_frac


class FloatingLeg:
    """
    A sequence of floating-rate coupons.

    Each coupon pays: notional * (forward_rate + spread) * year_fraction.
    Forward rates are implied from the projection curve.

    Supports dual-curve pricing:
        - projection_curve: used to compute forward rates
        - discount_curve: used to discount cashflows
    Single-curve is the special case where both are the same curve.
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
        payment_delay_days: int = 0,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if payment_delay_days < 0:
            raise ValueError(f"payment_delay_days must be >= 0, got {payment_delay_days}")

        self.start = start
        self.end = end
        self.frequency = frequency
        self.notional = notional
        self.spread = spread
        self.day_count = day_count
        self.payment_delay_days = payment_delay_days

        schedule = generate_schedule(
            start, end, frequency, calendar, convention, stub, eom,
        )

        self.cashflows = []
        for i in range(1, len(schedule)):
            accrual_start = schedule[i - 1]
            accrual_end = schedule[i]
            payment_date = accrual_end + timedelta(days=payment_delay_days)
            yf = year_fraction(accrual_start, accrual_end, day_count)
            self.cashflows.append(FloatingCashflow(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                payment_date=payment_date,
                notional=notional,
                year_frac=yf,
                spread=spread,
            ))

    def pv(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """
        Present value of the floating leg.

        Args:
            curve: discount curve (used for discounting cashflows).
            projection_curve: forward projection curve (used for forward rates).
                If None, uses the discount curve (single-curve pricing).
        """
        proj = projection_curve if projection_curve is not None else curve
        return sum(
            cf.amount(proj) * curve.df(cf.payment_date)
            for cf in self.cashflows
        )
