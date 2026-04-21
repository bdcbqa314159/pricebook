"""Floating leg of an interest rate swap (single-curve and dual-curve)."""

from datetime import date, timedelta
from dataclasses import dataclass

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixings import FixingsStore
from pricebook.schedule import Frequency, StubType, generate_schedule
from pricebook.calendar import Calendar, BusinessDayConvention


@dataclass
class FloatingCashflow:
    """A single floating-rate cashflow."""

    accrual_start: date
    accrual_end: date
    payment_date: date
    fixing_date: date
    observation_start: date
    observation_end: date
    notional: float
    year_frac: float
    spread: float

    def forward_rate(self, projection_curve: DiscountCurve) -> float:
        """Implied forward rate for the observation window.

        With observation shift, projects from [obs_start, obs_end] instead of
        [accrual_start, accrual_end]. Year fraction still uses accrual dates.
        """
        df1 = projection_curve.df(self.observation_start)
        df2 = projection_curve.df(self.observation_end)
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
        observation_shift_days: int = 0,
    ):
        if notional <= 0:
            raise ValueError(f"notional must be positive, got {notional}")
        if payment_delay_days < 0:
            raise ValueError(f"payment_delay_days must be >= 0, got {payment_delay_days}")
        if observation_shift_days < 0:
            raise ValueError(f"observation_shift_days must be >= 0, got {observation_shift_days}")

        self.start = start
        self.end = end
        self.frequency = frequency
        self.notional = notional
        self.spread = spread
        self.day_count = day_count
        self.calendar = calendar
        self.convention = convention
        self.stub = stub
        self.eom = eom
        self.payment_delay_days = payment_delay_days
        self.observation_shift_days = observation_shift_days

        schedule = generate_schedule(
            start, end, frequency, calendar, convention, stub, eom,
        )

        self.cashflows = []
        for i in range(1, len(schedule)):
            accrual_start = schedule[i - 1]
            accrual_end = schedule[i]
            if calendar is not None and payment_delay_days > 0:
                payment_date = calendar.add_business_days(accrual_end, payment_delay_days)
            else:
                payment_date = accrual_end + timedelta(days=payment_delay_days)
            if calendar is not None and observation_shift_days > 0:
                observation_start = calendar.add_business_days(accrual_start, -observation_shift_days)
                observation_end = calendar.add_business_days(accrual_end, -observation_shift_days)
            else:
                observation_start = accrual_start - timedelta(days=observation_shift_days)
                observation_end = accrual_end - timedelta(days=observation_shift_days)
            fixing_date = observation_start
            yf = year_fraction(accrual_start, accrual_end, day_count)
            self.cashflows.append(FloatingCashflow(
                accrual_start=accrual_start,
                accrual_end=accrual_end,
                payment_date=payment_date,
                fixing_date=fixing_date,
                observation_start=observation_start,
                observation_end=observation_end,
                notional=notional,
                year_frac=yf,
                spread=spread,
            ))

    def pv(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        fixings: FixingsStore | None = None,
        rate_name: str | None = None,
    ) -> float:
        """
        Present value of the floating leg.

        Args:
            curve: discount curve (used for discounting cashflows).
            projection_curve: forward projection curve (used for forward rates).
                If None, uses the discount curve (single-curve pricing).
            fixings: historical fixings store. If provided with rate_name,
                past accrual periods use the stored fixing instead of the
                forward curve projection.
            rate_name: index name in the fixings store (e.g. "SOFR", "EURIBOR").
        """
        proj = projection_curve if projection_curve is not None else curve
        ref = curve.reference_date
        total = 0.0
        for cf in self.cashflows:
            if fixings is not None and rate_name is not None and cf.fixing_date <= ref:
                fixing = fixings.get(rate_name, cf.fixing_date)
                if fixing is not None:
                    amount = cf.notional * (fixing + cf.spread) * cf.year_frac
                else:
                    amount = cf.amount(proj)
            else:
                amount = cf.amount(proj)
            total += amount * curve.df(cf.payment_date)
        return total
