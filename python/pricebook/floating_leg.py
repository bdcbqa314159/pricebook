"""Floating leg of an interest rate swap (single-curve and dual-curve)."""

from datetime import date, timedelta
from dataclasses import dataclass

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixings import FixingsStore
from pricebook.rate_index import CompoundingMethod, RateIndex
from pricebook.rfr import compound_rfr
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
        return (df1 - df2) / (self.year_frac * df2)

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
        self.rate_index: RateIndex | None = None

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

    @classmethod
    def from_rate_index(
        cls,
        rate_index: RateIndex,
        start: date,
        end: date,
        frequency: Frequency,
        notional: float = 1_000_000.0,
        spread: float = 0.0,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ) -> "FloatingLeg":
        """Create a FloatingLeg with conventions from a RateIndex.

        Automatically sets day_count, observation_shift, and payment_delay
        from the index definition. Stores rate_index for RFR compounding.
        """
        leg = cls(
            start=start,
            end=end,
            frequency=frequency,
            notional=notional,
            spread=spread,
            day_count=rate_index.day_count,
            calendar=calendar,
            convention=convention,
            stub=stub,
            eom=eom,
            payment_delay_days=rate_index.payment_delay,
            observation_shift_days=rate_index.observation_shift,
        )
        leg.rate_index = rate_index
        return leg

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
        use_compounding = (
            self.rate_index is not None
            and self.rate_index.compounding == CompoundingMethod.COMPOUNDED
            and fixings is not None
            and rate_name is not None
        )
        total = 0.0
        for cf in self.cashflows:
            if fixings is not None and rate_name is not None and cf.fixing_date <= ref:
                if use_compounding:
                    rate = self._compound_period(
                        cf, fixings, rate_name, self.day_count, self.calendar,
                    )
                else:
                    rate = fixings.get(rate_name, cf.fixing_date)
                if rate is not None:
                    amount = cf.notional * (rate + cf.spread) * cf.year_frac
                else:
                    amount = cf.amount(proj)
            else:
                amount = cf.amount(proj)
            total += amount * curve.df(cf.payment_date)
        return total

    @staticmethod
    def _compound_period(
        cf: FloatingCashflow,
        fixings: FixingsStore,
        rate_name: str,
        day_count: DayCountConvention,
        calendar: "Calendar | None" = None,
    ) -> float | None:
        """Compound daily overnight fixings over the observation window.

        Uses calendar.is_business_day() if available, otherwise skips
        weekends only. Returns the annualised compounded rate, or None
        if insufficient fixings.
        """
        from pricebook.calendar import Calendar

        daily_rates: list[float] = []
        day_fracs: list[float] = []
        d = cf.observation_start
        while d < cf.observation_end:
            if calendar is not None:
                is_bday = calendar.is_business_day(d)
            else:
                is_bday = d.weekday() < 5
            if is_bday:
                rate = fixings.get(rate_name, d)
                if rate is None:
                    return None  # Incomplete fixings → fall back to curve
                daily_rates.append(rate)
                # Day fraction covers from this bday to the next bday
                # (e.g. Fri→Mon = 3 calendar days)
                next_bday = d + timedelta(days=1)
                while next_bday < cf.observation_end:
                    if calendar is not None:
                        if calendar.is_business_day(next_bday):
                            break
                    else:
                        if next_bday.weekday() < 5:
                            break
                    next_bday += timedelta(days=1)
                day_fracs.append(year_fraction(d, next_bday, day_count))
            d = d + timedelta(days=1)
        if not daily_rates:
            return None
        return compound_rfr(daily_rates, day_fracs)
