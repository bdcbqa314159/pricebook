"""Interest rate swap."""

from datetime import date
from dataclasses import dataclass
from enum import Enum

from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.fixed_leg import FixedLeg
from pricebook.floating_leg import FloatingLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention


class SwapDirection(Enum):
    PAYER = "payer"        # pay fixed, receive floating
    RECEIVER = "receiver"  # receive fixed, pay floating


class InterestRateSwap:
    """
    A vanilla interest rate swap: fixed leg vs floating leg.

    Payer swap: pay fixed, receive floating -> PV = PV(float) - PV(fixed)
    Receiver swap: receive fixed, pay floating -> PV = PV(fixed) - PV(float)

    Supports dual-curve pricing:
        - discount_curve: used for discounting all cashflows
        - projection_curve: used for computing floating forward rates
    Single-curve is the special case where both are the same.
    """

    def __init__(
        self,
        start: date,
        end: date,
        fixed_rate: float,
        direction: SwapDirection = SwapDirection.PAYER,
        notional: float = 1_000_000.0,
        fixed_frequency: Frequency = Frequency.SEMI_ANNUAL,
        float_frequency: Frequency = Frequency.QUARTERLY,
        fixed_day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        float_day_count: DayCountConvention = DayCountConvention.ACT_360,
        spread: float = 0.0,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
        payment_delay_days: int = 0,
        observation_shift_days: int = 0,
    ):
        self.start = start
        self.end = end
        self.fixed_rate = fixed_rate
        self.direction = direction
        self.notional = notional
        self.fixed_frequency = fixed_frequency
        self.float_frequency = float_frequency
        self.fixed_day_count = fixed_day_count
        self.float_day_count = float_day_count
        self.spread = spread

        self.fixed_leg = FixedLeg(
            start, end, fixed_rate, fixed_frequency,
            notional=notional, day_count=fixed_day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
        )

        self.floating_leg = FloatingLeg(
            start, end, float_frequency,
            notional=notional, spread=spread, day_count=float_day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
            payment_delay_days=payment_delay_days,
            observation_shift_days=observation_shift_days,
        )

    def pv(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """
        Present value of the swap.

        Args:
            curve: discount curve.
            projection_curve: forward projection curve. If None, single-curve pricing.
        """
        pv_fixed = self.fixed_leg.pv(curve)
        pv_float = self.floating_leg.pv(curve, projection_curve)
        if self.direction == SwapDirection.PAYER:
            return pv_float - pv_fixed
        return pv_fixed - pv_float

    def par_rate(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """
        The fixed rate that makes PV = 0.

        par_rate = PV_float(projection, discount) / (notional * annuity(discount))
        """
        annuity = self.fixed_leg.annuity(curve)
        pv_float = self.floating_leg.pv(curve, projection_curve)
        return pv_float / (self.notional * annuity)

    def cashflow_schedule(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> list[dict]:
        """Produce a full cashflow schedule for both legs.

        Returns a list of dicts with: leg, accrual_start, accrual_end,
        payment_date, rate, year_frac, amount, df, pv.
        """
        proj = projection_curve if projection_curve is not None else curve
        rows = []
        for cf in self.fixed_leg.cashflows:
            df = curve.df(cf.payment_date)
            rows.append({
                "leg": "fixed",
                "accrual_start": cf.accrual_start,
                "accrual_end": cf.accrual_end,
                "payment_date": cf.payment_date,
                "rate": cf.rate,
                "year_frac": cf.year_frac,
                "amount": cf.amount,
                "df": df,
                "pv": cf.amount * df,
            })
        for cf in self.floating_leg.cashflows:
            fwd = cf.forward_rate(proj)
            amount = cf.amount(proj)
            df = curve.df(cf.payment_date)
            rows.append({
                "leg": "float",
                "accrual_start": cf.accrual_start,
                "accrual_end": cf.accrual_end,
                "payment_date": cf.payment_date,
                "rate": fwd,
                "year_frac": cf.year_frac,
                "amount": amount,
                "df": df,
                "pv": amount * df,
            })
        return sorted(rows, key=lambda r: r["payment_date"])
