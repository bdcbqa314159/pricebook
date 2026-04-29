"""
Basis swap: float vs float at different tenors.

E.g. 3M LIBOR vs 6M LIBOR, or SOFR vs Fed Funds. Each leg has its
own projection curve, both discounted off OIS.

    swap = BasisSwap(
        start=date(2024,1,15), end=date(2029,1,15),
        spread=0.001,  # 1bp on leg 1
        leg1_frequency=Frequency.QUARTERLY,
        leg2_frequency=Frequency.SEMI_ANNUAL,
    )
    pv = swap.pv(discount_curve, proj_curve_1, proj_curve_2)
"""

from __future__ import annotations

from datetime import date

from pricebook.day_count import DayCountConvention
from pricebook.discount_curve import DiscountCurve
from pricebook.floating_leg import FloatingLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention


class BasisSwap:
    """Float-vs-float basis swap.

    Leg 1 pays floating rate + spread, leg 2 pays floating rate flat.
    PV = PV(leg1) - PV(leg2). Par basis = spread that makes PV = 0.

    Args:
        start: effective date.
        end: maturity date.
        spread: spread on leg 1 (e.g. 0.001 = 1bp).
        notional: common notional.
        leg1_frequency: coupon frequency for leg 1.
        leg2_frequency: coupon frequency for leg 2.
        day_count: day count for both legs.
        calendar: business day calendar.
        convention: business day convention.
    """

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.0,
        notional: float = 1_000_000.0,
        leg1_frequency: Frequency = Frequency.QUARTERLY,
        leg2_frequency: Frequency = Frequency.SEMI_ANNUAL,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
        payment_delay_days: int = 0,
        observation_shift_days: int = 0,
    ):
        self.start = start
        self.end = end
        self.spread = spread
        self.notional = notional
        self.leg1_frequency = leg1_frequency
        self.leg2_frequency = leg2_frequency
        self.day_count = day_count
        self.calendar = calendar
        self.convention = convention
        self.stub = stub
        self.eom = eom
        self.payment_delay_days = payment_delay_days
        self.observation_shift_days = observation_shift_days

        self.leg1 = FloatingLeg(
            start, end, leg1_frequency,
            notional=notional, spread=spread, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
            payment_delay_days=payment_delay_days,
            observation_shift_days=observation_shift_days,
        )

        self.leg2 = FloatingLeg(
            start, end, leg2_frequency,
            notional=notional, spread=0.0, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
            payment_delay_days=payment_delay_days,
            observation_shift_days=observation_shift_days,
        )

    def pv(
        self,
        discount_curve: DiscountCurve,
        projection_curve_1: DiscountCurve,
        projection_curve_2: DiscountCurve,
    ) -> float:
        """PV = PV(leg1) - PV(leg2).

        Args:
            discount_curve: OIS curve for discounting both legs.
            projection_curve_1: forward curve for leg 1.
            projection_curve_2: forward curve for leg 2.
        """
        pv1 = self.leg1.pv(discount_curve, projection_curve_1)
        pv2 = self.leg2.pv(discount_curve, projection_curve_2)
        return pv1 - pv2

    def par_spread(
        self,
        discount_curve: DiscountCurve,
        projection_curve_1: DiscountCurve,
        projection_curve_2: DiscountCurve,
    ) -> float:
        """Par basis spread: the spread on leg 1 that makes PV = 0.

        par_spread = (PV_leg2 - PV_leg1_flat) / annuity_leg1

        where PV_leg1_flat is leg 1 with zero spread.
        """
        # PV of leg 2
        pv2 = self.leg2.pv(discount_curve, projection_curve_2)

        # PV of leg 1 with zero spread (preserve all settings)
        leg1_flat = FloatingLeg(
            self.leg1.start, self.leg1.end, self.leg1.frequency,
            notional=self.notional, spread=0.0, day_count=self.leg1.day_count,
            calendar=self.leg1.calendar, convention=self.leg1.convention,
            stub=self.leg1.stub, eom=self.leg1.eom,
            payment_delay_days=self.leg1.payment_delay_days,
            observation_shift_days=self.leg1.observation_shift_days,
        )
        pv1_flat = leg1_flat.pv(discount_curve, projection_curve_1)

        # Annuity of leg 1: sum of year_frac * df
        annuity = sum(
            cf.year_frac * discount_curve.df(cf.payment_date)
            for cf in self.leg1.cashflows
        )

        return (pv2 - pv1_flat) / (self.notional * annuity)

    def dv01(
        self,
        discount_curve: DiscountCurve,
        projection_curve_1: DiscountCurve,
        projection_curve_2: DiscountCurve,
        shift: float = 0.0001,
    ) -> float:
        """Parallel DV01: PV change for a 1bp parallel shift in all curves."""
        pv_base = self.pv(discount_curve, projection_curve_1, projection_curve_2)
        pv_bumped = self.pv(
            discount_curve.bumped(shift),
            projection_curve_1.bumped(shift),
            projection_curve_2.bumped(shift),
        )
        return pv_bumped - pv_base
