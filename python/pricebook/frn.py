"""
Floating-rate note (FRN).

An FRN pays periodic coupons at a floating rate (forward rate + spread)
plus principal at maturity. Reuses FloatingLeg for coupon generation.

At par when spread = 0 and priced off its own projection curve.

    frn = FloatingRateNote(
        start=date(2024,1,15), end=date(2029,1,15),
        spread=0.005, notional=1_000_000,
    )
    dirty = frn.dirty_price(discount_curve, projection_curve)
"""

from __future__ import annotations

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.floating_leg import FloatingLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.solvers import brentq


class FloatingRateNote:
    """Floating-rate note: floating coupons + principal at maturity.

    Args:
        start: issue / effective date.
        end: maturity date.
        spread: fixed spread over the floating index (e.g. 0.005 = 50bp).
        notional: face value.
        frequency: coupon frequency.
        day_count: day count for accrual.
        calendar: business day calendar.
        convention: business day convention.
        stub: stub type.
        eom: end-of-month rule.
    """

    def __init__(
        self,
        start: date,
        end: date,
        spread: float = 0.0,
        notional: float = 1_000_000.0,
        frequency: Frequency = Frequency.QUARTERLY,
        day_count: DayCountConvention = DayCountConvention.ACT_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        self.start = start
        self.end = end
        self.spread = spread
        self.notional = notional

        self.floating_leg = FloatingLeg(
            start, end, frequency,
            notional=notional, spread=spread, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
        )

    def dirty_price(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Full price (per 100 face): PV of coupons + PV of principal.

        Args:
            curve: discount curve.
            projection_curve: forward rate projection curve. If None, single-curve.
        """
        pv_coupons = self.floating_leg.pv(curve, projection_curve)
        pv_principal = self.notional * curve.df(self.end)
        return (pv_coupons + pv_principal) / self.notional * 100.0

    def clean_price(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
        settlement: date | None = None,
    ) -> float:
        """Quoted price: dirty price minus accrued interest.

        Accrued = notional * (forward_rate + spread) * accrual_fraction
        from the last coupon date to settlement.
        """
        dirty = self.dirty_price(curve, projection_curve)
        if settlement is None:
            settlement = curve.reference_date

        accrued = self._accrued_interest(curve, projection_curve, settlement)
        return dirty - accrued / self.notional * 100.0

    def _accrued_interest(
        self,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None,
        settlement: date,
    ) -> float:
        """Accrued interest at settlement date."""
        proj = projection_curve if projection_curve is not None else curve
        for cf in self.floating_leg.cashflows:
            if cf.accrual_start <= settlement < cf.accrual_end:
                fwd = proj.forward_rate(cf.accrual_start, cf.accrual_end)
                yf_accrued = year_fraction(
                    cf.accrual_start, settlement, self.floating_leg.day_count,
                )
                return self.notional * (fwd + self.spread) * yf_accrued
        return 0.0

    def discount_margin(
        self,
        market_price: float,
        curve: DiscountCurve,
        projection_curve: DiscountCurve | None = None,
    ) -> float:
        """Discount margin: the spread to the discount curve that reprices the FRN.

        Solves for DM such that dirty_price(curve_shifted_by_DM) = market_price.
        Approximated by adjusting the floating spread and repricing.
        """
        def objective(dm: float) -> float:
            # Create an FRN with adjusted spread and reprice
            shifted = FloatingRateNote(
                self.start, self.end,
                spread=self.spread + dm,
                notional=self.notional,
                frequency=self.floating_leg.frequency,
            )
            return shifted.dirty_price(curve, projection_curve) - market_price

        return brentq(objective, -0.05, 0.05)
