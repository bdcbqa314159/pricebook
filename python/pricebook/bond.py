"""Fixed-rate bond."""

from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixed_leg import FixedLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention


class FixedRateBond:
    """
    A fixed-rate bond: periodic coupons plus principal at maturity.

    Dirty price = PV of all remaining cashflows (coupons + principal)
    Clean price = dirty price - accrued interest
    Accrued interest = coupon rate * year_fraction from last coupon to settlement
    """

    def __init__(
        self,
        issue_date: date,
        maturity: date,
        coupon_rate: float,
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        face_value: float = 100.0,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
    ):
        if face_value <= 0:
            raise ValueError(f"face_value must be positive, got {face_value}")
        if issue_date >= maturity:
            raise ValueError(f"issue_date ({issue_date}) must be before maturity ({maturity})")

        self.issue_date = issue_date
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.frequency = frequency
        self.face_value = face_value
        self.day_count = day_count

        self.coupon_leg = FixedLeg(
            issue_date, maturity, coupon_rate, frequency,
            notional=face_value, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
        )

    def dirty_price(self, curve: DiscountCurve) -> float:
        """Full price: PV of coupons + PV of principal, per unit of face value."""
        pv_coupons = self.coupon_leg.pv(curve)
        pv_principal = self.face_value * curve.df(self.maturity)
        return (pv_coupons + pv_principal) / self.face_value * 100.0

    def accrued_interest(self, settlement: date) -> float:
        """
        Accrued interest per 100 face at the settlement date.

        Looks for the accrual period containing the settlement date.
        """
        for cf in self.coupon_leg.cashflows:
            if cf.accrual_start <= settlement < cf.accrual_end:
                yf = year_fraction(cf.accrual_start, settlement, self.day_count)
                return self.coupon_rate * yf * 100.0

        # Settlement on or after last coupon: no accrued
        return 0.0

    def clean_price(self, curve: DiscountCurve, settlement: date | None = None) -> float:
        """
        Quoted price: dirty price minus accrued interest.

        If settlement is None, uses the curve's reference date.
        """
        if settlement is None:
            settlement = curve.reference_date
        return self.dirty_price(curve) - self.accrued_interest(settlement)
