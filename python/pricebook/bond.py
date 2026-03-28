"""Fixed-rate bond."""

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixed_leg import FixedLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.solvers import brentq


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

    def yield_to_maturity(self, market_price: float) -> float:
        """
        Yield to maturity: the constant rate that discounts all cashflows
        to the given market (dirty) price.

        Uses bond-equivalent yield convention: compounding at the coupon frequency.

        YTM solves: price = sum(C_i / (1 + y/freq)^(freq*t_i)) + face / (1 + y/freq)^(freq*t_n)
        """
        freq = self.frequency.value  # months per period
        periods_per_year = 12 / freq

        def _price_from_yield(y: float) -> float:
            pv = 0.0
            for cf in self.coupon_leg.cashflows:
                t = year_fraction(self.issue_date, cf.payment_date, self.day_count)
                n = t * periods_per_year
                pv += cf.amount / (1.0 + y / periods_per_year) ** n
            # Principal
            t_mat = year_fraction(self.issue_date, self.maturity, self.day_count)
            n_mat = t_mat * periods_per_year
            pv += self.face_value / (1.0 + y / periods_per_year) ** n_mat
            return pv / self.face_value * 100.0

        def objective(y: float) -> float:
            return _price_from_yield(y) - market_price

        return brentq(objective, -0.05, 1.0)

    def macaulay_duration(self, ytm: float) -> float:
        """
        Macaulay duration: weighted-average time to cashflows,
        where weights are PV of each cashflow / total PV.
        """
        freq = self.frequency.value
        periods_per_year = 12 / freq

        weighted_t = 0.0
        total_pv = 0.0
        for cf in self.coupon_leg.cashflows:
            t = year_fraction(self.issue_date, cf.payment_date, self.day_count)
            n = t * periods_per_year
            pv = cf.amount / (1.0 + ytm / periods_per_year) ** n
            weighted_t += t * pv
            total_pv += pv

        # Principal
        t_mat = year_fraction(self.issue_date, self.maturity, self.day_count)
        n_mat = t_mat * periods_per_year
        pv_prin = self.face_value / (1.0 + ytm / periods_per_year) ** n_mat
        weighted_t += t_mat * pv_prin
        total_pv += pv_prin

        return weighted_t / total_pv

    def modified_duration(self, ytm: float) -> float:
        """Modified duration: Macaulay duration / (1 + ytm/freq)."""
        periods_per_year = 12 / self.frequency.value
        return self.macaulay_duration(ytm) / (1.0 + ytm / periods_per_year)

    def convexity(self, ytm: float) -> float:
        """
        Convexity: second-order price sensitivity to yield.

        C = (1/P) * sum(t_i * (t_i + 1/freq) * PV_i) / (1 + y/freq)^2
        """
        freq = self.frequency.value
        periods_per_year = 12 / freq
        discount = 1.0 + ytm / periods_per_year

        weighted = 0.0
        total_pv = 0.0
        for cf in self.coupon_leg.cashflows:
            t = year_fraction(self.issue_date, cf.payment_date, self.day_count)
            n = t * periods_per_year
            pv = cf.amount / discount ** n
            weighted += n * (n + 1) * pv
            total_pv += pv

        t_mat = year_fraction(self.issue_date, self.maturity, self.day_count)
        n_mat = t_mat * periods_per_year
        pv_prin = self.face_value / discount ** n_mat
        weighted += n_mat * (n_mat + 1) * pv_prin
        total_pv += pv_prin

        return weighted / (total_pv * periods_per_year ** 2 * discount ** 2)

    def dv01_yield(self, ytm: float) -> float:
        """Dollar value of a basis point: price change for a 1bp yield shift."""
        return self.modified_duration(ytm) * self._price_from_ytm(ytm) / 10000.0

    def _price_from_ytm(self, ytm: float) -> float:
        """Dirty price per 100 face from a yield."""
        freq = self.frequency.value
        periods_per_year = 12 / freq
        pv = 0.0
        for cf in self.coupon_leg.cashflows:
            t = year_fraction(self.issue_date, cf.payment_date, self.day_count)
            n = t * periods_per_year
            pv += cf.amount / (1.0 + ytm / periods_per_year) ** n
        t_mat = year_fraction(self.issue_date, self.maturity, self.day_count)
        n_mat = t_mat * periods_per_year
        pv += self.face_value / (1.0 + ytm / periods_per_year) ** n_mat
        return pv / self.face_value * 100.0
