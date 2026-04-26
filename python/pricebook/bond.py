"""Fixed-rate bond.

Includes yield-based pricing (simply-compounded, continuous Hull-form)
for Treasury Lock and other yield-curve analytics (Pucci 2019).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.fixed_leg import FixedLeg
from pricebook.schedule import Frequency, StubType
from pricebook.calendar import Calendar, BusinessDayConvention
from pricebook.solvers import brentq


# ---- Yield-based bond pricing (standalone functions) ----
# Used by Treasury Lock, bond analytics, and FixedRateBond methods.
# Reference: Pucci (2019) Eq 2-5, SSRN 3386521.

def bond_price_from_yield(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """Simply-compounded bond price from yield (Pucci Eq 2).

    P(y) = prod_i 1/(1+alpha_i*y) + c * sum_i alpha_i * prod_{j<=i} 1/(1+alpha_j*y)
    """
    n = len(accrual_factors)
    if n == 0:
        return 1.0
    cum_df = 1.0
    coupon_pv = 0.0
    for i in range(n):
        cum_df /= (1 + accrual_factors[i] * y)
        coupon_pv += coupon_rate * accrual_factors[i] * cum_df
    return cum_df + coupon_pv


def bond_price_from_yield_stub(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
    stub_fraction: float,
) -> float:
    """Simply-compounded bond price within a coupon period (Pucci Eq 3).

    Fractional first-period discount, then remaining bond + first coupon.
    """
    if len(accrual_factors) < 1:
        return 1.0
    first_df = (1 / (1 + accrual_factors[0] * y)) ** stub_fraction
    remaining = accrual_factors[1:]
    bond_from_2 = bond_price_from_yield(coupon_rate, remaining, y) if remaining else 1.0
    value_at_t1 = coupon_rate * accrual_factors[0] + bond_from_2
    return first_df * value_at_t1


def bond_price_continuous(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> float:
    """Continuously-compounded bond price, Hull-form (Pucci Eq 4).

    P(y) = e^{-T*y} + c * sum alpha_i * e^{-(t_i-t)*y}
    """
    redemption = math.exp(-time_to_maturity * y)
    annuity = sum(
        coupon_rate * alpha * math.exp(-tau * y)
        for alpha, tau in zip(accrual_factors, times_to_coupon)
    )
    return redemption + annuity


def bond_yield_derivatives(
    coupon_rate: float,
    accrual_factors: list[float],
    times_to_coupon: list[float],
    time_to_maturity: float,
    y: float,
) -> tuple[float, float, float]:
    """First, second, third yield derivatives of continuous bond price (Pucci Eq 5).

    D_y^k[P] = (-T)^k * e^{-Ty} + c * sum alpha_i * (-tau_i)^k * e^{-tau_i*y}

    Returns (D1, D2, D3). Signs alternate: D1 < 0, D2 > 0, D3 < 0.
    """
    T = time_to_maturity
    exp_T = math.exp(-T * y)
    D1 = (-T) * exp_T
    D2 = T**2 * exp_T
    D3 = (-T)**3 * exp_T
    for alpha, tau in zip(accrual_factors, times_to_coupon):
        q = -tau
        exp_tau = math.exp(-tau * y)
        D1 += coupon_rate * alpha * q * exp_tau
        D2 += coupon_rate * alpha * q**2 * exp_tau
        D3 += coupon_rate * alpha * q**3 * exp_tau
    return D1, D2, D3


def bond_irr(
    market_price: float,
    coupon_rate: float,
    accrual_factors: list[float],
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Internal rate of return: solve P(y) = market_price (simply-compounded).

    Newton from y0 = coupon_rate, bisect fallback.
    """
    def price_and_deriv(y):
        p = bond_price_from_yield(coupon_rate, accrual_factors, y)
        h = 1e-6
        dp = (bond_price_from_yield(coupon_rate, accrual_factors, y + h)
              - bond_price_from_yield(coupon_rate, accrual_factors, y - h)) / (2 * h)
        return p, dp

    y = coupon_rate if coupon_rate > 0 else 0.05
    for _ in range(max_iter):
        p, dp = price_and_deriv(y)
        if abs(p - market_price) < tol:
            return y
        if abs(dp) < 1e-15:
            break
        y -= (p - market_price) / dp
        y = max(-0.5, min(y, 2.0))

    lo, hi = -0.1, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = bond_price_from_yield(coupon_rate, accrual_factors, mid)
        if p > market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def bond_risk_factor(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """RiskFactor = -dP/dy (Pucci notation). Positive for positive yields."""
    h = 1e-6
    p_up = bond_price_from_yield(coupon_rate, accrual_factors, y + h)
    p_dn = bond_price_from_yield(coupon_rate, accrual_factors, y - h)
    return -(p_up - p_dn) / (2 * h)


def ytm_cmt_bridge(
    R_cmt: float,
    K: float,
    B: float,
    n: int,
) -> float:
    """YTM-CMT Taylor bridge (Pucci 2014, Eq 4).

    R^ytm ≈ R^cmt + (K - R^cmt) - R^cmt / (1 - (1+R^cmt)^{-n}) * (B - 1)

    Maps a CMT rate to an approximate YTM given coupon K and bond price B.
    Exact when B = 1 and K = R^cmt.
    """
    if abs(R_cmt) < 1e-15:
        # Degenerate: use large-n limit
        return K - (B - 1) / max(n, 1)

    discount_factor_n = (1 + R_cmt) ** (-n)
    annuity_factor = (1 - discount_factor_n) / R_cmt

    if abs(annuity_factor) < 1e-15:
        return R_cmt + (K - R_cmt)

    return R_cmt + (K - R_cmt) - R_cmt / annuity_factor * (B - 1)


def bond_dv01_from_yield(
    coupon_rate: float,
    accrual_factors: list[float],
    y: float,
) -> float:
    """Dollar value of a basis point from yield."""
    h = 0.00005
    return (bond_price_from_yield(coupon_rate, accrual_factors, y - h)
            - bond_price_from_yield(coupon_rate, accrual_factors, y + h))


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
        settlement_days: int = 0,
        ex_div_days: int = 0,
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
        self.calendar = calendar
        self.settlement_days = settlement_days
        self.ex_div_days = ex_div_days

        self.coupon_leg = FixedLeg(
            issue_date, maturity, coupon_rate, frequency,
            notional=face_value, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
        )

    def settlement_date(self, trade_date: date) -> date:
        """Compute settlement date from trade date using settlement_days.

        Uses calendar-aware business days if calendar is provided,
        otherwise calendar days.
        """
        if self.settlement_days == 0:
            return trade_date
        if self.calendar is not None:
            return self.calendar.add_business_days(trade_date, self.settlement_days)
        return trade_date + timedelta(days=self.settlement_days)

    def _future_cashflows(self, settlement: date) -> list:
        """Return only cashflows with payment_date > settlement."""
        return [cf for cf in self.coupon_leg.cashflows if cf.payment_date > settlement]

    def dirty_price(self, curve: DiscountCurve) -> float:
        """Full price: PV of remaining coupons + PV of principal, per 100 face."""
        settlement = curve.reference_date
        pv = sum(
            cf.amount * curve.df(cf.payment_date)
            for cf in self._future_cashflows(settlement)
        )
        if self.maturity > settlement:
            pv += self.face_value * curve.df(self.maturity)
        return pv / self.face_value * 100.0

    def accrued_interest(self, settlement: date) -> float:
        """
        Accrued interest per 100 face at the settlement date.

        Looks for the accrual period containing the settlement date.
        If ex_div_days > 0 and settlement is within ex_div_days of the
        next coupon, accrued is negative (buyer doesn't receive coupon).
        """
        for cf in self.coupon_leg.cashflows:
            if cf.accrual_start <= settlement < cf.accrual_end:
                # Check ex-dividend period
                if self.ex_div_days > 0:
                    ex_date = cf.accrual_end - timedelta(days=self.ex_div_days)
                    if settlement >= ex_date:
                        # In ex-div period: accrued is negative
                        yf_remaining = year_fraction(settlement, cf.accrual_end, self.day_count)
                        return -self.coupon_rate * yf_remaining * 100.0
                yf = year_fraction(cf.accrual_start, settlement, self.day_count)
                return self.coupon_rate * yf * 100.0

        # Settlement on or after last coupon: no accrued
        return 0.0

    def pv_ctx(self, ctx) -> float:
        """Price the bond from a PricingContext. Returns dirty price."""
        return self.dirty_price(ctx.discount_curve)

    def clean_price(self, curve: DiscountCurve, settlement: date | None = None) -> float:
        """
        Quoted price: dirty price minus accrued interest.

        If settlement is None, uses the curve's reference date.
        """
        if settlement is None:
            settlement = curve.reference_date
        return self.dirty_price(curve) - self.accrued_interest(settlement)

    def yield_to_maturity(self, market_price: float, settlement: date | None = None) -> float:
        """
        Yield to maturity: the constant rate that discounts all remaining
        cashflows to the given market (dirty) price.

        Uses bond-equivalent yield convention: compounding at the coupon frequency.
        Time is measured from settlement, not issue date.

        Args:
            market_price: dirty price per 100 face.
            settlement: date from which to discount. If None, uses issue_date
                for backward compat (new code should always pass settlement).
        """
        settle = settlement if settlement is not None else self.issue_date
        return brentq(
            lambda y: self._price_from_ytm(y, settle) - market_price,
            -0.10, 2.0,
        )

    def macaulay_duration(self, ytm: float, settlement: date | None = None) -> float:
        """
        Macaulay duration: weighted-average time to cashflows from settlement,
        where weights are PV of each cashflow / total PV.
        """
        settle = settlement if settlement is not None else self.issue_date
        freq = self.frequency.value
        periods_per_year = 12 / freq

        weighted_t = 0.0
        total_pv = 0.0
        for cf in self._future_cashflows(settle):
            t = year_fraction(settle, cf.payment_date, self.day_count)
            n = t * periods_per_year
            pv = cf.amount / (1.0 + ytm / periods_per_year) ** n
            weighted_t += t * pv
            total_pv += pv

        # Principal
        t_mat = year_fraction(settle, self.maturity, self.day_count)
        n_mat = t_mat * periods_per_year
        pv_prin = self.face_value / (1.0 + ytm / periods_per_year) ** n_mat
        weighted_t += t_mat * pv_prin
        total_pv += pv_prin

        if total_pv <= 0:
            return 0.0
        return weighted_t / total_pv

    def modified_duration(self, ytm: float, settlement: date | None = None) -> float:
        """Modified duration: Macaulay duration / (1 + ytm/freq)."""
        periods_per_year = 12 / self.frequency.value
        return self.macaulay_duration(ytm, settlement) / (1.0 + ytm / periods_per_year)

    def convexity(self, ytm: float, settlement: date | None = None) -> float:
        """
        Convexity: second-order price sensitivity to yield.

        C = (1/P) * sum(n_i * (n_i + 1) * PV_i) / (freq^2 * (1 + y/freq)^2)
        """
        settle = settlement if settlement is not None else self.issue_date
        freq = self.frequency.value
        periods_per_year = 12 / freq
        discount = 1.0 + ytm / periods_per_year

        weighted = 0.0
        total_pv = 0.0
        for cf in self._future_cashflows(settle):
            t = year_fraction(settle, cf.payment_date, self.day_count)
            n = t * periods_per_year
            pv = cf.amount / discount ** n
            weighted += n * (n + 1) * pv
            total_pv += pv

        t_mat = year_fraction(settle, self.maturity, self.day_count)
        n_mat = t_mat * periods_per_year
        pv_prin = self.face_value / discount ** n_mat
        weighted += n_mat * (n_mat + 1) * pv_prin
        total_pv += pv_prin

        if total_pv <= 0:
            return 0.0
        return weighted / (total_pv * periods_per_year ** 2 * discount ** 2)

    def dv01_yield(self, ytm: float, settlement: date | None = None) -> float:
        """Dollar value of a basis point: price change for a 1bp yield shift."""
        settle = settlement if settlement is not None else self.issue_date
        return self.modified_duration(ytm, settle) * self._price_from_ytm(ytm, settle) / 10000.0

    # ---- Yield-based interface (for T-Lock and analytics) ----

    def accrual_schedule(self, settlement: date | None = None
                         ) -> tuple[list[float], list[float], float]:
        """Extract accrual factors and times-to-coupon from the bond schedule.

        Returns (accrual_factors, times_to_coupon, time_to_maturity).
        """
        settle = settlement if settlement is not None else self.issue_date
        future_cfs = self._future_cashflows(settle)
        accrual_factors = [cf.year_frac for cf in future_cfs]
        times_to_coupon = [
            year_fraction(settle, cf.payment_date, self.day_count)
            for cf in future_cfs
        ]
        time_to_maturity = year_fraction(settle, self.maturity, self.day_count)
        return accrual_factors, times_to_coupon, time_to_maturity

    def price_from_yield_sc(self, y: float, settlement: date | None = None) -> float:
        """Simply-compounded bond price from yield (Pucci Eq 2)."""
        alphas, _, _ = self.accrual_schedule(settlement)
        return bond_price_from_yield(self.coupon_rate, alphas, y)

    def irr_sc(self, market_price: float, settlement: date | None = None) -> float:
        """IRR using simply-compounded convention."""
        alphas, _, _ = self.accrual_schedule(settlement)
        return bond_irr(market_price / 100.0 * self.face_value / self.face_value,
                        self.coupon_rate, alphas)

    def risk_factor_sc(self, y: float, settlement: date | None = None) -> float:
        """RiskFactor = -dP/dy (simply-compounded)."""
        alphas, _, _ = self.accrual_schedule(settlement)
        return bond_risk_factor(self.coupon_rate, alphas, y)

    def _price_from_ytm(self, ytm: float, settlement: date | None = None) -> float:
        """Dirty price per 100 face from a yield, discounting from settlement."""
        settle = settlement if settlement is not None else self.issue_date
        freq = self.frequency.value
        periods_per_year = 12 / freq
        pv = 0.0
        for cf in self._future_cashflows(settle):
            t = year_fraction(settle, cf.payment_date, self.day_count)
            n = t * periods_per_year
            pv += cf.amount / (1.0 + ytm / periods_per_year) ** n
        t_mat = year_fraction(settle, self.maturity, self.day_count)
        n_mat = t_mat * periods_per_year
        pv += self.face_value / (1.0 + ytm / periods_per_year) ** n_mat
        return pv / self.face_value * 100.0
