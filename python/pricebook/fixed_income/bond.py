"""Fixed-rate bond.

Includes yield-based pricing (simply-compounded, continuous Hull-form)
for Treasury Lock and other yield-curve analytics (Pucci 2019).
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.discount_curve import DiscountCurve
from pricebook.fixed_income.fixed_leg import FixedLeg
from pricebook.core.schedule import Frequency, StubType
from pricebook.core.calendar import Calendar, BusinessDayConvention
from pricebook.core.solvers import brentq


# ---- Yield-based bond pricing (re-exported from bond_yield.py) ----
# Canonical location: pricebook.bond_yield
# Re-exported here for backward compatibility.

from pricebook.fixed_income.bond_yield import (  # noqa: F401
    bond_price_from_yield,
    bond_price_from_yield_stub,
    bond_price_continuous,
    bond_yield_derivatives,
    bond_irr,
    bond_risk_factor,
    ytm_cmt_bridge,
    bond_dv01_from_yield,
)


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
        face_value: float | list[float] = 100.0,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        calendar: Calendar | None = None,
        convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
        stub: StubType = StubType.SHORT_FRONT,
        eom: bool = True,
        settlement_days: int = 0,
        ex_div_days: int = 0,
    ):
        if issue_date >= maturity:
            raise ValueError(f"issue_date ({issue_date}) must be before maturity ({maturity})")

        self.issue_date = issue_date
        self.maturity = maturity
        self.coupon_rate = coupon_rate
        self.frequency = frequency
        self.day_count = day_count
        self.calendar = calendar
        self.convention = convention
        self.stub = stub
        self.eom = eom
        self.settlement_days = settlement_days
        self.ex_div_days = ex_div_days

        self.coupon_leg = FixedLeg(
            issue_date, maturity, coupon_rate, frequency,
            notional=face_value, day_count=day_count,
            calendar=calendar, convention=convention, stub=stub, eom=eom,
        )
        self.face_value = self.coupon_leg.notional  # always float (first period)

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
        """Full price: PV of remaining coupons + PV of principal, per 100 face.

        Fix T4-BOND1: when settlement falls in the ex-dividend window of an
        upcoming coupon, the buyer does NOT receive that coupon (it goes to
        the record-date holder).  Pre-fix `dirty_price` included every
        coupon with ``payment_date > settlement`` while `accrued_interest`
        correctly returned a NEGATIVE value in the ex-div window.  Result:
        ``clean_price = dirty - (-accrued) = dirty + |accrued|`` was about
        one full coupon TOO HIGH right after the ex-div boundary — a
        discontinuity of size ≈ coupon at the ex-div date, which by market
        convention should be barely perceptible (clean price moves only by
        a few days of accrued).

        Correct: in ex-div, exclude the upcoming coupon's cashflow from
        the dirty-price PV.  Then ``clean = dirty - accrued`` makes
        clean continuous across the ex-div boundary.
        """
        settlement = curve.reference_date
        pv = 0.0
        for cf in self._future_cashflows(settlement):
            if self.ex_div_days > 0:
                # Skip the upcoming coupon if settlement is on/after its ex-div date.
                ex_date = cf.accrual_end - timedelta(days=self.ex_div_days)
                if cf.accrual_start <= settlement < cf.accrual_end and settlement >= ex_date:
                    continue
            pv += cf.amount * curve.df(cf.payment_date)
        if self.maturity > settlement:
            # Principal at maturity uses last-period face value (sinking fund support)
            final_face = self.coupon_leg.notional_schedule[-1]
            pv += final_face * curve.df(self.maturity)
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
            t = self._ytm_time_to(settle, cf.payment_date)
            n = t * periods_per_year
            pv = cf.amount / (1.0 + ytm / periods_per_year) ** n
            weighted_t += t * pv
            total_pv += pv

        # Principal — Fix T2.14: use last-period notional (sinking-fund correct).
        t_mat = self._ytm_time_to(settle, self.maturity)
        n_mat = t_mat * periods_per_year
        pv_prin = self.coupon_leg.notional_schedule[-1] / (1.0 + ytm / periods_per_year) ** n_mat
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
            t = self._ytm_time_to(settle, cf.payment_date)
            n = t * periods_per_year
            pv = cf.amount / discount ** n
            weighted += n * (n + 1) * pv
            total_pv += pv

        # Fix T2.14: use last-period notional for redemption.
        t_mat = self._ytm_time_to(settle, self.maturity)
        n_mat = t_mat * periods_per_year
        pv_prin = self.coupon_leg.notional_schedule[-1] / discount ** n_mat
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
            self._ytm_time_to(settle, cf.payment_date) for cf in future_cfs
        ]
        time_to_maturity = self._ytm_time_to(settle, self.maturity)
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

    def _ytm_time_to(self, settle: date, target: date) -> float:
        """Time from `settle` to `target` for YTM discounting under this
        bond's day_count convention.

        For ACT/ACT ICMA the time is *not* the simple (target − settle) day
        count divided by 365 — that would treat the coupon-period structure
        as irrelevant. The correct value (per ICMA 251.1 / 251.2) is the
        number of coupon periods between `settle` and `target` divided by
        `coupons_per_year`, with a mid-period stub fraction at the start
        when `settle` doesn't coincide with a coupon date.

        For other day counts, falls back to `year_fraction(settle, target,
        day_count)` — the legacy multi-period path is exact for ACT/365F,
        ACT/360, 30/360, etc.

        Fix A.1 B1 Slice 4.
        """
        if self.day_count != DayCountConvention.ACT_ACT_ICMA:
            return year_fraction(settle, target, self.day_count)

        coupons_per_year = 12 // self.frequency.value if self.frequency.value > 0 else None
        if coupons_per_year is None:
            return year_fraction(settle, target, self.day_count)

        # Schedule dates: issue_date, then each accrual_end.
        # accrual_end of the last coupon == maturity, so this also covers
        # principal-at-maturity correctly.
        coupon_dates = [self.issue_date] + [
            cf.accrual_end for cf in self.coupon_leg.cashflows
        ]

        try:
            target_idx = coupon_dates.index(target)
        except ValueError:
            return year_fraction(settle, target, self.day_count)

        # Settle on a coupon boundary — count periods exactly.
        if settle in coupon_dates:
            settle_idx = coupon_dates.index(settle)
            n_periods = target_idx - settle_idx
            return n_periods / coupons_per_year

        # Mid-period: find the containing period and use the ICMA stub.
        for i in range(len(coupon_dates) - 1):
            if coupon_dates[i] < settle < coupon_dates[i + 1]:
                period_days = (coupon_dates[i + 1] - coupon_dates[i]).days
                days_to_next = (coupon_dates[i + 1] - settle).days
                stub_fraction = days_to_next / period_days
                n_full_periods = target_idx - (i + 1)
                return (stub_fraction + n_full_periods) / coupons_per_year

        # Settle before the first coupon date or after the last — fall back.
        return year_fraction(settle, target, self.day_count)

    def _price_from_ytm(self, ytm: float, settlement: date | None = None) -> float:
        """Dirty price per 100 face from a yield, discounting from settlement.

        Fix T2.14: pre-fix the redemption term used ``self.face_value`` which
        is the FIRST period's notional from the (possibly variable) notional
        schedule.  For a sinking-fund bond the actual redemption is the LAST
        period's notional — ``dirty_price`` (curve-based) already used this
        correctly (``notional_schedule[-1]``), so the YTM-based price and the
        curve-based price silently disagreed for amortising bonds.  Use the
        last-period notional consistently.
        """
        settle = settlement if settlement is not None else self.issue_date
        freq = self.frequency.value
        periods_per_year = 12 / freq
        pv = 0.0
        for cf in self._future_cashflows(settle):
            t = self._ytm_time_to(settle, cf.payment_date)
            n = t * periods_per_year
            pv += cf.amount / (1.0 + ytm / periods_per_year) ** n
        t_mat = self._ytm_time_to(settle, self.maturity)
        n_mat = t_mat * periods_per_year
        redemption = self.coupon_leg.notional_schedule[-1]
        pv += redemption / (1.0 + ytm / periods_per_year) ** n_mat
        return pv / self.face_value * 100.0

    @classmethod
    def treasury_note(
        cls,
        issue_date: date,
        maturity: date,
        coupon_rate: float,
        face_value: float = 100.0,
    ) -> "FixedRateBond":
        """US Treasury note/bond with correct market conventions.

        - ACT/ACT ICMA day count (not 30/360)
        - Semi-annual coupons
        - T+1 settlement
        - No ex-dividend period

        Args:
            issue_date: original issue or dated date.
            maturity: maturity date.
            coupon_rate: annual coupon (e.g. 0.04 for 4%).
        """
        return cls(
            issue_date=issue_date,
            maturity=maturity,
            coupon_rate=coupon_rate,
            frequency=Frequency.SEMI_ANNUAL,
            face_value=face_value,
            day_count=DayCountConvention.ACT_ACT_ICMA,
            settlement_days=1,
        )


@classmethod
def _bond_from_convention(cls, conv, issue_date, maturity, coupon_rate, face_value=100.0):
    """Create FixedRateBond from a convention object (SovereignConventions or similar).

    The convention provides: frequency, day_count, calendar_currency, settlement_days, ex_div_days.
    The caller provides: issue_date, maturity, coupon_rate, face_value.
    """
    from pricebook.core.calendar import get_calendar, BusinessDayConvention
    cal = get_calendar(conv.calendar_currency) if hasattr(conv, 'calendar_currency') else None
    return cls(
        issue_date=issue_date, maturity=maturity, coupon_rate=coupon_rate,
        frequency=conv.frequency, face_value=face_value, day_count=conv.day_count,
        calendar=cal, convention=BusinessDayConvention.MODIFIED_FOLLOWING,
        settlement_days=getattr(conv, 'settlement_days', 0),
        ex_div_days=getattr(conv, 'ex_div_days', 0),
    )

FixedRateBond.from_convention = _bond_from_convention

from pricebook.core.serialisable import serialisable as _serialisable
_serialisable("bond", ["issue_date", "maturity", "coupon_rate", "frequency", "face_value", "day_count", "settlement_days"])(FixedRateBond)
