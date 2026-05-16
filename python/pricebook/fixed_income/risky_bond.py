"""
Risky bond pricing, asset swaps, and Z-spread.

Risky bond: cashflows discounted with survival-weighted discount factors.
Asset swap: bond + IRS to convert fixed to floating, ASW spread.
Z-spread: constant spread over risk-free curve that reprices the bond.

    rb = RiskyBond(start, end, coupon_rate=0.05)
    price = rb.dirty_price(discount_curve, survival_curve, recovery=0.4)
"""

from __future__ import annotations

import math
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve
from pricebook.schedule import Frequency, generate_schedule
from pricebook.solvers import brentq


class RiskyBond:
    """Fixed-rate bond with credit risk.

    PV = sum(coupon * df * survival) + principal * df * survival
         + recovery * sum(df_mid * default_prob_per_period)
    """

    def __init__(
        self,
        start: date,
        end: date,
        coupon_rate: float,
        notional: float = 100.0,
        frequency: Frequency = Frequency.SEMI_ANNUAL,
        day_count: DayCountConvention = DayCountConvention.THIRTY_360,
        recovery: float = 0.4,
    ):
        self.start = start
        self.end = end
        self.coupon_rate = coupon_rate
        self.notional = notional
        self.recovery = recovery
        self.day_count = day_count
        self.frequency = frequency
        self.schedule = generate_schedule(start, end, frequency)

    def _future_periods(self, settlement: date) -> list[tuple[date, date]]:
        """Return (period_start, period_end) pairs where period_end > settlement."""
        return [
            (self.schedule[i - 1], self.schedule[i])
            for i in range(1, len(self.schedule))
            if self.schedule[i] > settlement
        ]

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
        settlement: date | None = None,
    ) -> float:
        """Full price, accounting for default risk. Only future cashflows."""
        settle = settlement if settlement is not None else discount_curve.reference_date
        pv = 0.0

        for t_start, t_end in self._future_periods(settle):
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)

            # Coupon conditional on survival
            pv += self.notional * self.coupon_rate * yf * df * surv

            # Recovery on default in this period (ISDA: default at mid-period)
            surv_prev = survival_curve.survival(t_start)
            default_prob = surv_prev - surv
            from pricebook.day_count import date_from_year_fraction, year_fraction as _yf
            t_mid_yf = _yf(discount_curve.reference_date, t_start, discount_curve.day_count) + yf / 2
            t_mid_date = date_from_year_fraction(discount_curve.reference_date, t_mid_yf)
            df_mid = discount_curve.df(t_mid_date)
            pv += self.recovery * self.notional * default_prob * df_mid

        # Principal at maturity conditional on survival
        pv += self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)

        return pv

    def risk_free_price(
        self,
        discount_curve: DiscountCurve,
        settlement: date | None = None,
    ) -> float:
        """Price without credit risk (survival = 1 everywhere). Only future cashflows."""
        settle = settlement if settlement is not None else discount_curve.reference_date
        pv = 0.0
        for t_start, t_end in self._future_periods(settle):
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            pv += self.notional * self.coupon_rate * yf * df
        pv += self.notional * discount_curve.df(self.end)
        return pv

    def yield_to_maturity(
        self,
        market_price: float,
        settlement: date | None = None,
    ) -> float:
        """Yield to maturity for a risky bond (from settlement, ignoring default).

        Finds the flat yield that discounts remaining cashflows to market_price.
        """
        settle = settlement if settlement is not None else self.start
        periods_per_year = 12 / self.frequency.value

        def _price(y: float) -> float:
            pv = 0.0
            for t_start, t_end in self._future_periods(settle):
                yf = year_fraction(t_start, t_end, self.day_count)
                t = year_fraction(settle, t_end, self.day_count)
                n = t * periods_per_year
                pv += self.notional * self.coupon_rate * yf / (1 + y / periods_per_year) ** n
            t_mat = year_fraction(settle, self.end, self.day_count)
            n_mat = t_mat * periods_per_year
            pv += self.notional / (1 + y / periods_per_year) ** n_mat
            return pv

        return brentq(lambda y: _price(y) - market_price, -0.05, 2.0)

    def modified_duration(
        self,
        ytm: float,
        settlement: date | None = None,
    ) -> float:
        """Modified duration from settlement."""
        settle = settlement if settlement is not None else self.start
        periods_per_year = 12 / self.frequency.value
        discount = 1 + ytm / periods_per_year

        weighted_t = 0.0
        total_pv = 0.0
        for t_start, t_end in self._future_periods(settle):
            yf = year_fraction(t_start, t_end, self.day_count)
            t = year_fraction(settle, t_end, self.day_count)
            n = t * periods_per_year
            cf_pv = self.notional * self.coupon_rate * yf / discount ** n
            weighted_t += t * cf_pv
            total_pv += cf_pv

        t_mat = year_fraction(settle, self.end, self.day_count)
        n_mat = t_mat * periods_per_year
        prin_pv = self.notional / discount ** n_mat
        weighted_t += t_mat * prin_pv
        total_pv += prin_pv

        if total_pv <= 0:
            return 0.0
        mac = weighted_t / total_pv
        return mac / discount


def z_spread(
    bond: RiskyBond,
    market_price: float,
    discount_curve: DiscountCurve,
    settlement: date | None = None,
) -> float:
    """Z-spread: constant spread over risk-free curve that reprices the bond.

    Only future cashflows from settlement are included.
    """
    settle = settlement if settlement is not None else discount_curve.reference_date

    def objective(z: float) -> float:
        bumped = discount_curve.bumped(z)
        pv = 0.0
        for t_start, t_end in bond._future_periods(settle):
            yf = year_fraction(t_start, t_end, bond.day_count)
            pv += bond.notional * bond.coupon_rate * yf * bumped.df(t_end)
        pv += bond.notional * bumped.df(bond.end)
        return pv - market_price

    return brentq(objective, -0.05, 1.0)


def asset_swap_spread(
    bond: RiskyBond,
    market_price: float,
    discount_curve: DiscountCurve,
    settlement: date | None = None,
) -> float:
    """Asset swap spread: floating spread that equates bond PV to par.

    Only future cashflows from settlement are included.
    """
    settle = settlement if settlement is not None else discount_curve.reference_date

    annuity = 0.0
    for t_start, t_end in bond._future_periods(settle):
        yf = year_fraction(t_start, t_end, bond.day_count)
        annuity += yf * discount_curve.df(t_end)

    if annuity <= 0:
        return 0.0

    risk_free_pv = bond.risk_free_price(discount_curve, settle)
    return (risk_free_pv - market_price) / (bond.notional * annuity)
