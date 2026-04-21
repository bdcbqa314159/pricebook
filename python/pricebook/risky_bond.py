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
         + recovery * sum(df * default_prob_per_period)
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
        self.schedule = generate_schedule(start, end, frequency)

    def dirty_price(
        self,
        discount_curve: DiscountCurve,
        survival_curve: SurvivalCurve,
    ) -> float:
        """Full price per 100 face, accounting for default risk."""
        pv = 0.0

        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            surv = survival_curve.survival(t_end)

            # Coupon conditional on survival
            pv += self.notional * self.coupon_rate * yf * df * surv

            # Recovery on default in this period (ISDA: default at mid-period)
            surv_prev = survival_curve.survival(t_start)
            default_prob = surv_prev - surv
            df_mid = (discount_curve.df(t_start) + df) / 2.0
            pv += self.recovery * self.notional * default_prob * df_mid

        # Principal at maturity conditional on survival
        pv += self.notional * discount_curve.df(self.end) * survival_curve.survival(self.end)

        return pv

    def risk_free_price(self, discount_curve: DiscountCurve) -> float:
        """Price without credit risk (survival = 1 everywhere)."""
        pv = 0.0
        for i in range(1, len(self.schedule)):
            t_start = self.schedule[i - 1]
            t_end = self.schedule[i]
            yf = year_fraction(t_start, t_end, self.day_count)
            df = discount_curve.df(t_end)
            pv += self.notional * self.coupon_rate * yf * df
        pv += self.notional * discount_curve.df(self.end)
        return pv


def z_spread(
    bond: RiskyBond,
    market_price: float,
    discount_curve: DiscountCurve,
) -> float:
    """Z-spread: constant spread over risk-free curve that reprices the bond.

    Solves for z such that PV(curve + z) = market_price.
    """
    def objective(z: float) -> float:
        bumped = discount_curve.bumped(z)
        # Price as risk-free with the bumped curve
        pv = 0.0
        for i in range(1, len(bond.schedule)):
            t_start = bond.schedule[i - 1]
            t_end = bond.schedule[i]
            yf = year_fraction(t_start, t_end, bond.day_count)
            pv += bond.notional * bond.coupon_rate * yf * bumped.df(t_end)
        pv += bond.notional * bumped.df(bond.end)
        return pv - market_price

    return brentq(objective, -0.05, 1.0)


def asset_swap_spread(
    bond: RiskyBond,
    market_price: float,
    discount_curve: DiscountCurve,
) -> float:
    """Asset swap spread: floating spread that equates bond PV to par.

    ASW spread ≈ (par - market_price) / annuity + coupon - par_swap_rate

    Simplified: solve for spread s such that
        market_price + PV(floating + s) = PV(fixed coupons) + 100
    """
    # Annuity: sum of year_frac * df
    annuity = 0.0
    for i in range(1, len(bond.schedule)):
        t_start = bond.schedule[i - 1]
        t_end = bond.schedule[i]
        yf = year_fraction(t_start, t_end, bond.day_count)
        annuity += yf * discount_curve.df(t_end)

    if annuity == 0:
        return 0.0

    # PV of fixed coupons + principal at par
    risk_free_pv = bond.risk_free_price(discount_curve)

    # ASW spread = (risk_free_price - market_price) / (notional * annuity)
    return (risk_free_pv - market_price) / (bond.notional * annuity)
