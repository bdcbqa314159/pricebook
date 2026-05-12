"""Callable/putable bond desk analytics: OAS, option value, effective duration.

Integrates callable_bond.py (Hull-White tree) into the bond trading desk.

    from pricebook.callable_bond_desk import (
        callable_bond_analytics, CallableBondAnalytics,
    )

References:
    Fabozzi (2007). Fixed Income Analysis, Ch. 18 (Bonds with Embedded Options).
    Tuckman & Serrat (2012). Ch. 7 (Empirical and Model-Based Approaches to
    Pricing Bonds with Embedded Options).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.discount_curve import DiscountCurve
from pricebook.hull_white import HullWhite
from pricebook.callable_bond import callable_bond_price, puttable_bond_price, oas


@dataclass
class CallableBondAnalytics:
    """Full analytics for a callable or puttable bond."""
    model_price: float
    straight_price: float
    option_value: float          # straight - callable (positive for callable)
    oas_bps: float
    effective_duration: float
    effective_convexity: float
    option_type: str             # "callable" or "puttable"

    def to_dict(self) -> dict:
        return {
            "model_price": self.model_price,
            "straight_price": self.straight_price,
            "option_value": self.option_value,
            "oas_bps": self.oas_bps,
            "effective_duration": self.effective_duration,
            "effective_convexity": self.effective_convexity,
            "option_type": self.option_type,
        }


def callable_bond_analytics(
    curve: DiscountCurve,
    market_price: float,
    coupon_rate: float,
    maturity_years: float,
    hw_a: float = 0.05,
    hw_sigma: float = 0.01,
    is_callable: bool = True,
    call_put_dates: list[float] | None = None,
    exercise_price: float = 100.0,
    n_steps: int = 100,
) -> CallableBondAnalytics:
    """Full analytics for a callable or puttable bond.

    Computes:
    - Model price (Hull-White tree with embedded option)
    - Straight bond price (no option)
    - Option value = straight - callable (or puttable - straight)
    - OAS (spread to make model price = market price)
    - Effective duration (bump-and-reprice with option exercise)
    - Effective convexity

    Args:
        curve: risk-free discount curve.
        market_price: observed market price.
        coupon_rate: annual coupon rate.
        maturity_years: years to maturity.
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White short rate vol.
        is_callable: True for callable, False for puttable.
        call_put_dates: exercise dates (years). None = all coupon dates.
        exercise_price: call/put price (per 100 face).
    """
    hw = HullWhite(a=hw_a, sigma=hw_sigma, curve=curve)

    # Model price with option
    if is_callable:
        model_price = callable_bond_price(hw, coupon_rate, maturity_years,
                                           call_put_dates, exercise_price, n_steps)
    else:
        model_price = puttable_bond_price(hw, coupon_rate, maturity_years,
                                           call_put_dates, exercise_price, n_steps)

    # Straight bond price (no option) — same tree, no exercise
    straight = callable_bond_price(
        hw, coupon_rate, maturity_years, call_dates_years=[], n_steps=n_steps)

    # Option value
    if is_callable:
        option_val = straight - model_price  # callable ≤ straight
    else:
        option_val = model_price - straight  # puttable ≥ straight

    # OAS
    oas_val = oas(hw, market_price, coupon_rate, maturity_years,
                   is_callable, call_put_dates, exercise_price, n_steps)

    # Effective duration: bump curve ±h, reprice with option
    h = 0.0001
    curve_up = curve.bumped(h)
    curve_dn = curve.bumped(-h)
    hw_up = HullWhite(a=hw_a, sigma=hw_sigma, curve=curve_up)
    hw_dn = HullWhite(a=hw_a, sigma=hw_sigma, curve=curve_dn)

    if is_callable:
        pv_up = callable_bond_price(hw_up, coupon_rate, maturity_years,
                                     call_put_dates, exercise_price, n_steps)
        pv_dn = callable_bond_price(hw_dn, coupon_rate, maturity_years,
                                     call_put_dates, exercise_price, n_steps)
    else:
        pv_up = puttable_bond_price(hw_up, coupon_rate, maturity_years,
                                     call_put_dates, exercise_price, n_steps)
        pv_dn = puttable_bond_price(hw_dn, coupon_rate, maturity_years,
                                     call_put_dates, exercise_price, n_steps)

    eff_dur = -(pv_up - pv_dn) / (2 * h * model_price) if model_price > 0 else 0.0
    eff_conv = (pv_up + pv_dn - 2 * model_price) / (h**2 * model_price) if model_price > 0 else 0.0

    return CallableBondAnalytics(
        model_price=float(model_price),
        straight_price=float(straight),
        option_value=float(option_val),
        oas_bps=float(oas_val * 1e4),
        effective_duration=float(eff_dur),
        effective_convexity=float(eff_conv),
        option_type="callable" if is_callable else "puttable",
    )
