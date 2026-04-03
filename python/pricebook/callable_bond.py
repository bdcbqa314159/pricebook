"""
Callable and puttable bonds priced on the Hull-White rate tree.

Callable: issuer can call (buy back) at par on coupon dates.
    Price = min(continuation, call_price) at each call date.

Puttable: investor can put (sell back) at par on put dates.
    Price = max(continuation, put_price) at each put date.

OAS: Option-Adjusted Spread — the spread to the risk-free curve
    that reprices the bond with its embedded optionality.

    from pricebook.callable_bond import callable_bond_price, puttable_bond_price, oas

    price = callable_bond_price(coupon=0.05, maturity=10, call_dates=[5,6,7,8,9,10],
                                hw=hull_white_model)
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from pricebook.hull_white import HullWhite
from pricebook.discount_curve import DiscountCurve
from pricebook.solvers import brentq


def _straight_bond_hw(
    hw: HullWhite,
    coupon_rate: float,
    maturity_years: float,
    n_steps: int = 100,
    notional: float = 100.0,
) -> float:
    """Straight bond price on HW tree (no optionality)."""
    # Simple: PV of coupons + principal using model discount factors
    ref = hw.curve.reference_date
    pv = 0.0
    n_coupons = max(1, int(maturity_years))
    for i in range(1, n_coupons + 1):
        t = min(i, maturity_years)
        d = date.fromordinal(ref.toordinal() + int(t * 365))
        df = hw.curve.df(d)
        pv += notional * coupon_rate * df

    d_mat = date.fromordinal(ref.toordinal() + int(maturity_years * 365))
    pv += notional * hw.curve.df(d_mat)
    return pv


def callable_bond_price(
    hw: HullWhite,
    coupon_rate: float,
    maturity_years: float,
    call_dates_years: list[float] | None = None,
    call_price: float = 100.0,
    n_steps: int = 100,
    notional: float = 100.0,
) -> float:
    """Callable bond price via Hull-White tree.

    At each call date: bond_value = min(continuation, call_price).

    Args:
        call_dates_years: list of times (in years) when the bond is callable.
            Default: callable at every coupon date from year 1 to maturity.
        call_price: price at which issuer can call (per 100 face). Default: par.
    """
    dt = maturity_years / n_steps
    a, sigma = hw.a, hw.sigma
    dr = sigma * math.sqrt(3.0 * dt)
    j_max = int(math.ceil(0.1835 / (a * dt)))
    n_nodes = 2 * j_max + 1
    mid = j_max

    if call_dates_years is None:
        call_dates_years = list(range(1, int(maturity_years) + 1))

    call_steps = set(int(round(t / dt)) for t in call_dates_years)
    coupon_steps = set(int(round(t / dt)) for t in range(1, int(maturity_years) + 1))

    # Terminal value: principal + last coupon
    values = np.full(n_nodes, notional * (1 + coupon_rate))

    # Evolve state prices backward
    Q, dr_val, j_max_val, r0 = hw._evolve_state_prices(maturity_years, n_steps)

    # Backward induction on the tree
    # Simplified: use the analytical ZCB for one-step discounting
    ref = hw.curve.reference_date
    for step in range(n_steps - 1, -1, -1):
        t = step * dt
        new_values = np.zeros(n_nodes)

        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)

            # Discount continuation
            new_values[idx] = values[idx] * one_step_df

        # Add coupon if coupon date
        if (step + 1) in coupon_steps:
            new_values += notional * coupon_rate

        # Apply call constraint
        if (step + 1) in call_steps:
            new_values = np.minimum(new_values, call_price)

        values = new_values

    return float(values[mid])


def puttable_bond_price(
    hw: HullWhite,
    coupon_rate: float,
    maturity_years: float,
    put_dates_years: list[float] | None = None,
    put_price: float = 100.0,
    n_steps: int = 100,
    notional: float = 100.0,
) -> float:
    """Puttable bond price via Hull-White tree.

    At each put date: bond_value = max(continuation, put_price).
    """
    dt = maturity_years / n_steps
    a, sigma = hw.a, hw.sigma
    dr = sigma * math.sqrt(3.0 * dt)
    j_max = int(math.ceil(0.1835 / (a * dt)))
    n_nodes = 2 * j_max + 1
    mid = j_max

    if put_dates_years is None:
        put_dates_years = list(range(1, int(maturity_years) + 1))

    put_steps = set(int(round(t / dt)) for t in put_dates_years)
    coupon_steps = set(int(round(t / dt)) for t in range(1, int(maturity_years) + 1))

    values = np.full(n_nodes, notional * (1 + coupon_rate))

    Q, dr_val, j_max_val, r0 = hw._evolve_state_prices(maturity_years, n_steps)

    ref = hw.curve.reference_date
    for step in range(n_steps - 1, -1, -1):
        new_values = np.zeros(n_nodes)

        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)
            new_values[idx] = values[idx] * one_step_df

        if (step + 1) in coupon_steps:
            new_values += notional * coupon_rate

        if (step + 1) in put_steps:
            new_values = np.maximum(new_values, put_price)

        values = new_values

    return float(values[mid])


def oas(
    hw: HullWhite,
    market_price: float,
    coupon_rate: float,
    maturity_years: float,
    is_callable: bool = True,
    call_put_dates: list[float] | None = None,
    exercise_price: float = 100.0,
    n_steps: int = 100,
    notional: float = 100.0,
) -> float:
    """Option-Adjusted Spread.

    The constant spread added to the risk-free curve that reprices the bond.
    """
    def objective(spread: float) -> float:
        bumped = hw.curve.bumped(spread)
        hw_bumped = HullWhite(hw.a, hw.sigma, bumped)
        if is_callable:
            model_price = callable_bond_price(
                hw_bumped, coupon_rate, maturity_years,
                call_put_dates, exercise_price, n_steps, notional,
            )
        else:
            model_price = puttable_bond_price(
                hw_bumped, coupon_rate, maturity_years,
                call_put_dates, exercise_price, n_steps, notional,
            )
        return model_price - market_price

    return brentq(objective, -0.10, 0.10)
