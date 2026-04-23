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
from pricebook.day_count import date_from_year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.solvers import brentq


def _straight_bond_hw(
    hw: HullWhite,
    coupon_rate: float,
    maturity_years: float,
    n_steps: int = 100,
    notional: float = 100.0,
    coupon_frequency: float = 1.0,
) -> float:
    """Straight bond price on HW tree (no optionality)."""
    ref = hw.curve.reference_date
    pv = 0.0
    t_pay = coupon_frequency
    while t_pay <= maturity_years + 1e-10:
        d = date_from_year_fraction(ref, t_pay)
        df = hw.curve.df(d)
        pv += notional * coupon_rate * coupon_frequency * df
        t_pay += coupon_frequency

    d_mat = date_from_year_fraction(ref, maturity_years)
    pv += notional * hw.curve.df(d_mat)
    return pv


def _trinomial_backward(
    hw: HullWhite,
    maturity_years: float,
    coupon_rate: float,
    notional: float,
    n_steps: int,
    coupon_frequency: float,
    option_steps: set,
    option_func,
) -> float:
    """Generic trinomial tree backward induction for bonds with options.

    Args:
        option_steps: set of step indices where the option is exercisable.
        option_func: callable(values_array, exercise_price) -> modified values.
    """
    dt = maturity_years / n_steps
    a, sigma = hw.a, hw.sigma
    dr = sigma * math.sqrt(3.0 * dt)
    j_max = max(1, int(math.ceil(0.1835 / (a * dt))))
    n_nodes = 2 * j_max + 1
    mid = j_max

    coupon_steps = set()
    t_pay = coupon_frequency
    while t_pay <= maturity_years + 1e-10:
        coupon_steps.add(int(round(t_pay / dt)))
        t_pay += coupon_frequency

    r0 = hw._forward_rate(0.0)

    # Terminal value: principal + last coupon
    values = np.full(n_nodes, notional * (1 + coupon_rate * coupon_frequency))

    for step in range(n_steps - 1, -1, -1):
        new_values = np.zeros(n_nodes)

        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)

            # Trinomial transition probabilities
            p_up = (1.0 / 6.0) + (j * j * a * a * dt * dt - j * a * dt) / 6.0
            p_mid = 2.0 / 3.0 - j * j * a * a * dt * dt / 3.0
            p_down = (1.0 / 6.0) + (j * j * a * a * dt * dt + j * a * dt) / 6.0

            # Clamp and normalise
            p_up = max(0.0, min(1.0, p_up))
            p_mid = max(0.0, min(1.0, p_mid))
            p_down = max(0.0, min(1.0, p_down))
            p_total = p_up + p_mid + p_down
            if p_total > 0:
                p_up /= p_total
                p_mid /= p_total
                p_down /= p_total

            j_up = min(j + 1, j_max)
            j_down = max(j - 1, -j_max)

            cont = (p_up * values[j_up + mid]
                     + p_mid * values[j + mid]
                     + p_down * values[j_down + mid])
            new_values[idx] = cont * one_step_df

        # Add coupon if coupon date
        if (step + 1) in coupon_steps:
            new_values += notional * coupon_rate * coupon_frequency

        # Apply option constraint
        if (step + 1) in option_steps:
            new_values = option_func(new_values)

        values = new_values

    return float(values[mid])


def callable_bond_price(
    hw: HullWhite,
    coupon_rate: float,
    maturity_years: float,
    call_dates_years: list[float] | None = None,
    call_price: float = 100.0,
    n_steps: int = 100,
    notional: float = 100.0,
    coupon_frequency: float = 1.0,
) -> float:
    """Callable bond price via Hull-White trinomial tree.

    At each call date: bond_value = min(continuation, call_price).
    """
    dt = maturity_years / n_steps
    if call_dates_years is None:
        call_dates_years = []
        t = coupon_frequency
        while t <= maturity_years + 1e-10:
            call_dates_years.append(t)
            t += coupon_frequency

    call_steps = set(int(round(t / dt)) for t in call_dates_years)

    return _trinomial_backward(
        hw, maturity_years, coupon_rate, notional, n_steps, coupon_frequency,
        call_steps, lambda v: np.minimum(v, call_price),
    )


def puttable_bond_price(
    hw: HullWhite,
    coupon_rate: float,
    maturity_years: float,
    put_dates_years: list[float] | None = None,
    put_price: float = 100.0,
    n_steps: int = 100,
    notional: float = 100.0,
    coupon_frequency: float = 1.0,
) -> float:
    """Puttable bond price via Hull-White trinomial tree.

    At each put date: bond_value = max(continuation, put_price).
    """
    dt = maturity_years / n_steps
    if put_dates_years is None:
        put_dates_years = []
        t = coupon_frequency
        while t <= maturity_years + 1e-10:
            put_dates_years.append(t)
            t += coupon_frequency

    put_steps = set(int(round(t / dt)) for t in put_dates_years)

    return _trinomial_backward(
        hw, maturity_years, coupon_rate, notional, n_steps, coupon_frequency,
        put_steps, lambda v: np.maximum(v, put_price),
    )


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
    coupon_frequency: float = 1.0,
) -> float:
    """Option-adjusted spread.

    The constant spread to the risk-free curve such that the model price
    (with embedded option) equals the market price.

    OAS = 0 means the bond is fairly priced relative to the model.
    OAS > 0 means the bond is cheap (higher yield than the model implies).
    """
    def objective(spread: float) -> float:
        bumped_curve = hw.curve.bumped(spread)
        hw_bumped = HullWhite(a=hw.a, sigma=hw.sigma, curve=bumped_curve)
        if is_callable:
            model_price = callable_bond_price(
                hw_bumped, coupon_rate, maturity_years, call_put_dates,
                exercise_price, n_steps, notional, coupon_frequency,
            )
        else:
            model_price = puttable_bond_price(
                hw_bumped, coupon_rate, maturity_years, call_put_dates,
                exercise_price, n_steps, notional, coupon_frequency,
            )
        return model_price - market_price

    return brentq(objective, -0.05, 0.20)
