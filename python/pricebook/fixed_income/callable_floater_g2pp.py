"""
Callable and puttable floating-rate notes (FRN) under the G2++ two-factor model.

The 1-factor callable FRN (Hull-White trinomial tree) is in callable_floater.py.
This module extends the pricing to the G2++ model, where two correlated
Ornstein-Uhlenbeck factors (x, y) drive the short rate:

    r(t) = x(t) + y(t) + phi(t)
    dx = -a*x*dt + sigma1*dW1
    dy = -b*y*dt + sigma2*dW2
    dW1 dW2 = rho dt

The floating coupon at each node is proportional to the prevailing short rate::

    coupon(x, y, t) = notional * (r(x, y, t) + spread) * period_length

Backward induction on the product 2D trinomial lattice prices the embedded
American-style call or put option.  The two-factor premium field captures
the difference between the G2++ price and the corresponding 1-factor result
(using only the x-factor, i.e. sigma2 = 0).

Usage::

    from pricebook.fixed_income.callable_floater_g2pp import (
        callable_frn_g2pp, puttable_frn_g2pp,
    )

    result = callable_frn_g2pp(g2pp, maturity_years=5, spread=0.005,
                               call_dates_years=[2, 3, 4])

References:
    Brigo, D. & Mercurio, F., *Interest Rate Models — Theory and Practice*,
    2nd ed., Ch. 4, Springer, 2006.
    Fabozzi, F., *Bond Markets, Analysis and Strategies*, Ch. 17, 2012.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.vasicek import G2PlusPlus
from pricebook.models.g2pp_tree import G2PPTree
from pricebook.core.day_count import date_from_year_fraction


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CallableFloaterG2PPResult:
    """Result from G2++ callable/puttable FRN pricing.

    Attributes:
        price: model price of the callable or puttable FRN.
        straight_price: price of the equivalent straight (non-optioned) FRN.
        option_value: |price - straight_price|; embedded option value.
        hw1f_price: equivalent 1-factor Hull-White price using only the
            x-factor (sigma2=0), for benchmarking.
        two_factor_premium: price - hw1f_price; captures the extra value
            contributed by the second factor.
    """
    price: float
    straight_price: float
    option_value: float
    hw1f_price: float
    two_factor_premium: float

    def to_dict(self) -> dict:
        return dict(vars(self))


# ---------------------------------------------------------------------------
# Internal: phi(t) deterministic shift for short rate at tree node
# ---------------------------------------------------------------------------


def _phi_at(g2pp: G2PlusPlus, t: float) -> float:
    """G2++ deterministic shift phi(t) = f_mkt(0,t) + variance correction.

    Fix T4-G2T3: same finite-difference defect as G2PPTree._fwd_rate
    (T4-G2T1) — eps=1e-5 was destroyed by date_from_year_fraction's
    day rounding, alternating fwd ≈ 0 / fwd ≈ 137·r across the time
    grid.  Use the curve's stable one-day-step instantaneous_forward.
    """
    a, b_ = g2pp.a, g2pp.b
    s1, s2, rho = g2pp.sigma1, g2pp.sigma2, g2pp.rho
    f_mkt = g2pp.curve.instantaneous_forward(t)

    ea = (1 - math.exp(-a * t)) if a > 0 else t
    eb = (1 - math.exp(-b_ * t)) if b_ > 0 else t
    corr = (s1**2 / (2 * a**2) * ea**2
            + s2**2 / (2 * b_**2) * eb**2
            + rho * s1 * s2 / (a * b_) * ea * eb)
    return f_mkt + corr


# ---------------------------------------------------------------------------
# Internal: straight FRN price on 2D G2++ tree
# ---------------------------------------------------------------------------


def _straight_frn_g2pp_tree(
    g2pp: G2PlusPlus,
    maturity_years: float,
    spread: float,
    frequency: int,
    notional: float,
    n_steps: int,
) -> float:
    """Straight FRN price on the G2++ 2D tree (no embedded option).

    The floating coupon at each node is notional * (r_node + spread) * period.
    For a flat curve the straight FRN should price close to par.
    """
    tree = G2PPTree(g2pp, maturity_years, n_steps)
    dt = tree.dt
    period = 1.0 / frequency

    coupon_steps: set[int] = set()
    t_pay = period
    while t_pay <= maturity_years + 1e-10:
        coupon_steps.add(int(round(t_pay / dt)))
        t_pay += period

    # Terminal: notional + last floating coupon at each node
    terminal = np.empty((tree.n_x, tree.n_y))
    phi_T = _phi_at(g2pp, maturity_years)
    for xi in range(tree.n_x):
        x_val = tree.x_nodes[xi]
        for yi in range(tree.n_y):
            y_val = tree.y_nodes[yi]
            r_node = x_val + y_val + phi_T
            coupon = notional * max(r_node + spread, 0.0) * period
            terminal[xi, yi] = notional + coupon

    # Pre-cache phi values for each step
    phi_cache = [_phi_at(g2pp, step * dt) for step in range(n_steps + 1)]

    def _frn_node_func(t_idx: int, xi: int, yi: int, cont: float) -> float:
        arriving_step = t_idx + 1
        if arriving_step in coupon_steps and arriving_step < n_steps:
            # Coupon at arriving time t_idx+1
            t_arr = arriving_step * dt
            x_val = tree.x_nodes[xi]
            y_val = tree.y_nodes[yi]
            r_node = x_val + y_val + phi_cache[arriving_step]
            coupon = notional * max(r_node + spread, 0.0) * period
            return cont + coupon
        return cont

    return tree.backward_induction(terminal_values=terminal, option_func=_frn_node_func)


# ---------------------------------------------------------------------------
# Internal: 2D tree backward induction for callable/puttable FRN
# ---------------------------------------------------------------------------


def _g2pp_frn_tree(
    g2pp: G2PlusPlus,
    maturity_years: float,
    spread: float,
    option_dates_years: list[float],
    exercise_price: float,
    n_steps: int,
    notional: float,
    frequency: int,
    is_callable: bool,
) -> float:
    """G2++ 2D tree pricing for callable or puttable floating-rate note.

    At each coupon date the floating coupon r(x,y,t) + spread is added.
    At each option date the call (min) or put (max) constraint is applied.

    Args:
        g2pp: calibrated G2PlusPlus model.
        maturity_years: FRN tenor in years.
        spread: fixed spread over the floating index (decimal).
        option_dates_years: list of exercise dates as year fractions.
        exercise_price: call/put strike (typically 100 per notional).
        n_steps: tree time steps.
        notional: face value.
        frequency: coupon periods per year.
        is_callable: True for callable (min), False for puttable (max).

    Returns:
        FRN price at the root node.
    """
    tree = G2PPTree(g2pp, maturity_years, n_steps)
    dt = tree.dt
    period = 1.0 / frequency

    coupon_steps: set[int] = set()
    t_pay = period
    while t_pay <= maturity_years + 1e-10:
        coupon_steps.add(int(round(t_pay / dt)))
        t_pay += period

    option_steps: set[int] = set(
        int(round(t / dt))
        for t in option_dates_years
        if 0 < t <= maturity_years + 1e-10
    )

    # Terminal: notional + last coupon node-by-node
    terminal = np.empty((tree.n_x, tree.n_y))
    phi_T = _phi_at(g2pp, maturity_years)
    for xi in range(tree.n_x):
        x_val = tree.x_nodes[xi]
        for yi in range(tree.n_y):
            y_val = tree.y_nodes[yi]
            r_node = x_val + y_val + phi_T
            coupon = notional * max(r_node + spread, 0.0) * period
            terminal[xi, yi] = notional + coupon

    # Pre-cache phi for efficiency
    phi_cache = [_phi_at(g2pp, step * dt) for step in range(n_steps + 1)]

    def _node_func(t_idx: int, xi: int, yi: int, cont: float) -> float:
        arriving_step = t_idx + 1
        value = cont

        # Add coupon at coupon dates (skip last: already in terminal)
        if arriving_step in coupon_steps and arriving_step < n_steps:
            x_val = tree.x_nodes[xi]
            y_val = tree.y_nodes[yi]
            r_node = x_val + y_val + phi_cache[arriving_step]
            coupon = notional * max(r_node + spread, 0.0) * period
            value += coupon

        # Apply exercise constraint
        if arriving_step in option_steps:
            if is_callable:
                value = min(value, exercise_price)
            else:
                value = max(value, exercise_price)

        return value

    return tree.backward_induction(terminal_values=terminal, option_func=_node_func)


# ---------------------------------------------------------------------------
# Internal: 1-factor benchmark (sigma2 = 0)
# ---------------------------------------------------------------------------


def _hw1f_frn(
    g2pp: G2PlusPlus,
    maturity_years: float,
    spread: float,
    option_dates_years: list[float],
    exercise_price: float,
    n_steps: int,
    notional: float,
    frequency: int,
    is_callable: bool,
) -> float:
    """Price a callable/puttable FRN with sigma2=0 (1-factor benchmark)."""
    from datetime import date
    from pricebook.fixed_income.callable_floater import callable_frn, puttable_frn

    ref = g2pp.curve.reference_date

    # Approximate r0 from the initial forward rate.  Fix T4-G2T3: use
    # the curve's stable instantaneous_forward (day-step) rather than
    # an eps=1e-5 finite difference that the date-rounding kills.
    r0 = g2pp.curve.instantaneous_forward(0.0)

    if is_callable:
        result = callable_frn(
            reference_date=ref,
            maturity_years=maturity_years,
            spread=spread,
            hw_a=g2pp.a,
            hw_sigma=g2pp.sigma1,
            r0=r0,
            call_dates_years=option_dates_years,
            call_price=exercise_price,
            frequency=frequency,
            notional=notional,
            n_steps=n_steps,
        )
    else:
        result = puttable_frn(
            reference_date=ref,
            maturity_years=maturity_years,
            spread=spread,
            hw_a=g2pp.a,
            hw_sigma=g2pp.sigma1,
            r0=r0,
            put_dates_years=option_dates_years,
            put_price=exercise_price,
            frequency=frequency,
            notional=notional,
            n_steps=n_steps,
        )
    return result.price


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def callable_frn_g2pp(
    g2pp: G2PlusPlus,
    maturity_years: float,
    spread: float,
    call_dates_years: list[float],
    call_price: float = 100.0,
    n_steps: int = 50,
    notional: float = 100.0,
    frequency: int = 4,
) -> CallableFloaterG2PPResult:
    """Callable floating-rate note priced on the G2++ 2D trinomial tree.

    The issuer can redeem on any call date at `call_price`.  At each call date
    the tree enforces ``value = min(continuation, call_price)``.

    The floating coupon at each tree node is computed as::

        coupon(x, y, t) = notional * max(r(x, y, t) + spread, 0) * period

    where r(x, y, t) = x + y + phi(t) is the G2++ short rate at the node.
    This captures the fact that the FRN coupon resets with the short rate at
    each period.

    Args:
        g2pp: calibrated G2PlusPlus model (carries curve + vol parameters).
        maturity_years: FRN tenor in years.
        spread: fixed spread over the floating index (decimal, e.g. 0.005 = 50 bp).
        call_dates_years: list of Bermudan call dates as year fractions.
        call_price: redemption price on call (per notional units, default 100).
        n_steps: number of tree time steps.
        notional: face value.
        frequency: coupon periods per year (default 4 = quarterly).

    Returns:
        :class:`CallableFloaterG2PPResult` with price, straight FRN benchmark,
        option value, HW1F benchmark, and two-factor premium.
    """
    price = _g2pp_frn_tree(
        g2pp, maturity_years, spread, call_dates_years,
        call_price, n_steps, notional, frequency, is_callable=True,
    )
    straight = _straight_frn_g2pp_tree(
        g2pp, maturity_years, spread, frequency, notional, n_steps,
    )
    option_value = max(0.0, straight - price)

    hw1f = _hw1f_frn(
        g2pp, maturity_years, spread, call_dates_years,
        call_price, n_steps, notional, frequency, is_callable=True,
    )
    two_factor_premium = price - hw1f

    return CallableFloaterG2PPResult(
        price=price,
        straight_price=straight,
        option_value=option_value,
        hw1f_price=hw1f,
        two_factor_premium=two_factor_premium,
    )


def puttable_frn_g2pp(
    g2pp: G2PlusPlus,
    maturity_years: float,
    spread: float,
    put_dates_years: list[float],
    put_price: float = 100.0,
    n_steps: int = 50,
    notional: float = 100.0,
    frequency: int = 4,
) -> CallableFloaterG2PPResult:
    """Puttable floating-rate note priced on the G2++ 2D trinomial tree.

    The investor can sell back on any put date at `put_price`.  At each put
    date the tree enforces ``value = max(continuation, put_price)``.

    Args:
        g2pp: calibrated G2PlusPlus model.
        maturity_years: FRN tenor in years.
        spread: fixed spread over the floating index (decimal).
        put_dates_years: list of put dates as year fractions.
        put_price: redemption price on put (per notional units, default 100).
        n_steps: number of tree time steps.
        notional: face value.
        frequency: coupon periods per year (default 4 = quarterly).

    Returns:
        :class:`CallableFloaterG2PPResult` with price, straight FRN benchmark,
        option value, HW1F benchmark, and two-factor premium.
    """
    price = _g2pp_frn_tree(
        g2pp, maturity_years, spread, put_dates_years,
        put_price, n_steps, notional, frequency, is_callable=False,
    )
    straight = _straight_frn_g2pp_tree(
        g2pp, maturity_years, spread, frequency, notional, n_steps,
    )
    option_value = max(0.0, price - straight)

    hw1f = _hw1f_frn(
        g2pp, maturity_years, spread, put_dates_years,
        put_price, n_steps, notional, frequency, is_callable=False,
    )
    two_factor_premium = price - hw1f

    return CallableFloaterG2PPResult(
        price=price,
        straight_price=straight,
        option_value=option_value,
        hw1f_price=hw1f,
        two_factor_premium=two_factor_premium,
    )
