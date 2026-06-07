"""
Callable and puttable bonds under the G2++ two-factor Hull-White model.

The 1-factor callable bond (Hull-White trinomial tree) is in callable_bond.py.
This module extends the pricing to the G2++ model, where two correlated
Ornstein-Uhlenbeck factors (x, y) drive the short rate:

    r(t) = x(t) + y(t) + phi(t)
    dx = -a*x*dt + sigma1*dW1
    dy = -b*y*dt + sigma2*dW2
    dW1 dW2 = rho dt

Backward induction on a product 2D trinomial lattice prices the embedded
American-style call or put option.  Each node discounts using the G2++
short rate r = x + y + phi(t).

The two-factor model captures a richer volatility term structure and
non-unit correlation between rate moves at different tenors, making option
values differ materially from the 1-factor Hull-White result.  The field
`two_factor_premium` quantifies this difference.

Usage::

    from pricebook.fixed_income.callable_bond_g2pp import (
        callable_bond_g2pp, puttable_bond_g2pp, callable_bond_oas_g2pp,
    )

    result = callable_bond_g2pp(g2pp, coupon_rate=0.05, maturity_years=10,
                                call_dates_years=[5, 6, 7, 8, 9, 10])

References:
    Brigo, D. & Mercurio, F., *Interest Rate Models — Theory and Practice*,
    2nd ed., Ch. 4.3, Springer, 2006.
    Hull, J., *Options, Futures, and Other Derivatives*, 10th ed., Ch. 31.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.vasicek import G2PlusPlus
from pricebook.models.g2pp_tree import G2PPTree
from pricebook.core.day_count import date_from_year_fraction
from pricebook.core.solvers import brentq


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CallableBondG2PPResult:
    """Result from G2++ callable/puttable bond pricing.

    Attributes:
        price: model price of the callable or puttable bond.
        straight_price: price of the equivalent straight (non-optioned) bond.
        option_value: |price - straight_price|; embedded option value.
        oas: option-adjusted spread (bp shift to market curve).  Non-zero only
            when computed via :func:`callable_bond_oas_g2pp`.
        hw1f_price: equivalent 1-factor Hull-White price using only the
            x-factor (sigma2=0), for benchmarking.
        two_factor_premium: price - hw1f_price; captures the extra value
            added by the second factor.
    """
    price: float
    straight_price: float
    option_value: float
    oas: float
    hw1f_price: float
    two_factor_premium: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Internal: straight bond price via G2++ term structure
# ---------------------------------------------------------------------------


def _straight_bond_g2pp(
    g2pp: G2PlusPlus,
    coupon_rate: float,
    maturity_years: float,
    notional: float,
    coupon_frequency: float,
) -> float:
    """Straight bond price using the G2++ market discount factors directly.

    Because phi(t) is calibrated to match P_mkt(T), the straight bond price
    is simply the present value of coupons and principal off the market curve.
    No tree required.
    """
    ref = g2pp.curve.reference_date
    period = 1.0 / coupon_frequency
    pv = 0.0
    t_pay = period
    while t_pay <= maturity_years + 1e-10:
        d = date_from_year_fraction(ref, t_pay)
        df = g2pp.curve.df(d)
        pv += notional * coupon_rate * period * df
        t_pay += period
    d_mat = date_from_year_fraction(ref, maturity_years)
    pv += notional * g2pp.curve.df(d_mat)
    return pv


# ---------------------------------------------------------------------------
# Internal: 1-factor benchmark (sigma2 = 0)
# ---------------------------------------------------------------------------


def _hw1f_callable_bond(
    g2pp: G2PlusPlus,
    coupon_rate: float,
    maturity_years: float,
    option_dates_years: list[float],
    exercise_price: float,
    n_steps: int,
    notional: float,
    coupon_frequency: float,
    is_callable: bool,
) -> float:
    """Price a callable/puttable bond with sigma2=0 (1-factor benchmark).

    Uses the Hull-White trinomial tree from callable_bond.py with parameters
    taken from the G2++ x-factor only.
    """
    from pricebook.models.hull_white import HullWhite
    from pricebook.fixed_income.callable_bond import (
        callable_bond_price, puttable_bond_price,
    )

    hw = HullWhite(a=g2pp.a, sigma=g2pp.sigma1, curve=g2pp.curve)
    if is_callable:
        return callable_bond_price(
            hw, coupon_rate, maturity_years, option_dates_years,
            exercise_price, n_steps, notional, coupon_frequency,
        )
    return puttable_bond_price(
        hw, coupon_rate, maturity_years, option_dates_years,
        exercise_price, n_steps, notional, coupon_frequency,
    )


# ---------------------------------------------------------------------------
# Internal: 2D tree backward induction for fixed-coupon optionable bond
# ---------------------------------------------------------------------------


def _g2pp_bond_tree(
    g2pp: G2PlusPlus,
    coupon_rate: float,
    maturity_years: float,
    option_dates_years: list[float],
    exercise_price: float,
    n_steps: int,
    notional: float,
    coupon_frequency: float,
    is_callable: bool,
) -> float:
    """Core G2++ product-lattice pricing for callable or puttable fixed bond.

    Terminal value: notional + last coupon (paid at maturity).
    Backward step: discount by exp(-r_node * dt), add coupons on coupon dates,
    apply option constraint on exercise dates.

    The G2PPTree.backward_induction applies discounting internally and calls
    option_func(t_idx, xi, yi, continuation) at every node.  We encode both
    coupon injection and exercise constraint into this single callback.

    Args:
        g2pp: calibrated G2PlusPlus model.
        coupon_rate: annual coupon rate (e.g. 0.05 for 5%).
        maturity_years: bond maturity.
        option_dates_years: list of exercise dates (years from today).
        exercise_price: call/put strike (typically 100 per notional).
        n_steps: number of time steps in the tree.
        notional: face value.
        coupon_frequency: coupon periods per year.
        is_callable: True for callable (min constraint), False for puttable.

    Returns:
        Bond price at the root node.
    """
    tree = G2PPTree(g2pp, maturity_years, n_steps)
    dt = tree.dt
    period = 1.0 / coupon_frequency
    coupon_amount = notional * coupon_rate * period

    # Identify coupon steps and option steps (as t_idx of the *arriving* slice)
    # The tree backward_induction calls option_func after discounting one step
    # from t_idx to t_idx+1, so t_idx is the time from which we rolled back.
    # Step t_idx means the slice just computed corresponds to time t_idx * dt.
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

    # Terminal value: notional + last coupon at every node
    terminal = np.full((tree.n_x, tree.n_y),
                       notional + coupon_amount)

    # The option_func callback receives the discounted continuation value.
    # We need to track which time step we are processing.  We use a list as
    # a mutable closure counter since the tree sweeps from n_steps-1 down to 0.
    # t_idx inside backward_induction corresponds to the FROM-step; the value
    # array being filled corresponds to time t_idx * dt.
    def _node_func(t_idx: int, xi: int, yi: int, cont: float) -> float:
        # t_idx is the step we rolled BACK FROM (time t_idx * dt).
        # The value at t_idx+1 has already been discounted into cont.
        # Now we add cash flows that arrive at t_idx+1 (index t_idx+1).
        arriving_step = t_idx + 1
        value = cont
        if arriving_step in coupon_steps and arriving_step < n_steps:
            # Last coupon already included in terminal; skip it here.
            value += coupon_amount
        if arriving_step in option_steps:
            if is_callable:
                value = min(value, exercise_price)
            else:
                value = max(value, exercise_price)
        return value

    return tree.backward_induction(terminal_values=terminal, option_func=_node_func)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def callable_bond_g2pp(
    g2pp: G2PlusPlus,
    coupon_rate: float,
    maturity_years: float,
    call_dates_years: list[float],
    call_price: float = 100.0,
    n_steps: int = 50,
    notional: float = 100.0,
    coupon_frequency: float = 2.0,
) -> CallableBondG2PPResult:
    """Callable bond price on the G2++ 2D trinomial tree.

    The issuer can redeem on any call date at `call_price`.  At each call date
    the tree enforces ``value = min(continuation, call_price)``.

    Args:
        g2pp: calibrated G2PlusPlus model (carries curve + vol parameters).
        coupon_rate: annual coupon rate (e.g. 0.05 = 5%).
        maturity_years: bond tenor in years.
        call_dates_years: list of Bermudan call dates as year fractions.
        call_price: redemption price on call (per notional units, default 100).
        n_steps: number of tree time steps (higher = more accurate, slower).
        notional: face value.
        coupon_frequency: coupon periods per year (default 2 = semi-annual).

    Returns:
        :class:`CallableBondG2PPResult` with price, straight price, option value,
        HW1F benchmark, and two-factor premium.
    """
    price = _g2pp_bond_tree(
        g2pp, coupon_rate, maturity_years, call_dates_years,
        call_price, n_steps, notional, coupon_frequency, is_callable=True,
    )
    straight = _straight_bond_g2pp(
        g2pp, coupon_rate, maturity_years, notional, coupon_frequency,
    )
    option_value = max(0.0, straight - price)

    hw1f = _hw1f_callable_bond(
        g2pp, coupon_rate, maturity_years, call_dates_years,
        call_price, n_steps, notional, coupon_frequency, is_callable=True,
    )
    two_factor_premium = price - hw1f

    return CallableBondG2PPResult(
        price=price,
        straight_price=straight,
        option_value=option_value,
        oas=0.0,
        hw1f_price=hw1f,
        two_factor_premium=two_factor_premium,
    )


def puttable_bond_g2pp(
    g2pp: G2PlusPlus,
    coupon_rate: float,
    maturity_years: float,
    put_dates_years: list[float],
    put_price: float = 100.0,
    n_steps: int = 50,
    notional: float = 100.0,
    coupon_frequency: float = 2.0,
) -> CallableBondG2PPResult:
    """Puttable bond price on the G2++ 2D trinomial tree.

    The investor can sell back on any put date at `put_price`.  At each put
    date the tree enforces ``value = max(continuation, put_price)``.

    Args:
        g2pp: calibrated G2PlusPlus model.
        coupon_rate: annual coupon rate.
        maturity_years: bond tenor in years.
        put_dates_years: list of put dates as year fractions.
        put_price: redemption price on put (per notional units, default 100).
        n_steps: number of tree time steps.
        notional: face value.
        coupon_frequency: coupon periods per year.

    Returns:
        :class:`CallableBondG2PPResult` with price, straight price, option value,
        HW1F benchmark, and two-factor premium.
    """
    price = _g2pp_bond_tree(
        g2pp, coupon_rate, maturity_years, put_dates_years,
        put_price, n_steps, notional, coupon_frequency, is_callable=False,
    )
    straight = _straight_bond_g2pp(
        g2pp, coupon_rate, maturity_years, notional, coupon_frequency,
    )
    option_value = max(0.0, price - straight)

    hw1f = _hw1f_callable_bond(
        g2pp, coupon_rate, maturity_years, put_dates_years,
        put_price, n_steps, notional, coupon_frequency, is_callable=False,
    )
    two_factor_premium = price - hw1f

    return CallableBondG2PPResult(
        price=price,
        straight_price=straight,
        option_value=option_value,
        oas=0.0,
        hw1f_price=hw1f,
        two_factor_premium=two_factor_premium,
    )


def callable_bond_oas_g2pp(
    g2pp: G2PlusPlus,
    market_price: float,
    coupon_rate: float,
    maturity_years: float,
    call_dates_years: list[float],
    call_price: float = 100.0,
    n_steps: int = 50,
) -> float:
    """Option-adjusted spread (OAS) for a callable bond under G2++.

    The OAS is the constant parallel shift `s` added to the market discount
    curve such that the G2++ callable bond model price matches the observed
    market price::

        price(g2pp with bumped curve, oas=s) == market_price

    A positive OAS indicates the bond is cheap relative to the model
    (higher yield than the model-implied fair value).

    Args:
        g2pp: calibrated G2PlusPlus model.
        market_price: observed market price (per notional 100).
        coupon_rate: annual coupon rate.
        maturity_years: bond tenor in years.
        call_dates_years: Bermudan call date schedule.
        call_price: call strike (default 100).
        n_steps: tree time steps.

    Returns:
        OAS in decimal (e.g. 0.005 = 50 bp).
    """
    notional = 100.0
    coupon_frequency = 2.0

    def _objective(spread: float) -> float:
        bumped_curve = g2pp.curve.bumped(spread)
        g2pp_bumped = G2PlusPlus(
            a=g2pp.a, b=g2pp.b,
            sigma1=g2pp.sigma1, sigma2=g2pp.sigma2,
            rho=g2pp.rho, curve=bumped_curve,
        )
        model_price = _g2pp_bond_tree(
            g2pp_bumped, coupon_rate, maturity_years, call_dates_years,
            call_price, n_steps, notional, coupon_frequency, is_callable=True,
        )
        return model_price - market_price

    return brentq(_objective, -0.10, 0.50)
