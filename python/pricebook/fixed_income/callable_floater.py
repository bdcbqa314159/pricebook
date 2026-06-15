"""
Callable and puttable floating-rate notes (FRN) priced on the Hull-White tree.

A callable FRN gives the issuer the right to redeem at a fixed call price on
selected coupon dates. A puttable FRN gives the investor the right to sell
back at a fixed put price. In both cases the embedded option is valued via
backward induction on the Hull-White trinomial tree.

Coupon at each payment date:
    coupon_t = notional * (r_t + spread) * period_length

where r_t is the floating rate (the prevailing short rate at the fixing date
— a standard Libor/SOFR proxy used in tree-based models).

    from pricebook.fixed_income.callable_floater import callable_frn, puttable_frn

    result = callable_frn(
        reference_date=date(2024, 1, 15),
        maturity_years=5.0,
        spread=0.005,
        hw_a=0.05, hw_sigma=0.01, r0=0.04,
        call_dates_years=[2.0, 3.0, 4.0],
    )

References:
    Fabozzi, *Bond Markets, Analysis and Strategies*, Ch. 17, 2012.
    Tuckman & Serrat, *Fixed Income Securities*, Ch. 19, 2012.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import date_from_year_fraction


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CallableFloaterResult:
    """Result from callable/puttable FRN pricing.

    Attributes:
        price: price of the callable/puttable FRN (per notional units).
        straight_frn_price: price of the equivalent straight (non-optioned) FRN.
        option_value: |price - straight_frn_price|; value of the embedded option.
        oas: option-adjusted spread (set only when computed via callable_frn_oas).
        call_probability: estimated probability the issuer will call.
        expected_call_date: probability-weighted average call date (years).
    """
    price: float
    straight_frn_price: float
    option_value: float
    oas: float
    call_probability: float
    expected_call_date: float

    def to_dict(self) -> dict:
        return vars(self)


# ---------------------------------------------------------------------------
# Internal: straight FRN price on HW tree (floating coupons + principal)
# ---------------------------------------------------------------------------


def _straight_frn_tree(
    hw_a: float,
    hw_sigma: float,
    r0: float,
    maturity_years: float,
    spread: float,
    frequency: int,
    notional: float,
    n_steps: int,
) -> float:
    """Straight FRN value on HW trinomial tree.

    Coupon at each period = notional * (r_node + spread) * period_length.
    For a flat curve the straight FRN should equal notional.
    """
    period = 1.0 / frequency
    dt = maturity_years / n_steps
    dr = hw_sigma * math.sqrt(3.0 * dt)
    j_max = max(1, int(math.ceil(0.1835 / (max(hw_a, 1e-6) * dt))))
    n_nodes = 2 * j_max + 1
    mid = j_max

    coupon_steps: set[int] = set()
    t_pay = period
    while t_pay <= maturity_years + 1e-10:
        coupon_steps.add(int(round(t_pay / dt)))
        t_pay += period

    # Terminal: principal + last floating coupon (cum-coupon at maturity).
    values = np.zeros(n_nodes)
    last_step = int(round(maturity_years / dt))
    for j in range(-j_max, j_max + 1):
        r_j = r0 + j * dr
        coupon = notional * max(r_j + spread, 0.0) * period
        values[j + mid] = notional + coupon

    # Fix T4-CF1: pre-fix used ``/6`` in the trinomial drift terms (Hull
    # §32.4 has ``/2``) and added the coupon AFTER the backward
    # discount.  Both are corrected below — coupon is added to ``values``
    # at step+1 BEFORE the one-step discount so it gets the right DF.
    for step in range(n_steps - 1, -1, -1):
        sp1 = step + 1
        # Apply coupon at step+1 (before discount).  Terminal coupon is
        # already in the init — skip it.
        if sp1 in coupon_steps and sp1 != last_step:
            for j in range(-j_max, j_max + 1):
                r_j = r0 + j * dr
                values[j + mid] += notional * max(r_j + spread, 0.0) * period

        new_values = np.zeros(n_nodes)
        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr
            one_step_df = math.exp(-r_j * dt)

            # Textbook HW trinomial probabilities — drift term / 2.
            p_up = 1.0/6 + (j*j * hw_a*hw_a * dt*dt - j * hw_a * dt) / 2
            p_dn = 1.0/6 + (j*j * hw_a*hw_a * dt*dt + j * hw_a * dt) / 2
            p_up = max(0.0, min(1.0, p_up))
            p_dn = max(0.0, min(1.0, p_dn))
            p_mid = max(0.0, 1.0 - p_up - p_dn)

            j_up = min(j + 1, j_max)
            j_dn = max(j - 1, -j_max)

            cont = (p_up * values[j_up + mid]
                    + p_mid * values[j + mid]
                    + p_dn * values[j_dn + mid])
            new_values[idx] = cont * one_step_df

        values = new_values

    return float(values[mid])


# ---------------------------------------------------------------------------
# Internal: generic trinomial tree with callable/puttable constraint
# ---------------------------------------------------------------------------


def _frn_tree_with_option(
    hw_a: float,
    hw_sigma: float,
    r0: float,
    maturity_years: float,
    spread: float,
    frequency: int,
    notional: float,
    n_steps: int,
    option_steps: dict[int, float],  # step -> exercise price
    is_callable: bool,               # True=callable (min), False=puttable (max)
    oas_shift: float = 0.0,
) -> tuple[float, list[tuple[int, float]]]:
    """Trinomial tree for FRN with embedded call or put option.

    Returns:
        (price_at_root, list of (step, exercise_gain)) for diagnostics.
    """
    period = 1.0 / frequency
    dt = maturity_years / n_steps
    dr = hw_sigma * math.sqrt(3.0 * dt)
    j_max = max(1, int(math.ceil(0.1835 / (max(hw_a, 1e-6) * dt))))
    n_nodes = 2 * j_max + 1
    mid = j_max

    coupon_steps: set[int] = set()
    t_pay = period
    while t_pay <= maturity_years + 1e-10:
        coupon_steps.add(int(round(t_pay / dt)))
        t_pay += period

    last_step = int(round(maturity_years / dt))

    # Terminal: principal + last coupon (cum-coupon)
    values = np.zeros(n_nodes)
    for j in range(-j_max, j_max + 1):
        r_j = r0 + j * dr + oas_shift
        coupon = notional * max(r_j + spread, 0.0) * period
        values[j + mid] = notional + coupon

    exercise_gains: list[tuple[int, float]] = []

    # Same T4-CF1 fixes as ``_straight_frn_tree`` plus option-applied-
    # before-discount: pre-fix the call/put compared discounted
    # continuation (in step units) against the undiscounted par strike
    # (in step+1 units), biasing the exercise decision by ~exp(r·dt)
    # per step.
    for step in range(n_steps - 1, -1, -1):
        sp1 = step + 1

        # Apply coupon at step+1 (before discount).  Terminal already
        # in init.
        if sp1 in coupon_steps and sp1 != last_step:
            for j in range(-j_max, j_max + 1):
                r_j = r0 + j * dr + oas_shift
                values[j + mid] += notional * max(r_j + spread, 0.0) * period

        # Apply option at step+1 (before discount) so the comparison is
        # in consistent step+1 units.
        if sp1 in option_steps:
            ex_price = option_steps[sp1]
            before = float(values[mid])
            if is_callable:
                values = np.minimum(values, ex_price)
            else:
                values = np.maximum(values, ex_price)
            after = float(values[mid])
            exercise_gains.append((sp1, abs(after - before)))

        new_values = np.zeros(n_nodes)
        for j in range(-j_max, j_max + 1):
            idx = j + mid
            r_j = r0 + j * dr + oas_shift
            one_step_df = math.exp(-r_j * dt)

            # Textbook HW trinomial probabilities — drift term / 2.
            p_up = 1.0/6 + (j*j * hw_a*hw_a * dt*dt - j * hw_a * dt) / 2
            p_dn = 1.0/6 + (j*j * hw_a*hw_a * dt*dt + j * hw_a * dt) / 2
            p_up = max(0.0, min(1.0, p_up))
            p_dn = max(0.0, min(1.0, p_dn))
            p_mid = max(0.0, 1.0 - p_up - p_dn)

            j_up = min(j + 1, j_max)
            j_dn = max(j - 1, -j_max)

            cont = (p_up * values[j_up + mid]
                    + p_mid * values[j + mid]
                    + p_dn * values[j_dn + mid])
            new_values[idx] = cont * one_step_df

        values = new_values

    return float(values[mid]), exercise_gains


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def callable_frn(
    reference_date: date,
    maturity_years: float,
    spread: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    call_dates_years: list[float],
    call_price: float = 100.0,
    frequency: int = 4,
    notional: float = 100.0,
    n_steps: int = 200,
) -> CallableFloaterResult:
    """Callable FRN priced on the Hull-White trinomial tree.

    The issuer can redeem on any call date at call_price. At each call date
    the tree enforces: ``node_value = min(continuation_value, call_price)``.

    Args:
        reference_date: pricing date (not used directly; tree starts at r0).
        maturity_years: bond tenor in years.
        spread: fixed spread over the floating index (e.g. 0.005 = 50 bp).
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        call_dates_years: list of call dates as year fractions.
        call_price: redemption price at call (per notional units).
        frequency: coupon frequency (periods per year, default 4).
        notional: face value.
        n_steps: number of tree time steps.

    Returns:
        CallableFloaterResult with price, straight FRN benchmark, and diagnostics.
    """
    dt = maturity_years / n_steps
    option_steps = {int(round(t / dt)): call_price
                    for t in call_dates_years
                    if 0 < t <= maturity_years + 1e-10}

    price, ex_gains = _frn_tree_with_option(
        hw_a, hw_sigma, r0, maturity_years, spread, frequency,
        notional, n_steps, option_steps, is_callable=True,
    )

    straight_price = _straight_frn_tree(
        hw_a, hw_sigma, r0, maturity_years, spread, frequency, notional, n_steps,
    )

    option_value = max(0.0, straight_price - price)

    # Call probability: fraction of steps where call was triggered, weighted
    n_call_steps = len(option_steps)
    call_prob = min(1.0, sum(g for _, g in ex_gains) / max(notional, 1.0)) if ex_gains else 0.0
    call_prob = min(1.0, call_prob)

    # Expected call date: weight by gain magnitude
    total_gain = sum(g for _, g in ex_gains)
    if total_gain > 0:
        expected_call_date = sum(s * dt * g for s, g in ex_gains) / total_gain
    else:
        expected_call_date = maturity_years

    return CallableFloaterResult(
        price=price,
        straight_frn_price=straight_price,
        option_value=option_value,
        oas=0.0,
        call_probability=call_prob,
        expected_call_date=expected_call_date,
    )


def puttable_frn(
    reference_date: date,
    maturity_years: float,
    spread: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    put_dates_years: list[float],
    put_price: float = 100.0,
    frequency: int = 4,
    notional: float = 100.0,
    n_steps: int = 200,
) -> CallableFloaterResult:
    """Puttable FRN priced on the Hull-White trinomial tree.

    The investor can sell back on any put date at put_price. At each put date
    the tree enforces: ``node_value = max(continuation_value, put_price)``.

    Args:
        reference_date: pricing date.
        maturity_years: bond tenor in years.
        spread: spread over the floating index.
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        put_dates_years: list of put dates as year fractions.
        put_price: redemption price at put (per notional units).
        frequency: coupon frequency (periods per year, default 4).
        notional: face value.
        n_steps: number of tree time steps.

    Returns:
        CallableFloaterResult with price, straight FRN benchmark, and diagnostics.
    """
    dt = maturity_years / n_steps
    option_steps = {int(round(t / dt)): put_price
                    for t in put_dates_years
                    if 0 < t <= maturity_years + 1e-10}

    price, ex_gains = _frn_tree_with_option(
        hw_a, hw_sigma, r0, maturity_years, spread, frequency,
        notional, n_steps, option_steps, is_callable=False,
    )

    straight_price = _straight_frn_tree(
        hw_a, hw_sigma, r0, maturity_years, spread, frequency, notional, n_steps,
    )

    option_value = max(0.0, price - straight_price)

    total_gain = sum(g for _, g in ex_gains)
    put_prob = min(1.0, total_gain / max(notional, 1.0)) if ex_gains else 0.0
    dt_val = maturity_years / n_steps
    if total_gain > 0:
        expected_put_date = sum(s * dt_val * g for s, g in ex_gains) / total_gain
    else:
        expected_put_date = maturity_years

    return CallableFloaterResult(
        price=price,
        straight_frn_price=straight_price,
        option_value=option_value,
        oas=0.0,
        call_probability=put_prob,
        expected_call_date=expected_put_date,
    )


def callable_frn_oas(
    market_price: float,
    reference_date: date,
    maturity_years: float,
    spread: float,
    hw_a: float,
    hw_sigma: float,
    r0: float,
    call_dates_years: list[float],
    call_price: float = 100.0,
    frequency: int = 4,
    notional: float = 100.0,
    n_steps: int = 200,
) -> CallableFloaterResult:
    """Option-adjusted spread (OAS) for a callable FRN.

    Finds the constant spread to the short rate such that the model price
    matches the market price. Uses Brent root-finding.

    The OAS is the shift ``s`` applied uniformly to all discount rates:
        ``price(oas=s) = market_price``

    Args:
        market_price: observed market price (per notional units).
        reference_date: pricing date.
        maturity_years: bond tenor in years.
        spread: coupon spread over floating (e.g. 0.005).
        hw_a: Hull-White mean reversion speed.
        hw_sigma: Hull-White volatility.
        r0: initial short rate.
        call_dates_years: call date schedule.
        call_price: call redemption price.
        frequency: coupon frequency.
        notional: face value.
        n_steps: tree time steps.

    Returns:
        CallableFloaterResult with the OAS field populated.
    """
    from pricebook.core.solvers import brentq

    dt = maturity_years / n_steps
    option_steps = {int(round(t / dt)): call_price
                    for t in call_dates_years
                    if 0 < t <= maturity_years + 1e-10}

    def _price_at_oas(s: float) -> float:
        p, _ = _frn_tree_with_option(
            hw_a, hw_sigma, r0 + s, maturity_years, spread, frequency,
            notional, n_steps, option_steps, is_callable=True, oas_shift=0.0,
        )
        return p - market_price

    try:
        oas_val = brentq(_price_at_oas, -0.05, 0.20, xtol=1e-6, maxiter=100)
    except Exception:
        oas_val = float("nan")

    # Price at solved OAS
    price, ex_gains = _frn_tree_with_option(
        hw_a, hw_sigma, r0 + (oas_val if math.isfinite(oas_val) else 0.0),
        maturity_years, spread, frequency,
        notional, n_steps, option_steps, is_callable=True,
    )

    straight_price = _straight_frn_tree(
        hw_a, hw_sigma, r0, maturity_years, spread, frequency, notional, n_steps,
    )

    total_gain = sum(g for _, g in ex_gains)
    call_prob = min(1.0, total_gain / max(notional, 1.0)) if ex_gains else 0.0
    if total_gain > 0:
        expected_call_date = sum(s * dt * g for s, g in ex_gains) / total_gain
    else:
        expected_call_date = maturity_years

    return CallableFloaterResult(
        price=market_price,
        straight_frn_price=straight_price,
        option_value=max(0.0, straight_price - market_price),
        oas=oas_val,
        call_probability=call_prob,
        expected_call_date=expected_call_date,
    )


# ---------------------------------------------------------------------------
# HullWhite-object convenience wrappers
# ---------------------------------------------------------------------------


def callable_frn_hw(
    hw,
    reference_date: date,
    maturity_years: float,
    spread: float,
    call_dates_years: list[float],
    call_price: float = 100.0,
    frequency: int = 4,
    notional: float = 100.0,
    n_steps: int = 200,
) -> CallableFloaterResult:
    """Convenience: callable FRN from a HullWhite object.

    Extracts ``hw.a``, ``hw.sigma``, and the instantaneous forward rate at
    t=0 from the curve, then delegates to :func:`callable_frn`.

    Args:
        hw: a ``pricebook.models.hull_white.HullWhite`` instance.
        reference_date: pricing date.
        maturity_years: bond tenor in years.
        spread: fixed spread over the floating index.
        call_dates_years: list of call dates as year fractions.
        call_price: redemption price at call (per notional units).
        frequency: coupon frequency (periods per year).
        notional: face value.
        n_steps: number of tree time steps.

    Returns:
        CallableFloaterResult with price, straight FRN benchmark, and diagnostics.
    """
    return callable_frn(
        reference_date, maturity_years, spread,
        hw.a, hw.sigma, hw.curve.instantaneous_forward(0.0),
        call_dates_years, call_price, frequency, notional, n_steps,
    )


def puttable_frn_hw(
    hw,
    reference_date: date,
    maturity_years: float,
    spread: float,
    put_dates_years: list[float],
    put_price: float = 100.0,
    frequency: int = 4,
    notional: float = 100.0,
    n_steps: int = 200,
) -> CallableFloaterResult:
    """Convenience: puttable FRN from a HullWhite object.

    Extracts ``hw.a``, ``hw.sigma``, and the instantaneous forward rate at
    t=0 from the curve, then delegates to :func:`puttable_frn`.

    Args:
        hw: a ``pricebook.models.hull_white.HullWhite`` instance.
        reference_date: pricing date.
        maturity_years: bond tenor in years.
        spread: fixed spread over the floating index.
        put_dates_years: list of put dates as year fractions.
        put_price: redemption price at put (per notional units).
        frequency: coupon frequency (periods per year).
        notional: face value.
        n_steps: number of tree time steps.

    Returns:
        CallableFloaterResult with price, straight FRN benchmark, and diagnostics.
    """
    return puttable_frn(
        reference_date, maturity_years, spread,
        hw.a, hw.sigma, hw.curve.instantaneous_forward(0.0),
        put_dates_years, put_price, frequency, notional, n_steps,
    )
