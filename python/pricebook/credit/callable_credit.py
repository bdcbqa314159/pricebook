"""Callable bonds with credit risk — credit-risky OAS.

Prices callable bonds where the issuer can both default and call.
The OAS on a survival-weighted tree isolates optionality from credit.

    from pricebook.credit.callable_credit import (
        callable_credit_bond_price, credit_risky_oas,
    )

References:
    Das & Sundaram (2007). An Integrated Model for Hybrid Securities.
    Jarrow & Turnbull (2000). The Intersection of Market and Credit Risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.solvers import brentq


@dataclass
class CallableCreditResult:
    """Result of callable credit bond pricing."""
    price: float
    price_no_call: float         # straight bond (no call, with credit)
    price_no_credit: float       # callable but no credit risk
    call_option_value: float     # price_no_call - price (issuer's call value)
    credit_spread_bp: float      # implied credit spread
    oas_bp: float | None         # option-adjusted spread if market price given

    def to_dict(self) -> dict:
        return vars(self)


def callable_credit_bond_price(
    coupon: float,
    maturity_years: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.40,
    call_dates_years: list[float] | None = None,
    call_price: float = 100.0,
    face: float = 100.0,
    freq: int = 2,
    n_steps: int = 50,
    _compute_decomposition: bool = True,
) -> CallableCreditResult:
    """Price a callable bond with credit risk via backward induction.

    At each time step:
    - Probability of survival → continue (or call if call date)
    - Probability of default → receive recovery × face

    The issuer calls when continuation value > call_price AND they are not in default.

    Args:
        coupon: annual coupon rate.
        maturity_years: years to maturity.
        discount_curve: risk-free curve.
        survival_curve: issuer survival curve.
        recovery: recovery rate on default.
        call_dates_years: when the bond can be called (default: every coupon date).
        call_price: call strike (default par).
        face: face value.
        freq: coupon frequency.
        n_steps: number of time steps.
    """
    dt = maturity_years / n_steps
    ref = discount_curve.reference_date
    cpn_per_step = coupon / freq * face

    if call_dates_years is None:
        call_dates_years = [i / freq for i in range(1, int(maturity_years * freq) + 1)]
    call_steps = set(int(round(t / dt)) for t in call_dates_years if t <= maturity_years)

    # Coupon steps
    cpn_interval = max(1, int(round(1.0 / freq / dt)))

    # Forward rates and survival probabilities
    dfs = []
    survs = []
    for i in range(n_steps + 1):
        t = i * dt
        d = _date_at(ref, t)
        dfs.append(discount_curve.df(d))
        survs.append(survival_curve.survival(d))

    # Backward induction
    # Value at maturity: face + last coupon (if surviving)
    values = np.full(1, face + cpn_per_step)

    for step in range(n_steps - 1, -1, -1):
        t = step * dt

        # Conditional survival over this step
        q_prev = survs[step]
        q_next = survs[step + 1]
        p_survive = min(q_next / q_prev, 1.0) if q_prev > 0 else 0.0
        p_default = 1.0 - p_survive

        # One-step discount
        df_ratio = dfs[step + 1] / dfs[step] if dfs[step] > 0 else 1.0

        # Continuation value: survive × (discounted future + coupon) + default × recovery
        cont = p_survive * values[0] * df_ratio + p_default * recovery * face * df_ratio

        # Add coupon at coupon dates
        if (step + 1) % cpn_interval == 0 and step > 0:
            cont += p_survive * cpn_per_step * df_ratio

        # Call decision: issuer calls if continuation > call_price
        if step in call_steps:
            cont = min(cont, call_price)

        values[0] = cont

    price = float(values[0])

    # Decomposition (skip for internal calls to avoid recursion)
    if _compute_decomposition:
        price_no_call = _straight_credit_price(
            coupon, maturity_years, discount_curve, survival_curve, recovery, face, freq,
        )
        price_no_credit = _callable_riskfree_price(
            coupon, maturity_years, discount_curve, call_dates_years, call_price, face, freq, n_steps,
        )
        credit_spread = _implied_credit_spread(
            price, coupon, maturity_years, discount_curve, face, freq,
        )
    else:
        price_no_call = price
        price_no_credit = price
        credit_spread = 0.0

    call_value = price_no_call - price

    return CallableCreditResult(
        price=price,
        price_no_call=price_no_call,
        price_no_credit=price_no_credit,
        call_option_value=max(call_value, 0),
        credit_spread_bp=credit_spread * 10_000,
        oas_bp=None,
    )


def credit_risky_oas(
    market_price: float,
    coupon: float,
    maturity_years: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.40,
    call_dates_years: list[float] | None = None,
    call_price: float = 100.0,
    face: float = 100.0,
) -> float:
    """Credit-risky OAS: spread over risk-free that reprices the callable credit bond.

    Unlike standard OAS (which doesn't separate credit from optionality),
    credit-risky OAS uses the survival curve to handle default risk explicitly,
    so the OAS represents pure optionality + liquidity.
    """
    def objective(spread: float) -> float:
        shifted = discount_curve.bumped(spread)
        result = callable_credit_bond_price(
            coupon, maturity_years, shifted, survival_curve, recovery,
            call_dates_years, call_price, face,
        )
        return result.price - market_price

    try:
        return brentq(objective, -0.05, 0.20)
    except ValueError:
        return 0.0


def _date_at(ref, t):
    from datetime import date
    return date.fromordinal(ref.toordinal() + int(t * 365))


def _straight_credit_price(coupon, T, dc, sc, recovery, face, freq):
    """Straight (non-callable) credit bond price."""
    ref = dc.reference_date
    n = int(T * freq)
    cpn = coupon / freq * face
    pv = 0.0
    prev_surv = 1.0
    for i in range(1, n + 1):
        t = i / freq
        d = _date_at(ref, t)
        df = dc.df(d)
        surv = sc.survival(d)
        pv += cpn * df * surv
        pv += recovery * face * df * (prev_surv - surv)
        prev_surv = surv
    d_mat = _date_at(ref, T)
    pv += face * dc.df(d_mat) * sc.survival(d_mat)
    return pv


def _callable_riskfree_price(coupon, T, dc, call_dates, call_price, face, freq, n_steps):
    """Callable risk-free bond (no credit risk)."""
    from pricebook.core.survival_curve import SurvivalCurve
    ref = dc.reference_date
    # Use zero-hazard survival curve
    sc = SurvivalCurve.flat(ref, 0.0, tenors=list(range(1, int(T) + 2)))
    result = callable_credit_bond_price(
        coupon, T, dc, sc, 0.0, call_dates, call_price, face, freq, n_steps,
        _compute_decomposition=False,
    )
    return result.price


def _implied_credit_spread(price, coupon, T, dc, face, freq):
    """Implied credit spread from price (Z-spread)."""
    def obj(s):
        ref = dc.reference_date
        pv = 0.0
        n = int(T * freq)
        cpn = coupon / freq * face
        for i in range(1, n + 1):
            t = i / freq
            d = _date_at(ref, t)
            pv += cpn * dc.df(d) * math.exp(-s * t)
        pv += face * dc.df(_date_at(ref, T)) * math.exp(-s * T)
        return pv - price
    try:
        return brentq(obj, -0.05, 0.50)
    except ValueError:
        return 0.0
