"""Bermudan CDS swaption: multiple exercise dates.

The holder can enter a CDS (payer or receiver of protection)
at any of the specified exercise dates. At each date, they compare
the exercise value (forward CDS PV) against the continuation value.

    from pricebook.credit.bermudan_cds_swaption import (
        bermudan_cds_swaption_price, BermudanCDSSwaptionResult,
    )

References:
    Pedersen (2003). Bermudan Swaptions in the LIBOR Market Model.
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


@dataclass
class BermudanCDSSwaptionResult:
    """Bermudan CDS swaption pricing result."""
    price: float
    european_price: float           # price with only first exercise date
    early_exercise_premium: float   # bermudan - european
    exercise_probability: float     # fraction of steps where exercise is optimal
    n_exercise_dates: int
    maturity_years: float

    def to_dict(self) -> dict:
        return vars(self)


def bermudan_cds_swaption_price(
    reference_date: date,
    exercise_dates: list[date],
    cds_maturity: date,
    strike_spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    notional: float = 10_000_000.0,
    is_payer: bool = True,
    n_steps: int = 100,
) -> BermudanCDSSwaptionResult:
    """Price a Bermudan CDS swaption via backward induction.

    The holder can exercise at any of the exercise_dates to enter a
    CDS (payer = buy protection, receiver = sell protection) at the
    strike spread, with the CDS running to cds_maturity.

    At each exercise date:
        exercise_value = forward CDS PV from exercise_date to cds_maturity
        value = max(continuation, exercise_value)

    Args:
        exercise_dates: dates at which exercise is allowed.
        cds_maturity: CDS end date if exercised.
        strike_spread: running spread of the CDS entered on exercise.
        is_payer: True = right to buy protection (payer swaption).

    Returns:
        BermudanCDSSwaptionResult.
    """
    dc = DayCountConvention.ACT_365_FIXED
    T = year_fraction(reference_date, cds_maturity, dc)
    dt = T / n_steps

    # Map exercise dates to step indices
    exercise_steps = set()
    for ed in exercise_dates:
        t_ex = year_fraction(reference_date, ed, dc)
        step = round(t_ex / dt)
        step = max(1, min(step, n_steps - 1))
        exercise_steps.add(step)

    # Backward induction
    values = [0.0] * (n_steps + 1)
    exercise_count = 0
    total_exercise_checks = 0

    for step in range(n_steps - 1, -1, -1):
        t = step * dt
        t_next = (step + 1) * dt

        t_date = reference_date + timedelta(days=round(t * 365.25))
        t_next_date = reference_date + timedelta(days=round(t_next * 365.25))

        q_t = survival_curve.survival(t_date)
        q_next = survival_curve.survival(t_next_date)
        p_survive = min(q_next / max(q_t, 1e-10), 1.0)
        p_default = max(1 - p_survive, 0.0)

        df = discount_curve.df(t_next_date) / max(discount_curve.df(t_date), 1e-10) \
            if t_next < T else 1.0

        # Continuation value (discounted, survival-weighted)
        cont = values[step + 1] * p_survive * df

        # Exercise decision at exercise dates
        if step in exercise_steps:
            # Forward CDS PV from this point to maturity
            exercise_value = _forward_cds_pv(
                t, T, dt, strike_spread, recovery, notional,
                discount_curve, survival_curve, reference_date, is_payer,
            )

            total_exercise_checks += 1
            if exercise_value > cont:
                exercise_count += 1

            values[step] = max(cont, exercise_value)
        else:
            values[step] = cont

    bermudan_price = values[0]

    # European price (first exercise date only, same backward induction method)
    if len(exercise_dates) > 1:
        euro_result = bermudan_cds_swaption_price(
            reference_date, [exercise_dates[0]], cds_maturity,
            strike_spread, discount_curve, survival_curve,
            recovery, notional, is_payer, n_steps,
        )
        european_price = euro_result.price
    else:
        european_price = bermudan_price  # single date → Bermudan = European

    exercise_prob = exercise_count / max(total_exercise_checks, 1)

    return BermudanCDSSwaptionResult(
        price=bermudan_price,
        european_price=european_price,
        early_exercise_premium=max(bermudan_price - european_price, 0),
        exercise_probability=exercise_prob,
        n_exercise_dates=len(exercise_dates),
        maturity_years=T,
    )


def _forward_cds_pv(
    t_start: float,
    t_end: float,
    dt: float,
    spread: float,
    recovery: float,
    notional: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    reference_date: date,
    is_payer: bool,
) -> float:
    """Compute forward CDS PV from t_start to t_end (in year fractions)."""
    prot_pv = 0.0
    prem_pv = 0.0

    n_sub = max(1, int((t_end - t_start) / dt))
    sub_dt = (t_end - t_start) / n_sub

    prev_q = survival_curve.survival(reference_date + timedelta(days=round(t_start * 365.25)))

    for i in range(1, n_sub + 1):
        t = t_start + i * sub_dt
        t_date = reference_date + timedelta(days=round(t * 365.25))

        q = survival_curve.survival(t_date)
        df = discount_curve.df(t_date)
        default_prob = max(prev_q - q, 0)

        prot_pv += (1 - recovery) * notional * default_prob * df
        prem_pv += spread * sub_dt * notional * q * df

        prev_q = q

    if is_payer:
        return prot_pv - prem_pv  # buy protection: receive protection, pay premium
    else:
        return prem_pv - prot_pv  # sell protection: receive premium, pay protection


def _european_cds_swaption_price(
    reference_date: date,
    expiry: date,
    cds_maturity: date,
    strike_spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float,
    notional: float,
    is_payer: bool,
) -> float:
    """Price a European CDS swaption (exercise at single date)."""
    dc = DayCountConvention.ACT_365_FIXED
    T_exp = year_fraction(reference_date, expiry, dc)
    T_mat = year_fraction(reference_date, cds_maturity, dc)

    # Survival to expiry (knockout factor)
    q_exp = survival_curve.survival(expiry)

    # Forward CDS PV
    fwd_pv = _forward_cds_pv(
        T_exp, T_mat, 0.25, strike_spread, recovery, notional,
        discount_curve, survival_curve, reference_date, is_payer,
    )

    # Option value: max(fwd_pv, 0) × survival_to_expiry
    return max(fwd_pv, 0) * q_exp
