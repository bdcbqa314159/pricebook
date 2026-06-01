"""Callable Credit-Linked Note.

A CLN with an embedded call option: the issuer can redeem at par
(or a specified call price) on pre-defined dates. When credit improves,
the issuer calls to refinance at a lower spread.

    from pricebook.credit.callable_cln import (
        callable_cln_price, CallableCLNResult,
    )

Decomposition:
    Callable CLN PV = Straight CLN PV - Call Option Value
    Call option value > 0 when CLN trades above par (tight credit)

References:
    O'Kane (2008). Modelling Single-name and Multi-name Credit Derivatives.
    Tavella (2002). Quantitative Methods in Derivatives Pricing, Ch. 10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency, generate_schedule


@dataclass
class CallableCLNResult:
    """Callable CLN pricing result."""
    callable_price: float           # price with call option
    straight_price: float           # price without call (vanilla CLN)
    call_option_value: float        # straight - callable (issuer's call value)
    early_call_probability: float   # fraction of scenarios where call is exercised
    expected_call_date_years: float | None  # expected time to call
    par_spread_callable: float      # spread that makes callable CLN = notional

    def to_dict(self) -> dict:
        return vars(self)


def callable_cln_price(
    start: date,
    end: date,
    coupon_rate: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.4,
    notional: float = 100.0,
    call_dates: list[date] | None = None,
    call_price: float = 100.0,
    frequency: Frequency = Frequency.QUARTERLY,
    day_count: DayCountConvention = DayCountConvention.ACT_360,
) -> CallableCLNResult:
    """Price a callable CLN via backward induction.

    At each call date, the issuer redeems if the CLN's continuation value
    exceeds the call price (i.e., credit has improved enough that the
    note trades above par).

    Args:
        start: CLN start date.
        end: CLN maturity date.
        coupon_rate: annual coupon rate.
        discount_curve: risk-free curve.
        survival_curve: reference entity survival curve.
        recovery: recovery rate on default.
        notional: face value.
        call_dates: dates at which issuer can call. Default: annual from year 1.
        call_price: call strike (typically par = 100).
        frequency: coupon frequency.
        day_count: coupon day count.

    Returns:
        CallableCLNResult.
    """
    schedule = generate_schedule(start, end, frequency)
    n_periods = len(schedule) - 1

    if call_dates is None:
        from dateutil.relativedelta import relativedelta
        call_dates = []
        for y in range(1, int(year_fraction(start, end, DayCountConvention.ACT_365_FIXED))):
            call_dates.append(start + relativedelta(years=y))

    call_date_set = set(call_dates)

    # Step 1: Price straight (non-callable) CLN
    straight_price = _price_straight_cln(
        schedule, coupon_rate, discount_curve, survival_curve,
        recovery, notional, day_count,
    )

    # Step 2: Backward induction with call constraint
    # At each period end (going backward), compute:
    #   continuation = coupon_pv + principal_pv + recovery_pv (looking forward)
    #   if call date: value = min(continuation, call_price × notional / 100)

    # Build per-period values
    period_values = [0.0] * (n_periods + 1)
    period_values[n_periods] = notional  # terminal: return notional (if survived)

    call_exercised_count = 0
    total_call_dates = 0
    first_call_period = None

    for i in range(n_periods - 1, -1, -1):
        t_start = schedule[i]
        t_end = schedule[i + 1]
        tau = year_fraction(t_start, t_end, day_count)
        df = discount_curve.df(t_end)
        df_ratio = discount_curve.df(t_end) / max(discount_curve.df(t_start), 1e-10)

        q_start = survival_curve.survival(t_start)
        q_end = survival_curve.survival(t_end)
        p_survive = min(q_end / max(q_start, 1e-10), 1.0)
        p_default = max(1 - p_survive, 0.0)

        # Continuation value (looking forward from t_end)
        forward_value = period_values[i + 1]

        # This period's cashflow
        coupon = notional * coupon_rate * tau
        recovery_cf = recovery * notional * p_default

        # Value at t_start: discount back
        cont_value = (p_survive * forward_value + recovery_cf) * df_ratio + coupon * df_ratio

        # Apply call constraint
        is_call_date = t_end in call_date_set
        if is_call_date:
            total_call_dates += 1
            if cont_value > call_price:
                cont_value = call_price
                call_exercised_count += 1
                if first_call_period is None:
                    first_call_period = i + 1

        period_values[i] = cont_value

    callable_price = period_values[0]
    call_option_value = max(straight_price - callable_price, 0)

    # Call probability (simplified: fraction of call dates where value > call_price)
    early_call_prob = call_exercised_count / max(total_call_dates, 1)

    # Expected call date
    if first_call_period is not None:
        expected_call_years = year_fraction(start, schedule[first_call_period],
                                             DayCountConvention.ACT_365_FIXED)
    else:
        expected_call_years = None

    # Par spread for callable CLN (spread that makes callable_price = notional)
    # Approximate: par_spread ≈ coupon_rate × (straight / callable) adjustment
    if callable_price > 0:
        par_spread = coupon_rate * notional / callable_price
    else:
        par_spread = coupon_rate

    return CallableCLNResult(
        callable_price=callable_price,
        straight_price=straight_price,
        call_option_value=call_option_value,
        early_call_probability=early_call_prob,
        expected_call_date_years=expected_call_years,
        par_spread_callable=par_spread,
    )


def _price_straight_cln(
    schedule: list[date],
    coupon_rate: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float,
    notional: float,
    day_count: DayCountConvention,
) -> float:
    """Price a straight (non-callable) CLN."""
    pv = 0.0
    prev_surv = 1.0

    for i in range(1, len(schedule)):
        tau = year_fraction(schedule[i-1], schedule[i], day_count)
        df = discount_curve.df(schedule[i])
        surv = survival_curve.survival(schedule[i])
        default_prob = prev_surv - surv

        pv += notional * coupon_rate * tau * df * surv
        pv += recovery * notional * default_prob * df

        prev_surv = surv

    pv += notional * discount_curve.df(schedule[-1]) * survival_curve.survival(schedule[-1])
    return pv
