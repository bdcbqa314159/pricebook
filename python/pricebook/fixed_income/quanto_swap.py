"""Quanto (differential) interest rate swap.

A quanto swap pays a foreign floating rate in domestic currency,
or pays the difference between two rates in different currencies.
The FX-rate correlation creates a convexity adjustment.

* :class:`QuantoSwapResult` — pricing result.
* :func:`quanto_swap_price` — quanto IR swap pricing.
* :func:`differential_swap_price` — rate differential swap.
* :func:`quanto_adjustment_term_structure` — adjustment per period.
* :func:`quanto_fra` — single-period quanto forward rate agreement.

References:
    Brigo & Mercurio, *Interest Rate Models*, Ch. 14.4 (Quanto Swaps).
    Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 33.
    Piterbarg, *Rates Squared*, Risk, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from dateutil.relativedelta import relativedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.day_count import DayCountConvention, year_fraction
from pricebook.core.schedule import Frequency


@dataclass
class QuantoSwapResult:
    """Quanto swap pricing result."""
    pv: float
    domestic_pv: float          # PV in domestic currency
    par_spread: float           # spread to make PV = 0
    quanto_adjustment_total: float  # total convexity adjustment
    n_periods: int
    foreign_rate_avg: float     # average forward foreign rate
    adjusted_rate_avg: float    # average quanto-adjusted rate

    def to_dict(self) -> dict:
        return dict(vars(self))


def quanto_swap_price(
    reference_date: date,
    maturity_years: float,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    fixed_rate: float,
    rate_vol: float,
    fx_vol: float,
    correlation: float,
    notional: float = 10_000_000.0,
    pay_foreign_float: bool = True,
    frequency: Frequency = Frequency.QUARTERLY,
    spread: float = 0.0,
) -> QuantoSwapResult:
    """Price a quanto interest rate swap.

    Structure: pay foreign floating rate (quanto-adjusted) + spread,
    receive domestic fixed rate. All cashflows in domestic currency.

    The foreign rate fixing L^f is adjusted:
        E^d[L^f(T)] = L^f × (1 − σ_L × σ_FX × ρ × T)

    When ρ > 0: foreign rate up → FX appreciates → domestic value
    of foreign rate is lower → negative adjustment.

    Args:
        domestic_curve: domestic (payment) currency discount curve.
        foreign_curve: foreign rate projection curve.
        fixed_rate: fixed leg rate.
        rate_vol: vol of the foreign floating rate.
        fx_vol: vol of the FX rate (domestic per foreign).
        correlation: correlation between foreign rate and FX.
        pay_foreign_float: True = pay foreign float, receive fixed.
        spread: additional spread on floating leg.
    """
    n_per_year = {
        Frequency.QUARTERLY: 4, Frequency.SEMI_ANNUAL: 2,
        Frequency.ANNUAL: 1, Frequency.MONTHLY: 12,
    }.get(frequency, 4)

    dt = 1.0 / n_per_year
    n_periods = int(maturity_years * n_per_year)

    float_pv = 0.0
    fixed_pv = 0.0
    total_adj = 0.0
    rate_sum = 0.0
    adj_rate_sum = 0.0

    for i in range(1, n_periods + 1):
        t_start = (i - 1) * dt
        t_end = i * dt

        start_date = reference_date + relativedelta(months=int(t_start * 12))
        end_date = reference_date + relativedelta(months=int(t_end * 12))

        # Foreign forward rate for this period
        df_for_start = foreign_curve.df(start_date)
        df_for_end = foreign_curve.df(end_date)
        fwd_foreign = (df_for_start / df_for_end - 1) / dt if df_for_end > 0 else 0

        # Quanto adjustment: E^d[L] = L × (1 − σ_L × σ_FX × ρ × T_fix)
        # T_fix is the fixing time (start of accrual period), not payment time
        adjustment = -fwd_foreign * rate_vol * fx_vol * correlation * t_start
        adjusted_rate = fwd_foreign + adjustment + spread

        # Domestic discount factor for payment
        df_dom = domestic_curve.df(end_date)

        # Float leg: adjusted foreign rate × dt × notional × df_dom
        float_pv += adjusted_rate * dt * notional * df_dom

        # Fixed leg: fixed_rate × dt × notional × df_dom
        fixed_pv += fixed_rate * dt * notional * df_dom

        total_adj += adjustment * dt * notional * df_dom
        rate_sum += fwd_foreign
        adj_rate_sum += adjusted_rate

    if pay_foreign_float:
        pv = fixed_pv - float_pv  # receive fixed, pay float
    else:
        pv = float_pv - fixed_pv  # receive float, pay fixed

    # Par spread: spread that makes PV = 0
    annuity = sum(
        dt * domestic_curve.df(reference_date + relativedelta(months=int(i * dt * 12)))
        for i in range(1, n_periods + 1)
    )
    par_spread = (fixed_pv / notional - (float_pv - spread * notional * annuity) / notional) / annuity if annuity > 0 else 0

    return QuantoSwapResult(
        pv=pv,
        domestic_pv=pv,
        par_spread=par_spread,
        quanto_adjustment_total=total_adj,
        n_periods=n_periods,
        foreign_rate_avg=rate_sum / n_periods if n_periods > 0 else 0,
        adjusted_rate_avg=adj_rate_sum / n_periods if n_periods > 0 else 0,
    )


@dataclass
class DifferentialSwapResult:
    """Differential (diff) swap result."""
    pv: float
    rate_differential: float    # avg(rate_1 − rate_2)
    quanto_adj_1: float
    quanto_adj_2: float
    n_periods: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def differential_swap_price(
    reference_date: date,
    maturity_years: float,
    curve_1: DiscountCurve,
    curve_2: DiscountCurve,
    payment_curve: DiscountCurve,
    fixed_spread: float,
    vol_1: float,
    vol_2: float,
    fx_vol: float,
    corr_1_fx: float,
    corr_2_fx: float,
    notional: float = 10_000_000.0,
    frequency: Frequency = Frequency.QUARTERLY,
) -> DifferentialSwapResult:
    """Price a differential (diff) swap.

    Pays: rate_1 − rate_2 − fixed_spread, all in one currency.
    Both rates are quanto-adjusted to the payment currency.

    Common examples:
    - USD SOFR − EUR ESTR (paid in USD)
    - GBP SONIA − EUR ESTR (paid in GBP)

    Args:
        curve_1: rate 1 projection curve.
        curve_2: rate 2 projection curve.
        payment_curve: discount curve for payment currency.
        fixed_spread: fixed spread (diff swap strike).
        vol_1, vol_2: rate vols.
        fx_vol: FX vol (payment ccy per rate-2 ccy).
        corr_1_fx: correlation(rate_1, FX).
        corr_2_fx: correlation(rate_2, FX).
    """
    n_per_year = {
        Frequency.QUARTERLY: 4, Frequency.SEMI_ANNUAL: 2,
        Frequency.ANNUAL: 1, Frequency.MONTHLY: 12,
    }.get(frequency, 4)

    dt = 1.0 / n_per_year
    n_periods = int(maturity_years * n_per_year)

    pv = 0.0
    total_adj_1 = 0.0
    total_adj_2 = 0.0
    diff_sum = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        start_date = reference_date + relativedelta(months=int((i - 1) * dt * 12))
        end_date = reference_date + relativedelta(months=int(t * 12))

        # Forward rates
        df1_s = curve_1.df(start_date)
        df1_e = curve_1.df(end_date)
        fwd_1 = (df1_s / df1_e - 1) / dt if df1_e > 0 else 0

        df2_s = curve_2.df(start_date)
        df2_e = curve_2.df(end_date)
        fwd_2 = (df2_s / df2_e - 1) / dt if df2_e > 0 else 0

        # Quanto adjustments — both rates adjusted to payment currency
        # T_fix is start of accrual period
        t_fix = (i - 1) * dt
        adj_1 = -fwd_1 * vol_1 * fx_vol * corr_1_fx * t_fix
        adj_2 = -fwd_2 * vol_2 * fx_vol * corr_2_fx * t_fix

        adjusted_1 = fwd_1 + adj_1
        adjusted_2 = fwd_2 + adj_2

        # Differential payment
        diff = adjusted_1 - adjusted_2 - fixed_spread
        df_pay = payment_curve.df(end_date)
        pv += diff * dt * notional * df_pay

        total_adj_1 += adj_1
        total_adj_2 += adj_2
        diff_sum += adjusted_1 - adjusted_2

    return DifferentialSwapResult(
        pv=pv,
        rate_differential=diff_sum / n_periods if n_periods > 0 else 0,
        quanto_adj_1=total_adj_1,
        quanto_adj_2=total_adj_2,
        n_periods=n_periods,
    )


def quanto_adjustment_term_structure(
    foreign_curve: DiscountCurve,
    reference_date: date,
    rate_vol: float,
    fx_vol: float,
    correlation: float,
    tenors_years: list[float] | None = None,
) -> list[dict]:
    """Quanto adjustment term structure per tenor.

    Shows how the adjustment grows with maturity.

    Args:
        foreign_curve: foreign rate curve.
        tenors_years: list of tenors (default: standard).
    """
    tenors = tenors_years or [0.25, 0.5, 1, 2, 3, 5, 7, 10]
    result = []

    for T in tenors:
        end_date = reference_date + relativedelta(months=int(T * 12))
        start_date = reference_date + relativedelta(months=int((T - 0.25) * 12))

        df_s = foreign_curve.df(start_date)
        df_e = foreign_curve.df(end_date)
        fwd = (df_s / df_e - 1) * 4 if df_e > 0 else 0  # quarterly

        adj = -fwd * rate_vol * fx_vol * correlation * T
        adj_bps = adj * 10_000

        result.append({
            "tenor_years": T,
            "forward_rate": fwd,
            "adjustment_bps": adj_bps,
            "adjusted_rate": fwd + adj,
        })

    return result


@dataclass
class QuantoFRAResult:
    """Quanto FRA result."""
    pv: float
    foreign_forward: float
    quanto_adjusted_forward: float
    adjustment_bps: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def quanto_fra(
    reference_date: date,
    fixing_date: date,
    maturity_date: date,
    domestic_curve: DiscountCurve,
    foreign_curve: DiscountCurve,
    strike: float,
    rate_vol: float,
    fx_vol: float,
    correlation: float,
    notional: float = 10_000_000.0,
) -> QuantoFRAResult:
    """Quanto forward rate agreement: single-period quanto.

    Pay foreign rate (adjusted) vs fixed strike, settled in domestic.

    Args:
        fixing_date: rate fixing date.
        maturity_date: payment date.
        strike: agreed fixed rate.
    """
    dc = DayCountConvention.ACT_360
    T_fix = year_fraction(reference_date, fixing_date, dc)
    tau = year_fraction(fixing_date, maturity_date, dc)

    # Foreign forward rate
    df_for_fix = foreign_curve.df(fixing_date)
    df_for_mat = foreign_curve.df(maturity_date)
    fwd = (df_for_fix / df_for_mat - 1) / tau if df_for_mat > 0 and tau > 0 else 0

    # Quanto adjustment
    adj = -fwd * rate_vol * fx_vol * correlation * T_fix
    adjusted_fwd = fwd + adj

    # PV: (adjusted_fwd − strike) × tau × notional × df_domestic
    df_dom = domestic_curve.df(maturity_date)
    pv = (adjusted_fwd - strike) * tau * notional * df_dom

    return QuantoFRAResult(
        pv=pv,
        foreign_forward=fwd,
        quanto_adjusted_forward=adjusted_fwd,
        adjustment_bps=adj * 10_000,
    )
