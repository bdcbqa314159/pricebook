"""Exotic credit payoffs: capped coupon bonds, digital CDS, range accrual, credit loans.

Phase C2 slices 215-216 consolidated.

* :class:`CappedCouponBond` — floating + spread, capped, with default risk.
* :class:`DigitalCDS` — fixed payout on default (no recovery uncertainty).
* :class:`CreditRangeAccrual` — accrues when spread stays in [L, U].
* :class:`CreditLinkedLoan` — loan with margin grid and covenant triggers.

References:
    Schönbucher, *Credit Derivatives Pricing Models*, Wiley, 2003, Ch. 10.
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Wiley, 2008, Ch. 12.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from pricebook.discount_curve import DiscountCurve
from pricebook.survival_curve import SurvivalCurve


# ---- Capped coupon bond ----

@dataclass
class CappedCouponBondResult:
    """Pricing result for a capped coupon bond with default risk."""
    dirty_price: float
    coupon_pv: float
    principal_pv: float
    recovery_pv: float
    risk_free_price: float


def capped_coupon_bond(
    notional: float,
    floating_rate: float,
    spread: float,
    cap: float,
    maturity_years: int,
    frequency: int = 2,
    discount_curve: DiscountCurve | None = None,
    survival_curve: SurvivalCurve | None = None,
    recovery: float = 0.4,
    flat_rate: float = 0.05,
    flat_hazard: float = 0.02,
) -> CappedCouponBondResult:
    """Price a capped floating-rate bond with default risk.

    Coupon per period = min(floating_rate + spread, cap) × notional × dt.
    On default: bondholder receives recovery × notional.

    Args:
        floating_rate: current floating rate (assumed constant for simplicity).
        spread: credit spread over floating.
        cap: maximum coupon rate.
        maturity_years: bond maturity in years.
        frequency: coupons per year.
    """
    n_periods = maturity_years * frequency
    dt = 1.0 / frequency
    coupon_rate = min(floating_rate + spread, cap)

    coupon_pv = 0.0
    principal_pv = 0.0
    recovery_pv = 0.0
    rf_price = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        if discount_curve:
            df = discount_curve.df(discount_curve.reference_date)  # simplified
            df = math.exp(-flat_rate * t)
        else:
            df = math.exp(-flat_rate * t)

        if survival_curve:
            surv = math.exp(-flat_hazard * t)
        else:
            surv = math.exp(-flat_hazard * t)

        surv_prev = math.exp(-flat_hazard * (t - dt))

        # Coupon conditional on survival
        coupon = coupon_rate * notional * dt
        coupon_pv += df * surv * coupon
        rf_price += df * coupon

        # Default in this period: recovery
        default_prob = surv_prev - surv
        recovery_pv += df * default_prob * recovery * notional

    # Principal at maturity conditional on survival
    T = maturity_years
    df_T = math.exp(-flat_rate * T)
    surv_T = math.exp(-flat_hazard * T)
    principal_pv = df_T * surv_T * notional
    rf_price += df_T * notional

    dirty = coupon_pv + principal_pv + recovery_pv

    return CappedCouponBondResult(dirty, coupon_pv, principal_pv, recovery_pv, rf_price)


# ---- Digital CDS ----

@dataclass
class DigitalCDSResult:
    """Pricing result for a digital CDS."""
    pv: float
    digital_payout: float
    premium_pv: float
    par_spread: float


def digital_cds(
    notional: float,
    digital_payout: float,
    spread: float,
    maturity_years: int,
    flat_rate: float = 0.05,
    flat_hazard: float = 0.02,
    frequency: int = 4,
) -> DigitalCDSResult:
    """Price a digital CDS: fixed payout on default, no recovery uncertainty.

    Protection leg: pays digital_payout if default occurs (vs standard
    CDS which pays notional × (1 − recovery)).

    Premium leg: periodic spread × notional × dt conditional on survival.

    PV(protection buyer) = PV(digital payout) − PV(premium).

    Args:
        digital_payout: fixed amount paid on default.
        spread: running premium as fraction of notional.
    """
    dt = 1.0 / frequency
    n_periods = maturity_years * frequency

    protection_pv = 0.0
    premium_pv = 0.0
    annuity = 0.0

    for i in range(1, n_periods + 1):
        t = i * dt
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * (t - dt))

        default_prob = surv_prev - surv
        protection_pv += df * default_prob * digital_payout
        premium_pv += df * surv * spread * notional * dt
        annuity += df * surv * dt

    pv = protection_pv - premium_pv

    # Par spread: spread that makes PV = 0
    if annuity * notional > 0:
        par = protection_pv / (annuity * notional)
    else:
        par = 0.0

    return DigitalCDSResult(pv, digital_payout, premium_pv, par)


# ---- Credit range accrual ----

@dataclass
class CreditRangeAccrualResult:
    """Pricing result for a credit range accrual."""
    pv: float
    expected_accrual_days: float
    total_days: int
    accrual_fraction: float


def credit_range_accrual(
    notional: float,
    coupon_rate: float,
    lower_spread: float,
    upper_spread: float,
    maturity_years: float,
    current_spread: float,
    spread_vol: float,
    flat_rate: float = 0.05,
    flat_hazard: float = 0.02,
    n_days: int = 252,
) -> CreditRangeAccrualResult:
    """Price a credit range accrual note.

    Pays coupon × (fraction of days where spread ∈ [L, U]) × notional.
    The fraction is estimated via a normal approximation on the
    spread process.

    On default: recovery × notional (standard credit risk).

    Args:
        lower_spread / upper_spread: range boundaries (in bps or fraction).
        current_spread: current CDS spread level.
        spread_vol: annualised volatility of the spread.
    """
    from scipy.stats import norm

    total_days = int(maturity_years * n_days)
    dt_day = maturity_years / total_days

    # Expected fraction of days in range (normal approximation)
    # Assume spread follows: S(t) ~ N(current_spread, spread_vol² × t)
    expected_in_range = 0.0
    for d in range(1, total_days + 1):
        t = d * dt_day
        std = spread_vol * math.sqrt(t)
        if std > 0:
            p_in = norm.cdf(upper_spread, current_spread, std) - \
                   norm.cdf(lower_spread, current_spread, std)
        else:
            p_in = 1.0 if lower_spread <= current_spread <= upper_spread else 0.0
        expected_in_range += p_in

    accrual_fraction = expected_in_range / total_days

    # PV: coupon × fraction × annuity × survival + recovery on default
    df_T = math.exp(-flat_rate * maturity_years)
    surv_T = math.exp(-flat_hazard * maturity_years)

    coupon_pv = coupon_rate * notional * accrual_fraction * df_T * surv_T * maturity_years
    recovery_pv = df_T * (1 - surv_T) * 0.4 * notional

    pv = coupon_pv + recovery_pv

    return CreditRangeAccrualResult(pv, expected_in_range, total_days, accrual_fraction)


# ---- Credit-linked loan ----

@dataclass
class CreditLinkedLoanResult:
    """Pricing result for a credit-linked loan."""
    pv: float
    expected_loss: float
    margin: float
    covenant_breached: bool
    effective_spread: float


def credit_linked_loan(
    principal: float,
    base_rate: float,
    margin: float,
    maturity_years: int,
    flat_hazard: float = 0.02,
    flat_rate: float = 0.05,
    recovery: float = 0.4,
    leverage_ratio: float = 3.0,
    max_leverage: float = 5.0,
    margin_grid: list[tuple[float, float]] | None = None,
) -> CreditLinkedLoanResult:
    """Price a loan with credit risk, margin grid, and covenant triggers.

    The margin adjusts based on a leverage-ratio grid:
        margin_grid = [(threshold, margin), ...] sorted by threshold.

    If leverage_ratio exceeds max_leverage, covenant is breached.

    Args:
        principal: loan amount.
        base_rate: risk-free floating rate.
        margin: initial credit margin (spread).
        leverage_ratio: current debt/EBITDA.
        max_leverage: covenant trigger level.
        margin_grid: [(leverage_threshold, margin_bps), ...].
            If None, uses flat margin.
    """
    covenant_breached = leverage_ratio > max_leverage

    # Determine effective margin from grid
    effective_margin = margin
    if margin_grid:
        for threshold, grid_margin in sorted(margin_grid):
            if leverage_ratio <= threshold:
                effective_margin = grid_margin
                break
        else:
            effective_margin = margin_grid[-1][1]

    # PV of loan cashflows with default risk
    total_rate = base_rate + effective_margin
    dt = 1.0
    pv = 0.0
    expected_loss = 0.0

    for i in range(1, maturity_years + 1):
        t = float(i)
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * (t - 1))

        # Interest conditional on survival
        pv += df * surv * total_rate * principal

        # Default in this year: loss = (1 - recovery) × principal
        default_prob = surv_prev - surv
        loss = default_prob * (1 - recovery) * principal
        expected_loss += df * loss

        # Recovery on default
        pv += df * default_prob * recovery * principal

    # Principal repayment at maturity
    df_T = math.exp(-flat_rate * maturity_years)
    surv_T = math.exp(-flat_hazard * maturity_years)
    pv += df_T * surv_T * principal

    return CreditLinkedLoanResult(
        pv, expected_loss, effective_margin, covenant_breached,
        effective_margin,
    )
