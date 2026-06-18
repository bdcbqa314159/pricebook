"""Asset-backed securities: auto loans, credit cards, student loans.

Cashflow engine for ABS with amortisation, controlled amortisation,
revolving periods, charge-offs, and recovery lags.

* :class:`ABSPool` — ABS pool specification.
* :class:`ABSResult` — pricing result.
* :func:`price_auto_abs` — auto loan ABS (fixed-rate amortising).
* :func:`price_credit_card_abs` — credit card ABS (revolving + controlled amort).
* :func:`price_student_loan_abs` — student loan ABS (income-driven, grace period).

References:
    Fabozzi, *The Handbook of Fixed Income Securities*, 8th ed., Ch. 21.
    Gorton & Souleles, *Special Purpose Vehicles and Securitization*, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from pricebook.core.discount_curve import DiscountCurve


@dataclass
class ABSPool:
    """ABS collateral pool specification."""
    original_balance: float
    wac: float              # weighted average coupon (annual)
    wal_months: int         # weighted average life (months)
    n_loans: int = 1000
    charge_off_rate: float = 0.02   # annual charge-off rate
    recovery_rate: float = 0.30     # recovery on charged-off loans
    recovery_lag_months: int = 6    # months until recovery received

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class ABSTranche:
    """ABS tranche specification."""
    name: str
    notional: float
    coupon: float           # annual coupon rate
    seniority: int          # 0 = most senior

    def to_dict(self) -> dict:
        return dict(vars(self))


@dataclass
class ABSResult:
    """ABS tranche pricing result."""
    price: float            # per 100 face
    yield_to_maturity: float
    wal: float              # weighted average life (years)
    credit_enhancement_pct: float
    expected_loss_pct: float
    break_even_loss_rate: float

    def to_dict(self) -> dict:
        return dict(vars(self))


# ═══════════════════════════════════════════════════════════════
# Auto Loan ABS
# ═══════════════════════════════════════════════════════════════

def price_auto_abs(
    pool: ABSPool,
    tranches: list[ABSTranche],
    discount_curve: DiscountCurve,
    spread: float = 0.0,
    abs_speed: float = 1.3,
) -> list[ABSResult]:
    """Price auto loan ABS tranches.

    Auto loans amortise monthly with fixed payments. Prepayments
    are measured in ABS speed (like PSA for MBS). Losses hit
    from the bottom tranche up.

    Args:
        pool: auto loan pool.
        tranches: tranche structure (senior to junior).
        discount_curve: risk-free curve.
        spread: static spread over curve.
        abs_speed: prepayment speed multiple (1.0 = baseline ~1.5% CPR).
    """
    n_months = pool.wal_months
    balance = pool.original_balance
    monthly_rate = pool.wac / 12.0
    ref = discount_curve.reference_date

    # Total subordination
    total_notional = sum(t.notional for t in tranches)
    sorted_tranches = sorted(tranches, key=lambda t: t.seniority)

    # Generate pool cashflows
    pool_interest = []
    pool_principal = []
    pool_losses = []

    for m in range(1, n_months + 1):
        if balance < 0.01:
            pool_interest.append(0)
            pool_principal.append(0)
            pool_losses.append(0)
            continue

        # Auto ABS prepayment: baseline ~1.5% CPR, ramping
        base_cpr = min(0.015 * abs_speed, 0.50)
        smm = 1.0 - (1.0 - base_cpr) ** (1.0 / 12.0)

        # Charge-offs
        monthly_co = balance * pool.charge_off_rate / 12.0

        interest = balance * monthly_rate
        remaining = n_months - m + 1
        if monthly_rate > 0 and remaining > 0:
            sched_payment = balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining))
            sched_principal = sched_payment - interest
        else:
            sched_principal = balance / max(remaining, 1)

        sched_principal = max(min(sched_principal, balance), 0)
        prepayment = (balance - sched_principal - monthly_co) * smm
        prepayment = max(prepayment, 0)

        pool_interest.append(interest)
        pool_principal.append(sched_principal + prepayment)
        pool_losses.append(monthly_co)

        balance -= (sched_principal + prepayment + monthly_co)
        balance = max(balance, 0)

    # Waterfall: allocate to tranches
    results = []
    for tranche in sorted_tranches:
        tr_balance = tranche.notional
        tr_pv = 0.0
        tr_principal_pv = 0.0
        tr_wal_num = 0.0
        tr_total_prin = 0.0
        cum_loss = 0.0

        # Credit enhancement: subordination below this tranche
        sub_below = sum(
            t.notional for t in sorted_tranches if t.seniority > tranche.seniority
        )
        ce_pct = sub_below / total_notional * 100 if total_notional > 0 else 0

        for m in range(n_months):
            if tr_balance < 0.01:
                break

            t_years = (m + 1) / 12.0
            try:
                cf_date = ref + timedelta(days=round(t_years * 365.25))
                df = discount_curve.df(cf_date) * math.exp(-spread * t_years)
            except Exception:
                df = math.exp(-(0.04 + spread) * t_years)

            # Interest
            interest = tr_balance * tranche.coupon / 12.0
            tr_pv += interest * df

            # Principal: sequential pay (senior gets all principal first)
            avail_prin = pool_principal[m] if m < len(pool_principal) else 0
            prin_to_tranche = min(avail_prin, tr_balance)
            pool_principal[m] = avail_prin - prin_to_tranche  # remaining for junior

            tr_pv += prin_to_tranche * df
            tr_wal_num += prin_to_tranche * t_years
            tr_total_prin += prin_to_tranche
            tr_balance -= prin_to_tranche

            # Losses absorbed by junior first
            loss = pool_losses[m] if m < len(pool_losses) else 0
            if tranche.seniority == max(t.seniority for t in sorted_tranches):
                cum_loss += loss

        wal = tr_wal_num / tr_total_prin if tr_total_prin > 0 else 0
        price = tr_pv / tranche.notional * 100 if tranche.notional > 0 else 0
        el_pct = cum_loss / tranche.notional * 100 if tranche.notional > 0 else 0

        # Break-even: loss rate that wipes this tranche's subordination
        be_loss = (sub_below / pool.original_balance) if pool.original_balance > 0 else 0

        results.append(ABSResult(
            price=price,
            yield_to_maturity=tranche.coupon + spread,
            wal=wal,
            credit_enhancement_pct=ce_pct,
            expected_loss_pct=el_pct,
            break_even_loss_rate=be_loss,
        ))

    return results


# ═══════════════════════════════════════════════════════════════
# Credit Card ABS
# ═══════════════════════════════════════════════════════════════

@dataclass
class CreditCardABSResult:
    """Credit card ABS pricing result."""
    price: float
    yield_spread: float
    wal: float
    excess_spread_pct: float    # portfolio yield - coupon - losses
    mpr: float                  # monthly payment rate
    charge_off_rate: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def price_credit_card_abs(
    pool_balance: float,
    portfolio_yield: float,
    mpr: float,
    charge_off_rate: float,
    tranche_coupon: float,
    revolving_months: int = 24,
    amort_months: int = 12,
    discount_curve: DiscountCurve | None = None,
    spread: float = 0.0,
    rate: float = 0.04,
) -> CreditCardABSResult:
    """Price credit card ABS with revolving + controlled amortisation.

    During revolving period: principal payments are reinvested.
    During controlled amortisation: principal pays down investor notes.

    Args:
        pool_balance: investor's share of pool.
        portfolio_yield: annual portfolio yield (finance charges / receivables).
        mpr: monthly payment rate (% of balance paid by cardholders).
        charge_off_rate: annual charge-off rate.
        tranche_coupon: investor coupon rate (annual).
        revolving_months: length of revolving period.
        amort_months: length of controlled amortisation.
        discount_curve: optional risk-free curve.
        spread: spread over curve.
        rate: flat rate if no curve provided.
    """
    total_months = revolving_months + amort_months
    balance = pool_balance
    pv = 0.0
    wal_num = 0.0
    total_prin = 0.0

    for m in range(1, total_months + 1):
        if balance < 0.01:
            break

        t_years = m / 12.0
        if discount_curve is not None:
            try:
                ref = discount_curve.reference_date
                cf_date = ref + timedelta(days=round(t_years * 365.25))
                df = discount_curve.df(cf_date) * math.exp(-spread * t_years)
            except Exception:
                df = math.exp(-(rate + spread) * t_years)
        else:
            df = math.exp(-(rate + spread) * t_years)

        # Interest to investor
        interest = balance * tranche_coupon / 12.0

        # Charge-offs
        monthly_co = balance * charge_off_rate / 12.0

        # Principal collections
        principal_collected = balance * mpr

        if m <= revolving_months:
            # Revolving: principal is reinvested, no paydown
            prin_to_investor = 0.0
        else:
            # Controlled amortisation: 1/amort_months of original balance
            target = pool_balance / amort_months
            prin_to_investor = min(target, balance)

        pv += (interest + prin_to_investor) * df
        wal_num += prin_to_investor * t_years
        total_prin += prin_to_investor

        if m > revolving_months:
            balance -= prin_to_investor
        balance -= monthly_co
        balance = max(balance, 0)

    price = pv / pool_balance * 100 if pool_balance > 0 else 0
    wal = wal_num / total_prin if total_prin > 0 else revolving_months / 12.0

    excess_spread = portfolio_yield - tranche_coupon - charge_off_rate

    return CreditCardABSResult(
        price=price,
        yield_spread=spread,
        wal=wal,
        excess_spread_pct=excess_spread * 100,
        mpr=mpr,
        charge_off_rate=charge_off_rate,
    )


# ═══════════════════════════════════════════════════════════════
# Student Loan ABS
# ═══════════════════════════════════════════════════════════════

@dataclass
class StudentLoanABSResult:
    """Student loan ABS pricing result."""
    price: float
    wal: float
    grace_months: int
    default_rate: float
    idr_pct: float          # % in income-driven repayment

    def to_dict(self) -> dict:
        return dict(vars(self))


def price_student_loan_abs(
    pool_balance: float,
    wac: float,
    maturity_months: int,
    grace_months: int = 6,
    default_rate: float = 0.05,
    idr_pct: float = 0.30,
    tranche_coupon: float = 0.04,
    discount_curve: DiscountCurve | None = None,
    spread: float = 0.0,
    rate: float = 0.04,
) -> StudentLoanABSResult:
    """Price student loan ABS.

    Student loans have grace periods (no payment after graduation),
    income-driven repayment (IDR, lower payments), and higher default
    rates for subprime borrowers.

    Args:
        grace_months: months of no payment after origination.
        default_rate: annual default rate.
        idr_pct: % of pool in income-driven repayment (lower payment).
    """
    balance = pool_balance
    monthly_rate = wac / 12.0
    pv = 0.0
    wal_num = 0.0
    total_prin = 0.0

    for m in range(1, maturity_months + 1):
        if balance < 0.01:
            break

        t_years = m / 12.0
        if discount_curve is not None:
            try:
                ref = discount_curve.reference_date
                cf_date = ref + timedelta(days=round(t_years * 365.25))
                df = discount_curve.df(cf_date) * math.exp(-spread * t_years)
            except Exception:
                df = math.exp(-(rate + spread) * t_years)
        else:
            df = math.exp(-(rate + spread) * t_years)

        # Grace period: interest accrues, no payments
        if m <= grace_months:
            balance *= (1 + monthly_rate)  # capitalise interest
            continue

        # Defaults
        monthly_default = balance * default_rate / 12.0

        # Payment: standard + IDR fraction pays less
        remaining = maturity_months - m + 1
        if monthly_rate > 0 and remaining > 0:
            full_payment = balance * monthly_rate / (1 - (1 + monthly_rate) ** (-remaining))
        else:
            full_payment = balance / max(remaining, 1)

        # IDR borrowers pay less (50% of scheduled)
        adj_payment = full_payment * (1 - idr_pct * 0.5)

        interest = balance * monthly_rate
        principal = max(adj_payment - interest, 0)
        principal = min(principal, balance - monthly_default)

        pv += (interest * (tranche_coupon / wac) + principal) * df
        wal_num += principal * t_years
        total_prin += principal

        balance -= (principal + monthly_default)
        balance = max(balance, 0)

    price = pv / pool_balance * 100 if pool_balance > 0 else 0
    wal = wal_num / total_prin if total_prin > 0 else maturity_months / 12.0

    return StudentLoanABSResult(
        price=price,
        wal=wal,
        grace_months=grace_months,
        default_rate=default_rate,
        idr_pct=idr_pct,
    )
