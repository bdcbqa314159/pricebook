"""Recovery-locked CDS and Loan CDS (LCDS).

Recovery-locked CDS: fixed recovery eliminates auction uncertainty.
LCDS: CDS on loans with higher recovery (70-80%) and prepayment cancellation.

* :class:`RecoveryLockedCDSResult` — pricing result with recovery lock premium.
* :func:`price_recovery_locked_cds` — recovery-locked CDS pricing.
* :func:`recovery_lock_premium` — premium for locking recovery.
* :class:`LCDSResult` — Loan CDS pricing result.
* :func:`price_lcds` — LCDS with prepayment cancellation.

References:
    O'Kane, *Modelling Single-name and Multi-name Credit Derivatives*,
    Ch. 5 (Recovery), Ch. 14 (LCDS), 2008.
    Markit, *Loan CDS Primer*, 2007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.core.day_count import DayCountConvention, year_fraction


# ═══════════════════════════════════════════════════════════════
# Recovery-Locked CDS
# ═══════════════════════════════════════════════════════════════

@dataclass
class RecoveryLockedCDSResult:
    """Recovery-locked CDS result."""
    pv: float
    par_spread: float
    locked_recovery: float
    market_recovery: float
    recovery_lock_premium_bp: float
    rpv01: float

    def to_dict(self) -> dict:
        return vars(self)


def recovery_lock_premium(
    market_spread: float,
    locked_recovery: float,
    market_recovery: float = 0.4,
) -> float:
    """Premium for locking recovery at a different level.

    When locked_recovery > market_recovery, the protection buyer
    receives less on default → lower spread → negative premium.

    premium ≈ market_spread × (1 − locked_recovery) / (1 − market_recovery) − market_spread

    Args:
        market_spread: standard CDS par spread.
        locked_recovery: fixed recovery in the recovery-locked CDS.
        market_recovery: assumed recovery in market CDS (typically 40%).
    """
    if market_recovery >= 1.0:
        return 0.0
    ratio = (1 - locked_recovery) / (1 - market_recovery)
    return market_spread * (ratio - 1)


def price_recovery_locked_cds(
    reference_date: date,
    maturity_years: float,
    market_spread: float,
    locked_recovery: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    market_recovery: float = 0.4,
    notional: float = 10_000_000.0,
    coupon_frequency: int = 4,
) -> RecoveryLockedCDSResult:
    """Price a recovery-locked CDS.

    The locked recovery eliminates auction uncertainty. The protection
    payment on default is notional × (1 − locked_recovery), regardless
    of what the auction determines.

    Args:
        reference_date: valuation date.
        maturity_years: CDS maturity in years.
        market_spread: market CDS par spread (for comparison).
        locked_recovery: fixed recovery rate.
        discount_curve: risk-free discount curve.
        survival_curve: credit survival curve.
        market_recovery: assumed recovery in standard CDS.
        notional: CDS notional.
        coupon_frequency: premium payments per year.
    """
    dt = 1.0 / coupon_frequency
    n_periods = int(maturity_years * coupon_frequency)

    # Protection leg with locked recovery
    prot_pv = 0.0
    prem_pv = 0.0

    prev_date = reference_date
    prev_q = 1.0

    for i in range(1, n_periods + 1):
        t = i * dt
        t_date = reference_date + timedelta(days=round(t * 365.25))

        q = survival_curve.survival(t_date)
        df = discount_curve.df(t_date)
        default_prob = max(prev_q - q, 0)

        prot_pv += (1 - locked_recovery) * notional * default_prob * df
        prem_pv += dt * q * df  # RPV01 contribution

        prev_q = q

    rpv01 = prem_pv
    par_spread = prot_pv / (notional * rpv01) if rpv01 > 0 else 0.0

    # Recovery lock premium
    lock_premium = recovery_lock_premium(market_spread, locked_recovery, market_recovery)

    return RecoveryLockedCDSResult(
        pv=prot_pv - market_spread * notional * rpv01,
        par_spread=par_spread,
        locked_recovery=locked_recovery,
        market_recovery=market_recovery,
        recovery_lock_premium_bp=lock_premium * 10_000,
        rpv01=rpv01,
    )


# ═══════════════════════════════════════════════════════════════
# Loan CDS (LCDS)
# ═══════════════════════════════════════════════════════════════

@dataclass
class LCDSResult:
    """Loan CDS pricing result."""
    pv: float
    par_spread: float
    recovery: float
    prepayment_rate: float
    effective_maturity: float
    cancellation_value: float
    rpv01: float

    def to_dict(self) -> dict:
        return vars(self)


def price_lcds(
    reference_date: date,
    maturity_years: float,
    spread: float,
    discount_curve: DiscountCurve,
    survival_curve: SurvivalCurve,
    recovery: float = 0.70,
    prepayment_rate: float = 0.15,
    notional: float = 10_000_000.0,
    coupon_frequency: int = 4,
) -> LCDSResult:
    """Price a Loan CDS with prepayment cancellation.

    LCDS differs from standard CDS:
    1. Higher recovery (70-80% for loans vs 40% for bonds).
    2. Cancellable on loan prepayment — LCDS terminates if the
       underlying loan is prepaid (no protection, no premium).
    3. Typically on first-lien secured debt.

    The prepayment acts as an additional survival factor: the LCDS
    terminates early if either (a) default occurs or (b) loan prepays.

    Joint survival: Q_joint(t) = Q_credit(t) × Q_prepay(t)
    where Q_prepay(t) = exp(−CPR × t).

    Args:
        recovery: loan recovery (typically 0.70-0.80).
        prepayment_rate: constant prepayment rate (CPR), annualised.
    """
    dt = 1.0 / coupon_frequency
    n_periods = int(maturity_years * coupon_frequency)

    prot_pv = 0.0
    prem_pv = 0.0
    prem_pv_no_prepay = 0.0

    prev_q_credit = 1.0
    prev_q_prepay = 1.0

    for i in range(1, n_periods + 1):
        t = i * dt
        t_date = reference_date + timedelta(days=round(t * 365.25))

        q_credit = survival_curve.survival(t_date)
        q_prepay = math.exp(-prepayment_rate * t)
        q_joint = q_credit * q_prepay

        df = discount_curve.df(t_date)

        # Default probability conditional on no prior prepayment
        prev_q_joint = prev_q_credit * prev_q_prepay
        default_prob_joint = max(prev_q_joint - q_joint, 0)
        # Isolate credit default: default given no prepayment
        credit_default = max(prev_q_credit - q_credit, 0) * prev_q_prepay

        prot_pv += (1 - recovery) * notional * credit_default * df
        prem_pv += dt * q_joint * df
        prem_pv_no_prepay += dt * q_credit * df

        prev_q_credit = q_credit
        prev_q_prepay = q_prepay

    rpv01 = prem_pv
    par_spread = prot_pv / (notional * rpv01) if rpv01 > 0 else 0.0

    # Cancellation value: difference in RPV01 with/without prepayment
    cancellation = spread * notional * (prem_pv_no_prepay - prem_pv)

    # Effective maturity (duration-weighted)
    if prem_pv > 0:
        eff_mat_num = 0.0
        for i in range(1, n_periods + 1):
            t = i * dt
            t_date = reference_date + timedelta(days=round(t * 365.25))
            q_c = survival_curve.survival(t_date)
            q_p = math.exp(-prepayment_rate * t)
            df = discount_curve.df(t_date)
            eff_mat_num += t * dt * q_c * q_p * df
        eff_mat = eff_mat_num / prem_pv
    else:
        eff_mat = maturity_years

    return LCDSResult(
        pv=prot_pv - spread * notional * rpv01,
        par_spread=par_spread,
        recovery=recovery,
        prepayment_rate=prepayment_rate,
        effective_maturity=eff_mat,
        cancellation_value=cancellation,
        rpv01=rpv01,
    )
