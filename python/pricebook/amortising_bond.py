"""Amortising and sinker bonds: schedules, prepayment, average life, DV01.

* :class:`AmortisingBond` — mortgage-style / sinking fund bonds.
* :func:`cpr_to_smm` / :func:`psa_schedule` — prepayment models.
* :func:`prepayment_bond_price` — bond with prepayment model.
* :func:`average_life` / :func:`weighted_average_maturity` — portfolio measures.

References:
    Fabozzi, *Handbook of Mortgage-Backed Securities*, 7th ed., McGraw-Hill, 2016.
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012.
    Hayre, *Salomon Smith Barney Guide to Mortgage-Backed and Asset-Backed
    Securities*, Wiley, 2001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Amortising bond ----

@dataclass
class AmortisingBondResult:
    """Amortising bond pricing result."""
    price: float
    average_life: float             # principal-weighted time to repayment
    duration: float                 # Macaulay duration
    dv01: float                     # DV01 (bump-and-reprice)
    schedule: list[tuple[float, float, float]]  # (time, principal, interest)


@dataclass
class AmortisingBond:
    """Bond with amortising principal schedule.

    Two common structures:
    - Mortgage-style: level payment, amortising principal.
    - Sinking fund: principal reduced by fixed amounts.

    Args:
        notional: initial face value.
        coupon_rate: annual coupon (on outstanding balance).
        maturity_years: time to final payment.
        n_payments: total number of payments.
        amortisation_type: "mortgage" (level payment) or "linear" (equal principal).
    """
    notional: float
    coupon_rate: float
    maturity_years: float
    n_payments: int
    amortisation_type: str = "mortgage"

    def schedule(self) -> list[tuple[float, float, float]]:
        """Return payment schedule: [(time, principal_paid, interest_paid), ...]."""
        dt = self.maturity_years / self.n_payments
        r_per_period = self.coupon_rate * dt
        entries = []
        balance = float(self.notional)

        if self.amortisation_type == "mortgage":
            # Level payment: P × r / (1 - (1+r)^-n)
            if r_per_period > 0:
                payment = self.notional * r_per_period / (1 - (1 + r_per_period) ** (-self.n_payments))
            else:
                payment = self.notional / self.n_payments

            for i in range(self.n_payments):
                t = (i + 1) * dt
                interest = balance * r_per_period
                principal = payment - interest
                balance -= principal
                entries.append((t, principal, interest))
        elif self.amortisation_type == "linear":
            # Equal principal each period
            principal_each = self.notional / self.n_payments
            for i in range(self.n_payments):
                t = (i + 1) * dt
                interest = balance * r_per_period
                balance -= principal_each
                entries.append((t, principal_each, interest))
        else:
            raise ValueError(f"Unknown amortisation_type: {self.amortisation_type}")

        return entries

    def price(self, rate: float) -> AmortisingBondResult:
        """Price, duration, DV01 at a flat discount rate."""
        schedule = self.schedule()

        pv = 0.0
        weighted_time = 0.0
        principal_weighted_time = 0.0
        total_principal = 0.0

        for t, principal, interest in schedule:
            cf = principal + interest
            df = math.exp(-rate * t)
            pv += cf * df
            weighted_time += cf * df * t
            principal_weighted_time += principal * t
            total_principal += principal

        duration = weighted_time / max(pv, 1e-10)
        avg_life = principal_weighted_time / max(total_principal, 1e-10)

        # DV01 via 1bp bump
        bumped_pv = 0.0
        for t, principal, interest in schedule:
            cf = principal + interest
            bumped_pv += cf * math.exp(-(rate + 1e-4) * t)
        dv01 = pv - bumped_pv

        return AmortisingBondResult(
            price=float(pv),
            average_life=float(avg_life),
            duration=float(duration),
            dv01=float(dv01),
            schedule=schedule,
        )


# ---- Prepayment models ----

def cpr_to_smm(cpr: float) -> float:
    """Convert Conditional Prepayment Rate (annual) to Single Monthly Mortality.

    SMM = 1 - (1 - CPR)^(1/12)
    """
    if cpr >= 1.0:
        return 1.0
    if cpr <= 0:
        return 0.0
    return 1 - (1 - cpr) ** (1 / 12)


def psa_schedule(psa_speed: float = 1.0, n_months: int = 360) -> np.ndarray:
    """PSA (Public Securities Association) standard prepayment schedule.

    Standard 100% PSA:
    - Months 1-30: CPR rises linearly from 0.2% to 6%.
    - Months 30-360: CPR stays flat at 6%.

    A 200% PSA = 2 × standard CPR.

    Args:
        psa_speed: PSA factor (1.0 = 100% PSA).
        n_months: horizon.

    Returns:
        Monthly SMM values.
    """
    cpr = np.zeros(n_months)
    for m in range(n_months):
        if m < 30:
            cpr[m] = 0.002 + (0.06 - 0.002) * (m + 1) / 30
        else:
            cpr[m] = 0.06
    cpr = cpr * psa_speed
    smm = np.array([cpr_to_smm(c) for c in cpr])
    return smm


@dataclass
class PrepaymentBondResult:
    """Prepayment bond pricing result."""
    price: float
    average_life: float
    projected_prepayments_pct: float   # total principal paid via prepay
    wac: float                          # weighted average coupon
    wam: float                          # weighted average maturity
    n_months: int


def prepayment_bond_price(
    notional: float,
    coupon_rate: float,
    maturity_years: float,
    psa_speed: float,
    discount_rate: float,
    n_payments_per_year: int = 12,
) -> PrepaymentBondResult:
    """Price a bond with PSA prepayment model.

    Each period:
    - Scheduled payment from mortgage amortisation.
    - Additional prepayment: SMM × remaining balance.

    Args:
        psa_speed: PSA factor (e.g., 100 for 100% PSA, 200 for fast).
    """
    n_months = int(maturity_years * n_payments_per_year)
    dt = 1.0 / n_payments_per_year
    r_per_period = coupon_rate * dt

    # Level payment (scheduled)
    if r_per_period > 0:
        level = notional * r_per_period / (1 - (1 + r_per_period) ** (-n_months))
    else:
        level = notional / n_months

    smm_schedule = psa_schedule(psa_speed / 100.0, n_months)

    balance = float(notional)
    total_prepayment = 0.0
    pv = 0.0
    total_principal_weighted_t = 0.0
    total_principal = 0.0

    for m in range(n_months):
        t = (m + 1) * dt
        # Scheduled: interest + principal portion
        interest = balance * r_per_period
        sched_principal = min(level - interest, balance)
        balance_after_sched = balance - sched_principal

        # Prepayment: SMM × balance_after_sched
        smm = smm_schedule[m]
        prepayment = smm * balance_after_sched
        balance_after_all = balance_after_sched - prepayment

        total_cf = interest + sched_principal + prepayment
        df = math.exp(-discount_rate * t)
        pv += total_cf * df

        total_principal_weighted_t += (sched_principal + prepayment) * t
        total_principal += sched_principal + prepayment
        total_prepayment += prepayment

        balance = balance_after_all
        if balance < 1e-6:
            break

    avg_life = total_principal_weighted_t / max(total_principal, 1e-10)
    prepay_pct = total_prepayment / notional
    wac = coupon_rate   # single-coupon bond
    wam = avg_life

    return PrepaymentBondResult(
        price=float(pv),
        average_life=float(avg_life),
        projected_prepayments_pct=float(prepay_pct),
        wac=coupon_rate,
        wam=float(wam),
        n_months=n_months,
    )


# ---- Portfolio measures ----

def average_life(
    cashflows: list[tuple[float, float]],   # list of (time, principal_paid)
) -> float:
    """Principal-weighted average life.

    AL = Σ t_i × P_i / Σ P_i
    """
    total_p = sum(p for _, p in cashflows)
    if total_p < 1e-10:
        return 0.0
    return sum(t * p for t, p in cashflows) / total_p


def weighted_average_maturity(
    bonds: list[tuple[float, float]],   # list of (maturity, weight)
) -> float:
    """Weighted average maturity across a portfolio of bonds."""
    total_w = sum(w for _, w in bonds)
    if total_w < 1e-10:
        return 0.0
    return sum(m * w for m, w in bonds) / total_w


# ---- Sinker vs bullet comparison ----

@dataclass
class SinkerComparisonResult:
    """Sinker vs bullet comparison."""
    sinker_price: float
    bullet_price: float
    sinker_dv01: float
    bullet_dv01: float
    sinker_duration: float
    bullet_duration: float


def sinker_vs_bullet(
    notional: float,
    coupon_rate: float,
    maturity_years: float,
    n_payments: int,
    rate: float,
) -> SinkerComparisonResult:
    """Compare amortising sinker vs bullet bond with same coupon/maturity."""
    sinker = AmortisingBond(notional, coupon_rate, maturity_years, n_payments, "linear")
    sinker_res = sinker.price(rate)

    # Bullet: all principal at maturity, coupon payments in between
    bullet_cfs = []
    dt = maturity_years / n_payments
    for i in range(n_payments):
        t = (i + 1) * dt
        interest = notional * coupon_rate * dt
        principal = notional if i == n_payments - 1 else 0.0
        bullet_cfs.append((t, principal, interest))

    bullet_pv = sum((p + i) * math.exp(-rate * t) for t, p, i in bullet_cfs)
    bullet_pv_bumped = sum((p + i) * math.exp(-(rate + 1e-4) * t) for t, p, i in bullet_cfs)
    bullet_dv01 = bullet_pv - bullet_pv_bumped
    bullet_duration = sum(t * (p + i) * math.exp(-rate * t) for t, p, i in bullet_cfs) / bullet_pv

    return SinkerComparisonResult(
        sinker_price=float(sinker_res.price),
        bullet_price=float(bullet_pv),
        sinker_dv01=float(sinker_res.dv01),
        bullet_dv01=float(bullet_dv01),
        sinker_duration=float(sinker_res.duration),
        bullet_duration=float(bullet_duration),
    )
