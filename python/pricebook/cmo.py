"""Collateralised Mortgage Obligation (CMO) tranches.

CMO structures redirect mortgage pool cashflows through a waterfall:
- Sequential: senior tranche receives all principal until paid off, then next.
- PAC: Planned Amortization Class — protected from prepayment within a band.
- TAC: Targeted Amortization Class — protected at one prepayment speed.
- IO/PO: Interest-Only and Principal-Only strips.
- Z-bond: accrual tranche — receives no cash until all prior tranches retire.

    from pricebook.cmo import CMOPool, sequential_cmo, pac_cmo, io_po_strip

References:
    Fabozzi, *Handbook of Mortgage-Backed Securities*, McGraw-Hill, 2016.
    Hayre, *Salomon Smith Barney Guide to MBS*, Wiley, 2001.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pricebook.amortising_bond import cpr_to_smm, psa_schedule


@dataclass
class CMOTranche:
    """A single CMO tranche."""
    name: str
    face: float
    coupon_rate: float
    tranche_type: str  # "sequential", "PAC", "TAC", "Z", "IO", "PO"


@dataclass
class TrancheCashflow:
    """Monthly cashflow for a tranche."""
    month: int
    principal: float
    interest: float
    balance: float


@dataclass
class CMOResult:
    """CMO structuring result."""
    tranches: dict[str, list[TrancheCashflow]]
    pool_balance: np.ndarray
    total_principal: dict[str, float]
    average_life: dict[str, float]
    prices: dict[str, float]


def _pool_cashflows(
    pool_balance: float,
    wac: float,
    n_months: int,
    psa_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate mortgage pool cashflows with prepayment.

    Returns (scheduled_principal, prepayment, interest) arrays.
    """
    smm = psa_schedule(psa_speed / 100.0, n_months)
    r = wac / 12.0
    balance = float(pool_balance)

    sched_principal = np.zeros(n_months)
    prepay = np.zeros(n_months)
    interest = np.zeros(n_months)

    remaining_months = n_months
    if r > 0:
        level = balance * r / (1 - (1 + r) ** (-remaining_months))
    else:
        level = balance / remaining_months

    for m in range(n_months):
        if balance < 1e-6:
            break
        int_pmt = balance * r
        sch_prin = min(level - int_pmt, balance)
        after_sched = balance - sch_prin
        pre = smm[m] * after_sched

        sched_principal[m] = sch_prin
        prepay[m] = pre
        interest[m] = int_pmt
        balance = after_sched - pre

    return sched_principal, prepay, interest


def sequential_cmo(
    pool_balance: float,
    wac: float,
    n_months: int,
    psa_speed: float,
    tranches: list[CMOTranche],
    discount_rate: float,
) -> CMOResult:
    """Sequential-pay CMO: principal flows to tranches in order.

    The first tranche receives all principal until retired, then the
    second tranche, and so on. Each tranche receives interest on its
    outstanding balance.
    """
    sched_p, prepay_p, pool_int = _pool_cashflows(pool_balance, wac, n_months, psa_speed)
    total_principal = sched_p + prepay_p

    balances = {t.name: float(t.face) for t in tranches}
    result_cfs: dict[str, list[TrancheCashflow]] = {t.name: [] for t in tranches}
    total_prin: dict[str, float] = {t.name: 0.0 for t in tranches}

    for m in range(n_months):
        remaining_principal = float(total_principal[m])
        for t in tranches:
            bal = balances[t.name]
            if bal < 1e-6:
                result_cfs[t.name].append(TrancheCashflow(m, 0.0, 0.0, 0.0))
                continue
            # Interest on current balance
            int_pmt = bal * t.coupon_rate / 12.0
            # Principal: sequential allocation
            prin = min(remaining_principal, bal)
            remaining_principal -= prin
            balances[t.name] = bal - prin
            total_prin[t.name] += prin
            result_cfs[t.name].append(TrancheCashflow(m, prin, int_pmt, balances[t.name]))

    # Compute average life and price for each tranche
    avg_life = {}
    prices = {}
    for t in tranches:
        cfs = result_cfs[t.name]
        tp = total_prin[t.name]
        if tp > 1e-10:
            avg_life[t.name] = sum(cf.principal * (cf.month + 1) / 12.0 for cf in cfs) / tp
        else:
            avg_life[t.name] = 0.0
        pv = sum(
            (cf.principal + cf.interest) * math.exp(-discount_rate * (cf.month + 1) / 12.0)
            for cf in cfs
        )
        prices[t.name] = pv

    pool_bal = np.cumsum(total_principal)
    pool_bal = pool_balance - pool_bal

    return CMOResult(
        tranches=result_cfs,
        pool_balance=pool_bal,
        total_principal=total_prin,
        average_life=avg_life,
        prices=prices,
    )


def io_po_strip(
    pool_balance: float,
    wac: float,
    n_months: int,
    psa_speed: float,
    discount_rate: float,
) -> tuple[float, float]:
    """IO/PO strip pricing.

    IO strip receives all interest payments.
    PO strip receives all principal payments.

    Returns (io_price, po_price).
    """
    sched_p, prepay_p, interest = _pool_cashflows(pool_balance, wac, n_months, psa_speed)
    total_principal = sched_p + prepay_p

    io_pv = sum(
        float(interest[m]) * math.exp(-discount_rate * (m + 1) / 12.0)
        for m in range(n_months)
    )
    po_pv = sum(
        float(total_principal[m]) * math.exp(-discount_rate * (m + 1) / 12.0)
        for m in range(n_months)
    )
    return io_pv, po_pv


def pac_schedule(
    pool_balance: float,
    wac: float,
    n_months: int,
    low_psa: float,
    high_psa: float,
) -> np.ndarray:
    """PAC tranche principal schedule.

    The PAC band is defined by two PSA speeds. The PAC principal each
    month is the minimum of scheduled principal at both speeds.
    This creates a stable cashflow within the band.

    Returns monthly PAC principal allocation.
    """
    _, prepay_low, _ = _pool_cashflows(pool_balance, wac, n_months, low_psa)
    sched_low, _, _ = _pool_cashflows(pool_balance, wac, n_months, low_psa)
    total_low = sched_low + prepay_low

    _, prepay_high, _ = _pool_cashflows(pool_balance, wac, n_months, high_psa)
    sched_high, _, _ = _pool_cashflows(pool_balance, wac, n_months, high_psa)
    total_high = sched_high + prepay_high

    return np.minimum(total_low, total_high)
