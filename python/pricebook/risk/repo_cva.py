"""Repo CVA with wrong-way risk.

CVA on the unsecured exposure of a repo, with three wrong-way channels:
1. Counterparty = collateral issuer (classic)
2. Counterparty + collateral in same sector/country (systemic)
3. Collateral credit deterioration → margin call → gap risk

    from pricebook.risk.repo_cva import (
        repo_cva, repo_wrong_way_risk, repo_bilateral_cva, RepoCVAResult,
    )

References:
    Gregory (2015). The xVA Challenge, Ch 12.
    Pykhtin & Zhu (2007). A Guide to Modelling Counterparty Credit Risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class RepoCVAResult:
    """Result of repo CVA computation."""
    cva: float                   # credit value adjustment (positive = cost)
    dva: float                   # debit value adjustment
    bilateral_cva: float         # CVA - DVA
    wrong_way_add_on: float      # additional CVA from wrong-way risk
    expected_exposure: float     # average positive exposure
    peak_exposure: float         # maximum exposure across time grid
    counterparty_pd: float       # cumulative default probability
    n_time_steps: int

    def to_dict(self) -> dict:
        return vars(self)


def repo_cva(
    repo_notional: float,
    repo_days: int,
    repo_rate: float,
    haircut: float,
    counterparty_hazard: float,
    counterparty_recovery: float = 0.40,
    collateral_vol: float = 0.02,
    n_steps: int = 20,
) -> RepoCVAResult:
    """Compute repo CVA on unsecured exposure after haircut.

    The exposure at time t is:
        E(t) = max(0, cash_lent - collateral_value(t))
        ≈ max(0, notional - notional × (1 + haircut) × (1 + ΔV))

    where ΔV is the collateral price change.

    In a properly margined repo, exposure is small (only the haircut gap).
    CVA = ∫ EPE(t) × h(t) × (1-R) × df(t) dt

    Args:
        repo_notional: cash amount lent.
        repo_days: term in days.
        repo_rate: agreed repo rate.
        haircut: collateral haircut (fraction, e.g. 0.02 = 2%).
        counterparty_hazard: annual hazard rate.
        counterparty_recovery: counterparty recovery on default.
        collateral_vol: daily collateral price volatility.
        n_steps: time grid steps.
    """
    dt = repo_days / 365.0 / n_steps
    lgd = 1.0 - counterparty_recovery

    # Overcollateralisation: collateral = notional × (1 + haircut)
    collateral_value = repo_notional * (1 + haircut)

    cva = 0.0
    epe_sum = 0.0
    peak_exposure = 0.0

    for i in range(1, n_steps + 1):
        t = i * dt
        t_years = t

        # Survival to this step
        q = math.exp(-counterparty_hazard * t_years * 365.0 / 365.0)
        q_prev = math.exp(-counterparty_hazard * (t_years - dt) * 365.0 / 365.0)
        pd_step = q_prev - q  # marginal PD

        # Expected positive exposure: driven by collateral price decline
        # P(collateral drops below cash_lent) = P(ΔV < -haircut)
        # ΔV ~ N(0, vol × √t)
        daily_vol = collateral_vol
        period_vol = daily_vol * math.sqrt(i)

        if period_vol > 0:
            # Expected exposure above threshold
            # EPE ≈ notional × [vol×√t × φ(h/(vol√t)) - h × Φ(-h/(vol√t))]
            # where h = haircut, φ = pdf, Φ = cdf
            h_norm = haircut / period_vol if period_vol > 0 else float('inf')
            epe = repo_notional * (
                period_vol * norm.pdf(h_norm) - haircut * norm.cdf(-h_norm)
            )
        else:
            epe = 0.0

        epe = max(epe, 0.0)
        epe_sum += epe
        peak_exposure = max(peak_exposure, epe)

        # CVA contribution
        cva += epe * pd_step * lgd

    avg_epe = epe_sum / n_steps if n_steps > 0 else 0.0
    total_pd = 1.0 - math.exp(-counterparty_hazard * repo_days / 365.0)

    return RepoCVAResult(
        cva=cva,
        dva=0.0,
        bilateral_cva=cva,
        wrong_way_add_on=0.0,
        expected_exposure=avg_epe,
        peak_exposure=peak_exposure,
        counterparty_pd=total_pd,
        n_time_steps=n_steps,
    )


def repo_wrong_way_risk(
    base_cva: float,
    correlation_channel: str,
    correlation: float,
    collateral_hazard: float = 0.0,
    counterparty_hazard: float = 0.0,
) -> float:
    """Wrong-way risk add-on for repo CVA.

    Three channels:
    1. "issuer" — counterparty = collateral issuer (classic WWR)
       Add-on = base_CVA × ρ × (h_collateral / h_counterparty)
    2. "sector" — same sector/country (systemic WWR)
       Add-on = base_CVA × ρ × 0.5
    3. "spiral" — credit deterioration → margin call → can't post
       Add-on = base_CVA × ρ² × 2.0

    Args:
        base_cva: CVA without wrong-way risk.
        correlation_channel: "issuer", "sector", or "spiral".
        correlation: correlation between counterparty and collateral.
        collateral_hazard: collateral issuer hazard rate.
        counterparty_hazard: counterparty hazard rate.
    """
    if abs(correlation) < 1e-10:
        return 0.0

    if correlation_channel == "issuer":
        # Classic: counterparty IS the issuer
        ratio = collateral_hazard / max(counterparty_hazard, 1e-10)
        return base_cva * abs(correlation) * min(ratio, 5.0)
    elif correlation_channel == "sector":
        # Systemic: same sector/country
        return base_cva * abs(correlation) * 0.5
    elif correlation_channel == "spiral":
        # Margin spiral: non-linear
        return base_cva * correlation ** 2 * 2.0
    else:
        return 0.0


def repo_bilateral_cva(
    repo_notional: float,
    repo_days: int,
    repo_rate: float,
    haircut: float,
    counterparty_hazard: float,
    counterparty_recovery: float,
    own_hazard: float,
    own_recovery: float,
    collateral_vol: float = 0.02,
    correlation: float = 0.0,
    collateral_hazard: float = 0.0,
) -> RepoCVAResult:
    """Bilateral repo CVA: CVA - DVA + wrong-way risk.

    Args:
        own_hazard: our own hazard rate (for DVA).
        own_recovery: our recovery rate.
        correlation: counterparty-collateral default correlation.
        collateral_hazard: collateral issuer hazard (for WWR).
    """
    # CVA (their default, our loss)
    cva_result = repo_cva(
        repo_notional, repo_days, repo_rate, haircut,
        counterparty_hazard, counterparty_recovery, collateral_vol,
    )

    # DVA (our default, their loss) — symmetric
    dva_result = repo_cva(
        repo_notional, repo_days, repo_rate, haircut,
        own_hazard, own_recovery, collateral_vol,
    )

    # Wrong-way risk
    wwr = 0.0
    if abs(correlation) > 0.01:
        if collateral_hazard > 0 and abs(collateral_hazard - counterparty_hazard) < 1e-6:
            wwr = repo_wrong_way_risk(cva_result.cva, "issuer", correlation,
                                       collateral_hazard, counterparty_hazard)
        elif correlation > 0.3:
            wwr = repo_wrong_way_risk(cva_result.cva, "sector", correlation)
        if correlation > 0.5:
            wwr += repo_wrong_way_risk(cva_result.cva, "spiral", correlation)

    total_cva = cva_result.cva + wwr
    bilateral = total_cva - dva_result.cva

    return RepoCVAResult(
        cva=total_cva,
        dva=dva_result.cva,
        bilateral_cva=bilateral,
        wrong_way_add_on=wwr,
        expected_exposure=cva_result.expected_exposure,
        peak_exposure=cva_result.peak_exposure,
        counterparty_pd=cva_result.counterparty_pd,
        n_time_steps=cva_result.n_time_steps,
    )
