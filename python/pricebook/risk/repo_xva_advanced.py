"""Repo XVA with counterparty-collateral correlation.

Joint MC simulation of counterparty default + collateral value drop
for fully correlated repo XVA.

    from pricebook.risk.repo_xva_advanced import (
        repo_xva_correlated, repo_all_in_xva, RepoXVACorrelatedResult,
    )

References:
    Brigo, Morini & Pallavicini (2013). Counterparty Credit Risk, Ch 8.
    Gregory (2015). The xVA Challenge, Ch 14.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class RepoXVACorrelatedResult:
    """Fully correlated repo XVA."""
    cva: float
    fva: float
    kva: float
    mva: float
    gap_cost: float
    total_xva: float
    wrong_way_cva: float         # CVA with WWR
    independent_cva: float       # CVA without WWR
    correlation_impact: float    # wrong_way - independent
    n_paths: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def repo_xva_correlated(
    repo_notional: float,
    repo_days: int,
    repo_rate: float,
    haircut: float,
    counterparty_hazard: float,
    counterparty_recovery: float,
    collateral_spread_bp: float,
    collateral_duration: float,
    collateral_spread_vol: float = 0.30,
    correlation: float = 0.30,
    funding_spread: float = 0.005,
    capital_rate: float = 0.08,
    n_paths: int = 10_000,
    seed: int = 42,
) -> RepoXVACorrelatedResult:
    """Joint MC of counterparty default + collateral value drop.

    Simulates correlated paths:
    - Counterparty default time from exponential(hazard)
    - Collateral spread from GBM (correlated with counterparty factor)
    - Exposure = max(0, cash_lent - collateral_value(t))

    CVA = E[exposure_at_default × LGD | counterparty defaults]
    """
    rng = np.random.default_rng(seed)
    dt = repo_days / 365.0
    lgd_cp = 1.0 - counterparty_recovery

    # Counterparty default times
    U = rng.uniform(0, 1, n_paths)
    tau_cp = -np.log(U) / counterparty_hazard  # exponential default time

    # Correlated collateral spread paths
    Z_cp = norm.ppf(U)  # standard normal for counterparty
    Z_indep = rng.standard_normal(n_paths)
    Z_coll = correlation * Z_cp + math.sqrt(1 - correlation**2) * Z_indep

    # Collateral spread at default time (or maturity)
    eval_time = np.minimum(tau_cp, dt)
    spread_change = collateral_spread_vol * collateral_spread_bp / 10_000 * Z_coll * np.sqrt(eval_time)
    collateral_price_change = -collateral_duration * spread_change  # price = -dur × Δspread

    # Exposure at evaluation time
    collateral_value = repo_notional * (1 + haircut) * (1 + collateral_price_change)
    exposure = np.maximum(repo_notional - collateral_value, 0)

    # CVA: only paths where counterparty defaults before maturity
    defaults = tau_cp <= dt
    cva_paths = exposure * lgd_cp * defaults
    cva = float(np.mean(cva_paths))

    # Independent CVA (correlation = 0)
    Z_indep2 = rng.standard_normal(n_paths)
    spread_change_indep = collateral_spread_vol * collateral_spread_bp / 10_000 * Z_indep2 * np.sqrt(eval_time)
    price_change_indep = -collateral_duration * spread_change_indep
    coll_val_indep = repo_notional * (1 + haircut) * (1 + price_change_indep)
    exp_indep = np.maximum(repo_notional - coll_val_indep, 0)
    cva_indep = float(np.mean(exp_indep * lgd_cp * defaults))

    # FVA: funding cost on expected exposure
    avg_exposure = float(np.mean(exposure))
    fva = avg_exposure * funding_spread * dt

    # KVA: capital cost
    kva = avg_exposure * capital_rate * 0.08 * dt  # 8% RW × hurdle

    # MVA: margin funding cost (simplified as haircut × funding)
    mva = repo_notional * haircut * funding_spread * dt

    # Gap cost: probability of gap × expected loss given gap
    gap_prob = float(np.mean(exposure > 0))
    gap_cost = gap_prob * float(np.mean(exposure[exposure > 0])) * lgd_cp if gap_prob > 0 else 0.0

    total = cva + fva + kva + mva + gap_cost

    return RepoXVACorrelatedResult(
        cva=cva,
        fva=fva,
        kva=kva,
        mva=mva,
        gap_cost=gap_cost,
        total_xva=total,
        wrong_way_cva=cva,
        independent_cva=cva_indep,
        correlation_impact=cva - cva_indep,
        n_paths=n_paths,
    )


def repo_all_in_xva(
    repo_notional: float,
    repo_days: int,
    repo_rate: float,
    haircut: float,
    counterparty_hazard: float,
    counterparty_recovery: float = 0.40,
    collateral_spread_bp: float = 100.0,
    collateral_duration: float = 5.0,
    correlation: float = 0.30,
    funding_spread: float = 0.005,
) -> dict:
    """All-in repo XVA: interest income vs total XVA cost.

    Returns profitability analysis.
    """
    denom = 360.0
    t = repo_days / denom
    interest = repo_notional * repo_rate * t

    xva = repo_xva_correlated(
        repo_notional, repo_days, repo_rate, haircut,
        counterparty_hazard, counterparty_recovery,
        collateral_spread_bp, collateral_duration,
        correlation=correlation, funding_spread=funding_spread,
        n_paths=5000,
    )

    net = interest - xva.total_xva
    breakeven = xva.total_xva / (repo_notional * t) if repo_notional * t > 0 else 0.0

    return {
        "interest_income": interest,
        "total_xva": xva.total_xva,
        "cva": xva.cva,
        "fva": xva.fva,
        "kva": xva.kva,
        "mva": xva.mva,
        "gap_cost": xva.gap_cost,
        "net_income": net,
        "breakeven_rate": breakeven,
        "correlation_impact": xva.correlation_impact,
        "profitable": net > 0,
    }
