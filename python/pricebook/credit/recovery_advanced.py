"""Advanced recovery models: seniority waterfall, LGD cycle, stochastic recovery-intensity.

Phase C8 slices 229-230 consolidated.

* :class:`SeniorityWaterfall` — recovery by priority (senior/sub/equity).
* :func:`lgd_cycle` — pro-cyclical LGD as function of aggregate default rate.
* :func:`stochastic_recovery_cds` — CDS spread with correlated recovery-intensity.
* :func:`wrong_way_recovery_cva` — CVA with recovery falling in stress.

References:
    Altman, *Default Recovery Rates and LGD*, J. Portfolio Management, 2006.
    Schönbucher, *Credit Derivatives Pricing Models*, Ch. 6.
    Andersen & Sidenius, *Extensions to the Gaussian Copula*, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Seniority waterfall ----

@dataclass
class WaterfallResult:
    """Recovery allocation across seniority tranches."""
    senior_recovery: float
    subordinated_recovery: float
    equity_recovery: float
    total_recovery: float


def seniority_waterfall(
    asset_recovery: float,
    senior_face: float,
    subordinated_face: float,
    equity_face: float = 0.0,
) -> WaterfallResult:
    """Allocate recovered assets by seniority priority.

    Senior → subordinated → equity. Each tranche receives its
    face value or what's left, whichever is less.

    Args:
        asset_recovery: total recovered amount.
        senior_face: senior tranche face value.
        subordinated_face: subordinated tranche face value.
        equity_face: equity tranche face value.

    Returns:
        :class:`WaterfallResult` with recovery per tranche.
    """
    remaining = max(asset_recovery, 0.0)

    senior = min(remaining, senior_face)
    remaining -= senior

    sub = min(remaining, subordinated_face)
    remaining -= sub

    equity = min(remaining, equity_face) if equity_face > 0 else remaining

    return WaterfallResult(senior, sub, equity, senior + sub + equity)


def waterfall_recovery_rates(
    asset_recovery_rate: float,
    senior_pct: float = 0.60,
    sub_pct: float = 0.30,
    equity_pct: float = 0.10,
    total_debt: float = 100.0,
) -> WaterfallResult:
    """Recovery rates (per unit face) for each seniority class.

    Args:
        asset_recovery_rate: total recovery as fraction of total debt.
        senior_pct / sub_pct / equity_pct: capital structure.
    """
    recovered = asset_recovery_rate * total_debt
    result = seniority_waterfall(
        recovered,
        senior_pct * total_debt,
        sub_pct * total_debt,
        equity_pct * total_debt,
    )
    return WaterfallResult(
        result.senior_recovery / (senior_pct * total_debt) if senior_pct > 0 else 0,
        result.subordinated_recovery / (sub_pct * total_debt) if sub_pct > 0 else 0,
        result.equity_recovery / (equity_pct * total_debt) if equity_pct > 0 else 0,
        asset_recovery_rate,
    )


# ---- LGD cycle ----

@dataclass
class LGDCycleResult:
    """LGD as a function of aggregate default rate."""
    lgd: float
    recovery: float
    default_rate: float
    regime: str


def lgd_cycle(
    base_lgd: float,
    default_rate: float,
    sensitivity: float = 2.0,
    normal_default_rate: float = 0.02,
    floor_recovery: float = 0.10,
    cap_recovery: float = 0.80,
) -> LGDCycleResult:
    """Pro-cyclical LGD: recovery falls when default rates spike.

    R(t) = R_base − sensitivity × (default_rate − normal_default_rate)
    LGD(t) = 1 − R(t)

    In downturns (high default rate): recovery drops, LGD rises.
    In benign periods: recovery is at or above base level.

    Args:
        base_lgd: base-case LGD (e.g. 0.60).
        default_rate: current aggregate default rate.
        sensitivity: how much recovery drops per unit excess default rate.
        normal_default_rate: long-run average default rate.
        floor_recovery / cap_recovery: bounds on recovery.
    """
    base_recovery = 1 - base_lgd
    excess = default_rate - normal_default_rate
    recovery = base_recovery - sensitivity * excess
    recovery = max(min(recovery, cap_recovery), floor_recovery)
    lgd = 1 - recovery

    if default_rate > normal_default_rate * 1.5:
        regime = "downturn"
    elif default_rate < normal_default_rate * 0.5:
        regime = "benign"
    else:
        regime = "normal"

    return LGDCycleResult(lgd, recovery, default_rate, regime)


# ---- Stochastic recovery correlated with intensity ----

@dataclass
class StochasticRecoveryCDSResult:
    """CDS spread with stochastic recovery correlated to intensity."""
    spread_fixed_recovery: float
    spread_stochastic_recovery: float
    mean_recovery: float
    recovery_std: float
    wrong_way_premium: float


def stochastic_recovery_cds(
    flat_hazard: float,
    base_recovery: float,
    recovery_vol: float,
    hazard_recovery_corr: float,
    maturity_years: int = 5,
    flat_rate: float = 0.05,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> StochasticRecoveryCDSResult:
    """CDS spread with recovery correlated to default intensity.

    Key insight: E[(1−R) × PD] ≠ (1−E[R]) × PD when R and λ are correlated.
    Negative correlation (wrong-way): recovery drops when intensity spikes,
    making actual losses higher than the fixed-recovery model predicts.

    Args:
        recovery_vol: volatility of recovery rate.
        hazard_recovery_corr: correlation (negative = wrong-way risk).
    """
    rng = np.random.default_rng(seed)

    # Fixed-recovery CDS spread: λ(1−R)
    spread_fixed = flat_hazard * (1 - base_recovery)

    # Simulate joint (λ, R) paths
    dt = 1.0
    n_periods = maturity_years
    annuity = 0.0
    protection = 0.0

    for i in range(1, n_periods + 1):
        t = float(i)
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * (t - 1))
        default_prob = surv_prev - surv

        # Sample correlated recovery for this period
        z1 = rng.standard_normal(n_paths)  # hazard factor
        z2 = rng.standard_normal(n_paths)  # recovery factor
        z_r = hazard_recovery_corr * z1 + math.sqrt(1 - hazard_recovery_corr**2) * z2

        R = base_recovery + recovery_vol * z_r
        R = np.clip(R, 0.0, 1.0)

        # E[(1-R)] under correlation
        mean_lgd = float((1 - R).mean())
        protection += df * default_prob * mean_lgd
        annuity += df * surv * dt

    spread_stochastic = protection / annuity if annuity > 0 else 0.0

    # Recovery statistics
    z_r_sample = hazard_recovery_corr * rng.standard_normal(n_paths) + \
                 math.sqrt(1 - hazard_recovery_corr**2) * rng.standard_normal(n_paths)
    R_sample = np.clip(base_recovery + recovery_vol * z_r_sample, 0, 1)

    return StochasticRecoveryCDSResult(
        spread_fixed, spread_stochastic,
        float(R_sample.mean()), float(R_sample.std()),
        spread_stochastic - spread_fixed,
    )


# ---- Wrong-way recovery CVA ----

@dataclass
class WrongWayRecoveryCVA:
    """CVA with recovery falling in stress."""
    cva_fixed_recovery: float
    cva_stochastic_recovery: float
    wrong_way_adjustment: float


def wrong_way_recovery_cva(
    epe: list[float],
    time_grid: list[float],
    flat_hazard: float,
    flat_rate: float,
    base_recovery: float = 0.40,
    recovery_vol: float = 0.10,
    hazard_recovery_corr: float = -0.30,
    n_paths: int = 50_000,
    seed: int | None = None,
) -> WrongWayRecoveryCVA:
    """CVA with stochastic recovery correlated to default intensity.

    Standard CVA: CVA = Σ df × ΔPD × (1−R) × EPE.
    Wrong-way: R drops when default is more likely.

    Args:
        epe: expected positive exposure at each time point.
        time_grid: corresponding times.
    """
    rng = np.random.default_rng(seed)

    cva_fixed = 0.0
    cva_stoch = 0.0

    for i in range(1, len(time_grid)):
        t = time_grid[i]
        t_prev = time_grid[i - 1]
        dt = t - t_prev
        df = math.exp(-flat_rate * t)
        surv = math.exp(-flat_hazard * t)
        surv_prev = math.exp(-flat_hazard * t_prev)
        dpd = surv_prev - surv
        exposure = epe[i] if i < len(epe) else epe[-1]

        # Fixed recovery
        cva_fixed += df * dpd * (1 - base_recovery) * exposure

        # Stochastic: sample R conditioned on default
        z = rng.standard_normal(n_paths)
        R = np.clip(base_recovery + recovery_vol * z, 0, 1)
        # Wrong-way: in default scenarios, hazard is high → R shifts down
        R_adj = np.clip(R + hazard_recovery_corr * recovery_vol, 0, 1)
        mean_lgd = float((1 - R_adj).mean())
        cva_stoch += df * dpd * mean_lgd * exposure

    return WrongWayRecoveryCVA(cva_fixed, cva_stoch, cva_stoch - cva_fixed)
