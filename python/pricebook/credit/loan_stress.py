"""Loan portfolio stress testing: correlated defaults, macro scenarios,
migration matrices, concentration risk.

    from pricebook.loan_stress import (
        portfolio_loss_distribution, correlated_default_simulation,
        macro_stress_scenario, concentration_metrics, PREDEFINED_SCENARIOS,
    )

References:
    Gordy (2003). A Risk-Factor Model Foundation for Ratings-Based Capital Rules. JFI.
    Gordy & Lütkebohmert (2013). Granularity Adjustment for Regulatory Capital. JBF.
    Moody's (2022). Annual Default Study.
    Basel Committee (2006). International Convergence of Capital Measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class MacroScenario:
    """Macro stress scenario specification."""
    name: str
    gdp_shock: float         # e.g. -0.03 = -3% GDP
    rate_shock: float         # parallel rate shift (e.g. 0.02 = +200bp)
    spread_shock: float       # spread widening (e.g. 0.03 = +300bp)
    pd_multiplier: float      # PD scaling factor (e.g. 2.5)
    recovery_haircut: float   # recovery reduction (e.g. 0.15 = -15pp)
    prepay_shift: float       # CPR adjustment

    def to_dict(self) -> dict:
        return vars(self)


PREDEFINED_SCENARIOS: dict[str, MacroScenario] = {
    "recession": MacroScenario("recession", -0.03, -0.01, 0.03, 2.5, 0.10, -0.10),
    "stagflation": MacroScenario("stagflation", -0.02, 0.02, 0.02, 2.0, 0.08, -0.15),
    "credit_crisis": MacroScenario("credit_crisis", -0.05, -0.02, 0.05, 4.0, 0.20, -0.20),
    "rate_shock": MacroScenario("rate_shock", 0.00, 0.03, 0.01, 1.2, 0.03, -0.05),
    "recovery": MacroScenario("recovery", 0.03, 0.01, -0.01, 0.7, -0.05, 0.05),
}


@dataclass
class PortfolioStressResult:
    """Portfolio-level stress test output."""
    expected_loss: float
    unexpected_loss: float
    var_99: float
    var_999: float
    es_99: float
    loss_by_industry: dict[str, float]
    scenario_name: str

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class ConcentrationMetrics:
    """Portfolio concentration risk metrics."""
    hhi: float
    top_10_pct: float
    industry_hhi: float
    max_single_name_pct: float
    granularity_adjustment: float
    effective_n: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class MigrationResult:
    """Rating migration analysis result."""
    initial_distribution: dict[str, float]
    expected_distribution: dict[str, float]
    expected_default_pct: float
    upgrade_pct: float
    downgrade_pct: float

    def to_dict(self) -> dict:
        return vars(self)


# ═══════════════════════════════════════════════════════════════
# Core Functions
# ═══════════════════════════════════════════════════════════════

def macro_stress_scenario(scenario: str | MacroScenario) -> MacroScenario:
    """Look up or return a macro stress scenario."""
    if isinstance(scenario, MacroScenario):
        return scenario
    if scenario not in PREDEFINED_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario}'. Available: {list(PREDEFINED_SCENARIOS.keys())}")
    return PREDEFINED_SCENARIOS[scenario]


def correlated_default_simulation(
    n_obligors: int,
    pds: list[float] | np.ndarray,
    correlation: float = 0.20,
    n_paths: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    """Simulate correlated defaults via one-factor Gaussian copula.

    Z_i = sqrt(rho) * M + sqrt(1-rho) * eps_i
    Default if Phi(Z_i) < PD_i

    Args:
        n_obligors: number of obligors.
        pds: per-obligor PD (scalar broadcast or array).
        correlation: equi-correlation (asset correlation).
        n_paths: number of MC paths.
        seed: random seed.

    Returns:
        (n_paths, n_obligors) boolean default indicators.
    """
    rng = np.random.default_rng(seed)
    pds_arr = np.broadcast_to(np.asarray(pds), (n_obligors,))

    sqrt_rho = np.sqrt(max(correlation, 0.0))
    sqrt_1_rho = np.sqrt(1.0 - max(correlation, 0.0))

    # Systematic factor
    market = rng.standard_normal(n_paths)
    # Idiosyncratic factors
    idio = rng.standard_normal((n_paths, n_obligors))

    z = sqrt_rho * market[:, np.newaxis] + sqrt_1_rho * idio
    # Default threshold
    thresholds = norm.ppf(pds_arr)
    defaults = z < thresholds

    return defaults


def portfolio_loss_distribution(
    pds: list[float] | np.ndarray,
    notionals: list[float] | np.ndarray,
    recoveries: list[float] | np.ndarray,
    correlation: float = 0.20,
    n_paths: int = 10000,
    industries: list[str] | None = None,
    scenario: str | MacroScenario | None = None,
    seed: int = 42,
) -> PortfolioStressResult:
    """Full portfolio loss distribution with VaR/ES.

    Args:
        pds: per-obligor probability of default.
        notionals: per-obligor notional.
        recoveries: per-obligor recovery rate.
        correlation: equi-correlation.
        n_paths: MC paths.
        industries: per-obligor industry (for by-industry breakdown).
        scenario: optional macro scenario to apply.
        seed: random seed.
    """
    pds_arr = np.array(pds, dtype=float)
    notionals_arr = np.array(notionals, dtype=float)
    recoveries_arr = np.array(recoveries, dtype=float)
    n = len(pds_arr)
    scenario_name = "base"

    # Apply scenario adjustments
    if scenario is not None:
        sc = macro_stress_scenario(scenario)
        scenario_name = sc.name
        pds_arr = np.clip(pds_arr * sc.pd_multiplier, 0.0, 1.0)
        recoveries_arr = np.clip(recoveries_arr - sc.recovery_haircut, 0.0, 1.0)

    # Simulate defaults
    defaults = correlated_default_simulation(n, pds_arr, correlation, n_paths, seed)

    # Loss per path
    lgds = 1.0 - recoveries_arr
    losses_per_path = defaults * notionals_arr * lgds  # (n_paths, n_obligors)
    total_losses = np.sum(losses_per_path, axis=1)     # (n_paths,)

    el = float(np.mean(total_losses))
    ul = float(np.std(total_losses))
    var_99 = float(np.percentile(total_losses, 99))
    var_999 = float(np.percentile(total_losses, 99.9))

    # Expected shortfall at 99%
    tail = total_losses[total_losses >= var_99]
    es_99 = float(np.mean(tail)) if len(tail) > 0 else var_99

    # By-industry breakdown
    loss_by_industry: dict[str, float] = {}
    if industries is not None:
        for j in range(n):
            ind = industries[j]
            avg_loss = float(np.mean(losses_per_path[:, j]))
            loss_by_industry[ind] = loss_by_industry.get(ind, 0.0) + avg_loss

    return PortfolioStressResult(
        expected_loss=el, unexpected_loss=ul,
        var_99=var_99, var_999=var_999, es_99=es_99,
        loss_by_industry=loss_by_industry,
        scenario_name=scenario_name,
    )


def concentration_metrics(
    obligor_names: list[str],
    notionals: list[float] | np.ndarray,
    industries: list[str] | None = None,
) -> ConcentrationMetrics:
    """Compute portfolio concentration risk metrics.

    Args:
        obligor_names: obligor identifiers.
        notionals: per-obligor notional.
        industries: per-obligor industry.
    """
    notionals_arr = np.array(notionals, dtype=float)
    total = float(np.sum(notionals_arr))
    if total <= 0:
        raise ValueError("Total notional must be positive")

    weights = notionals_arr / total

    # HHI (name-level)
    hhi = float(np.sum(weights ** 2))
    effective_n = 1.0 / hhi if hhi > 0 else float(len(weights))

    # Top 10
    sorted_w = np.sort(weights)[::-1]
    top_10_pct = float(np.sum(sorted_w[:10]))
    max_single = float(sorted_w[0])

    # Industry HHI
    industry_hhi = 0.0
    if industries is not None:
        by_ind: dict[str, float] = {}
        for i, ind in enumerate(industries):
            by_ind[ind] = by_ind.get(ind, 0.0) + weights[i]
        industry_hhi = sum(w ** 2 for w in by_ind.values())

    # Granularity Adjustment (Gordy 2004, simplified)
    # GA ≈ 0.5 × Σ s_i² / (Σ s_i)² × average_capital_factor
    # Simplified: GA = 0.5 × HHI × (avg_PD × avg_LGD)
    ga = 0.5 * hhi  # simplified

    return ConcentrationMetrics(
        hhi=hhi, top_10_pct=top_10_pct,
        industry_hhi=industry_hhi,
        max_single_name_pct=max_single,
        granularity_adjustment=ga,
        effective_n=effective_n,
    )


def migration_matrix(
    initial_ratings: list[str],
    notionals: list[float] | np.ndarray,
    transition_probs: dict[str, dict[str, float]],
    horizon: int = 1,
) -> MigrationResult:
    """Project rating distribution forward using transition matrix.

    Args:
        initial_ratings: per-obligor current rating.
        notionals: per-obligor notional.
        transition_probs: {from_rating: {to_rating: probability}}.
            Must include "D" (default) as a destination.
        horizon: number of periods to project.
    """
    notionals_arr = np.array(notionals, dtype=float)
    total = float(np.sum(notionals_arr))
    if total <= 0:
        raise ValueError("Total notional must be positive")

    # Current distribution
    initial_dist: dict[str, float] = {}
    for i, rating in enumerate(initial_ratings):
        initial_dist[rating] = initial_dist.get(rating, 0.0) + notionals_arr[i]

    # All ratings — maintain order from transition_probs keys (assumed credit quality order)
    # then append any destinations not already seen
    all_ratings: list[str] = []
    for r in transition_probs:
        if r not in all_ratings:
            all_ratings.append(r)
    for probs in transition_probs.values():
        for r in probs:
            if r not in all_ratings:
                all_ratings.append(r)

    # Build transition matrix
    n_ratings = len(all_ratings)
    idx = {r: i for i, r in enumerate(all_ratings)}
    T = np.zeros((n_ratings, n_ratings))
    for from_r, probs in transition_probs.items():
        if from_r in idx:
            for to_r, p in probs.items():
                if to_r in idx:
                    T[idx[from_r], idx[to_r]] = p

    # Multi-period: T^horizon via matrix power
    if horizon > 1:
        T_h = np.linalg.matrix_power(T, horizon)
    else:
        T_h = T

    # Apply to current distribution
    current_vec = np.zeros(n_ratings)
    for r, amt in initial_dist.items():
        if r in idx:
            current_vec[idx[r]] = amt

    future_vec = current_vec @ T_h

    # Expected distribution
    expected_dist = {all_ratings[i]: float(future_vec[i]) for i in range(n_ratings) if future_vec[i] > 0}

    # Default rate
    default_pct = float(future_vec[idx.get("D", -1)]) / total if "D" in idx and total > 0 else 0.0

    # Upgrade/downgrade
    upgrades = 0.0
    downgrades = 0.0
    for i, r_from in enumerate(initial_ratings):
        if r_from not in idx:
            continue
        fi = idx[r_from]
        for j, r_to in enumerate(all_ratings):
            if j < fi:  # higher rating = upgrade
                upgrades += notionals_arr[i] * T_h[fi, j]
            elif j > fi:  # lower rating = downgrade
                downgrades += notionals_arr[i] * T_h[fi, j]

    return MigrationResult(
        initial_distribution={r: v / total for r, v in initial_dist.items()},
        expected_distribution={r: v / total for r, v in expected_dist.items()},
        expected_default_pct=default_pct,
        upgrade_pct=upgrades / total if total > 0 else 0.0,
        downgrade_pct=downgrades / total if total > 0 else 0.0,
    )
