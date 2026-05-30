"""Hawkes process analytics for credit contagion.

Scenario analysis, clustering metrics, and calibration quality.

    from pricebook.credit.hawkes_analytics import (
        contagion_scenario, clustering_metrics,
        kernel_comparison, hawkes_term_structure,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.hawkes_credit import (
    FractionalHawkesProcess, MultivariateHawkesProcess,
    HawkesKernel, evaluate_kernel,
)


@dataclass
class ContagionScenarioResult:
    """Result of a contagion scenario: "what if name X defaults?"."""
    trigger_name: int
    intensity_before: np.ndarray    # (N,) baseline intensity per name
    intensity_after: np.ndarray     # (N,) intensity immediately after trigger default
    intensity_jump: np.ndarray      # (N,) = after - before
    affected_names: list[int]       # names with jump > 0
    max_jump_name: int              # most affected name
    max_jump: float
    decay_half_life: float          # time for jump to decay by half

    def to_dict(self) -> dict:
        return {
            "trigger_name": self.trigger_name,
            "max_jump_name": self.max_jump_name,
            "max_jump": self.max_jump,
            "n_affected": len(self.affected_names),
            "decay_half_life": self.decay_half_life,
        }


def contagion_scenario(
    mv_hawkes: MultivariateHawkesProcess,
    trigger_name: int,
) -> ContagionScenarioResult:
    """Compute the immediate intensity impact of a single default.

    When name `trigger_name` defaults at t=0:
    - λᵢ(0+) = μᵢ + φᵢⱼ(0)  for all i
    - The jump φᵢⱼ(0) decays according to the kernel

    Args:
        mv_hawkes: configured multivariate Hawkes process.
        trigger_name: index of the defaulting name.
    """
    N = mv_hawkes.n_names
    before = mv_hawkes.mu.copy()

    # Immediate jump: φᵢⱼ(0+) for each name i
    after = mv_hawkes.mu.copy()
    jumps = np.zeros(N)
    for i in range(N):
        alpha_ij = mv_hawkes.alpha_matrix[i, trigger_name]
        if alpha_ij > 0:
            params = {**mv_hawkes.kernel_params, "alpha": alpha_ij}
            jump = evaluate_kernel(1e-6, mv_hawkes.kernel, params)  # at t=0+
            after[i] += jump
            jumps[i] = jump

    affected = [i for i in range(N) if jumps[i] > 1e-10 and i != trigger_name]
    max_idx = int(np.argmax(jumps)) if np.max(jumps) > 0 else trigger_name

    # Decay half-life (for exponential kernel)
    if mv_hawkes.kernel == HawkesKernel.EXPONENTIAL:
        beta = mv_hawkes.kernel_params.get("beta", 1.0)
        half_life = math.log(2) / beta
    else:
        # Numerical: find t where φ(t) = φ(0+)/2
        alpha_max = float(np.max(mv_hawkes.alpha_matrix))
        params = {**mv_hawkes.kernel_params, "alpha": alpha_max}
        phi_0 = evaluate_kernel(1e-6, mv_hawkes.kernel, params)
        half_life = 1.0  # default
        for t_test in np.linspace(0.01, 20.0, 200):
            if evaluate_kernel(t_test, mv_hawkes.kernel, params) <= phi_0 / 2:
                half_life = t_test
                break

    return ContagionScenarioResult(
        trigger_name=trigger_name,
        intensity_before=before,
        intensity_after=after,
        intensity_jump=jumps,
        affected_names=affected,
        max_jump_name=max_idx,
        max_jump=float(np.max(jumps)),
        decay_half_life=half_life,
    )


@dataclass
class ClusteringMetrics:
    """Default clustering statistics."""
    inter_arrival_mean: float
    inter_arrival_std: float
    inter_arrival_cv: float        # coefficient of variation (>1 = clustered)
    burstiness: float              # (cv-1)/(cv+1), 0=Poisson, >0=bursty
    n_events: int

    def to_dict(self) -> dict:
        return vars(self)


def clustering_metrics(event_times: list[float]) -> ClusteringMetrics:
    """Compute clustering statistics from event times.

    Key metric: coefficient of variation (CV) of inter-arrival times.
    - CV = 1: Poisson (no clustering)
    - CV > 1: clustered (bursty)
    - CV < 1: regular (anti-clustered)

    Burstiness B = (CV - 1) / (CV + 1):
    - B = 0: Poisson
    - B > 0: bursty
    - B = 1: maximally bursty
    """
    if len(event_times) < 3:
        return ClusteringMetrics(0, 0, 1.0, 0.0, len(event_times))

    events = sorted(event_times)
    inter_arrivals = np.diff(events)

    mean_ia = float(np.mean(inter_arrivals))
    std_ia = float(np.std(inter_arrivals))
    cv = std_ia / mean_ia if mean_ia > 0 else 1.0
    burstiness = (cv - 1) / (cv + 1)

    return ClusteringMetrics(
        inter_arrival_mean=mean_ia,
        inter_arrival_std=std_ia,
        inter_arrival_cv=cv,
        burstiness=burstiness,
        n_events=len(events),
    )


def kernel_comparison(
    mu: float = 0.02,
    maturity: float = 10.0,
    n_paths: int = 5_000,
    seed: int = 42,
) -> dict:
    """Compare exponential vs power-law kernel behaviour.

    Same baseline intensity μ, different memory kernels.
    Shows how long memory affects default clustering and CDS spreads.
    """
    from pricebook.credit.hawkes_cds import hawkes_cds_spread

    results = {}

    # Exponential: α=0.3, β=1.0 (BR=0.30, half-life=0.69Y)
    h_exp = FractionalHawkesProcess(
        mu, HawkesKernel.EXPONENTIAL, {"alpha": 0.3, "beta": 1.0})
    r_exp = hawkes_cds_spread(h_exp, maturity=5.0, n_paths=n_paths, seed=seed)
    sim_exp = h_exp.simulate(maturity, n_paths=min(500, n_paths), n_grid=100, seed=seed)

    # Power-law: α=0.1, H=0.3 (long memory, rough)
    h_pl = FractionalHawkesProcess(
        mu, HawkesKernel.POWER_LAW, {"alpha": 0.1, "H": 0.3})
    r_pl = hawkes_cds_spread(h_pl, maturity=5.0, n_paths=n_paths, seed=seed)
    sim_pl = h_pl.simulate(maturity, n_paths=min(500, n_paths), n_grid=100, seed=seed)

    # Clustering metrics
    exp_clustering = [clustering_metrics(e) for e in sim_exp.event_times[:100] if len(e) > 3]
    pl_clustering = [clustering_metrics(e) for e in sim_pl.event_times[:100] if len(e) > 3]

    results["exponential"] = {
        "kernel": "exponential",
        "branching_ratio": r_exp.branching_ratio,
        "cds_spread_bp": r_exp.par_spread_bp,
        "mean_burstiness": float(np.mean([c.burstiness for c in exp_clustering])) if exp_clustering else 0,
        "mean_cv": float(np.mean([c.inter_arrival_cv for c in exp_clustering])) if exp_clustering else 1,
    }
    results["power_law"] = {
        "kernel": "power_law (H=0.3)",
        "branching_ratio": r_pl.branching_ratio,
        "cds_spread_bp": r_pl.par_spread_bp,
        "mean_burstiness": float(np.mean([c.burstiness for c in pl_clustering])) if pl_clustering else 0,
        "mean_cv": float(np.mean([c.inter_arrival_cv for c in pl_clustering])) if pl_clustering else 1,
    }

    return results


def hawkes_term_structure(
    hawkes_process: FractionalHawkesProcess,
    maturities: list[float] | None = None,
    recovery: float = 0.4,
    n_paths: int = 10_000,
) -> list[dict]:
    """CDS spread term structure under Hawkes intensity.

    Returns spreads at multiple maturities to show the effect of
    self-excitation on the term structure shape.
    """
    from pricebook.credit.hawkes_cds import hawkes_cds_spread

    if maturities is None:
        maturities = [1, 2, 3, 5, 7, 10]

    results = []
    for T in maturities:
        r = hawkes_cds_spread(hawkes_process, maturity=float(T), recovery=recovery,
                              n_paths=n_paths)
        results.append({
            "maturity": T,
            "spread_bp": r.par_spread_bp,
            "survival": r.survival_5y if T >= 5 else 0,
        })

    return results
