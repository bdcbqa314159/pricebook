"""Basket CDS and CDO tranche pricing under multivariate Hawkes.

Replaces Gaussian copula default simulation with Hawkes-driven correlated
defaults. Captures contagion clustering that copula misses.

    from pricebook.credit.hawkes_basket import (
        hawkes_basket_defaults, hawkes_tranche_spread,
        hawkes_ftd_spread, hawkes_vs_copula,
        HawkesBasketResult,
    )

The key difference vs copula:
- Copula: defaults are conditionally independent given a common factor
- Hawkes: defaults are self-exciting — one default raises others' intensity

References:
    Errais, Giesecke & Goldberg (2010). Affine Point Processes.
    Ait-Sahalia, Cacho-Diaz & Laeven (2015). Modeling Financial Contagion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.models.hawkes_credit import MultivariateHawkesProcess, HawkesKernel


@dataclass
class HawkesBasketResult:
    """Basket default simulation under multivariate Hawkes."""
    default_times: np.ndarray      # (n_paths, n_names)
    loss_distribution: np.ndarray  # (n_paths,) portfolio loss at maturity
    n_defaults_mean: float
    n_defaults_std: float
    ftd_time_mean: float           # mean first-to-default time
    loss_mean: float               # E[L(T)]
    loss_std: float
    n_names: int
    n_paths: int
    kernel: str

    def to_dict(self) -> dict:
        return {
            "n_names": self.n_names,
            "n_paths": self.n_paths,
            "n_defaults_mean": self.n_defaults_mean,
            "loss_mean": self.loss_mean,
            "loss_std": self.loss_std,
            "ftd_time_mean": self.ftd_time_mean,
            "kernel": self.kernel,
        }


def hawkes_basket_defaults(
    mu: np.ndarray,
    alpha_matrix: np.ndarray,
    kernel: HawkesKernel = HawkesKernel.EXPONENTIAL,
    kernel_params: dict | None = None,
    maturity: float = 5.0,
    recoveries: np.ndarray | None = None,
    n_paths: int = 20_000,
    seed: int | None = 42,
) -> HawkesBasketResult:
    """Simulate correlated defaults via multivariate Hawkes.

    Args:
        mu: (N,) baseline intensities.
        alpha_matrix: (N, N) cross-excitation.
        kernel: HawkesKernel type.
        kernel_params: kernel parameters (shared, except alpha per pair).
        maturity: time horizon.
        recoveries: (N,) recovery rates (default 0.4 for all).
        n_paths: MC paths.
        seed: random seed.
    """
    N = len(mu)
    if recoveries is None:
        recoveries = np.full(N, 0.4)

    mv = MultivariateHawkesProcess(mu, alpha_matrix, kernel,
                                    kernel_params or {"beta": 1.0})
    sim = mv.simulate(maturity, n_paths, n_grid=100, seed=seed)

    # Default times: first event per name per path (inf = no default)
    default_times = sim.default_times  # (n_paths, N)

    # Portfolio loss at maturity: L = (1/N) Σ (1-Rᵢ) 1{τᵢ ≤ T}
    defaulted = default_times <= maturity
    losses = np.zeros(n_paths)
    for i in range(N):
        losses += (1 - recoveries[i]) * defaulted[:, i] / N

    # First-to-default time
    ftd_times = np.min(default_times, axis=1)
    ftd_finite = ftd_times[ftd_times < np.inf]
    ftd_mean = float(np.mean(ftd_finite)) if len(ftd_finite) > 0 else float('inf')

    # Number of defaults per path
    n_defaults = defaulted.sum(axis=1)

    return HawkesBasketResult(
        default_times=default_times,
        loss_distribution=losses,
        n_defaults_mean=float(np.mean(n_defaults)),
        n_defaults_std=float(np.std(n_defaults)),
        ftd_time_mean=ftd_mean,
        loss_mean=float(np.mean(losses)),
        loss_std=float(np.std(losses)),
        n_names=N,
        n_paths=n_paths,
        kernel=kernel.value,
    )


def hawkes_tranche_spread(
    basket: HawkesBasketResult,
    attachment: float,
    detachment: float,
    maturity: float = 5.0,
    discount_rate: float = 0.03,
    frequency: int = 4,
) -> dict:
    """Compute tranche par spread from Hawkes basket defaults.

    Tranche loss: L_tranche = min(max(L - a, 0), d - a) / (d - a)

    Args:
        basket: result from hawkes_basket_defaults.
        attachment: tranche attachment point (e.g. 0.03).
        detachment: tranche detachment point (e.g. 0.07).
    """
    if detachment <= attachment:
        raise ValueError(f"detachment ({detachment}) must exceed attachment ({attachment})")

    width = detachment - attachment
    losses = basket.loss_distribution

    # Tranche loss per path
    tranche_losses = np.minimum(np.maximum(losses - attachment, 0), width) / width

    # Expected tranche loss
    el = float(np.mean(tranche_losses))

    # Approximate par spread: S ≈ (1 × EL) / risky_annuity
    # Risky annuity ≈ Σ dt × df(t) × (1 - EL_cumulative(t))
    # Simplified: use flat annuity with expected survival
    dt = 1.0 / frequency
    annuity = sum(dt * math.exp(-discount_rate * i * dt)
                  for i in range(1, int(maturity * frequency) + 1))
    tranche_annuity = annuity * (1 - el)  # approx risky annuity

    par_spread = el / max(tranche_annuity, 1e-10)

    return {
        "attachment": attachment,
        "detachment": detachment,
        "expected_loss": el,
        "par_spread": par_spread,
        "par_spread_bp": par_spread * 10_000,
        "width": width,
        "n_paths": basket.n_paths,
    }


def hawkes_ftd_spread(
    basket: HawkesBasketResult,
    maturity: float = 5.0,
    recovery: float = 0.4,
    discount_rate: float = 0.03,
) -> float:
    """First-to-default spread from Hawkes basket.

    FTD protection = (1-R) × P(at least 1 default by T).
    """
    ftd_times = np.min(basket.default_times, axis=1)
    ftd_prob = float(np.mean(ftd_times <= maturity))

    if ftd_prob < 1e-10:
        return 0.0

    # Approximate annuity
    dt = 0.25  # quarterly
    annuity = sum(dt * math.exp(-discount_rate * i * dt) * (1 - ftd_prob * i / (maturity * 4))
                  for i in range(1, int(maturity * 4) + 1))

    protection = (1 - recovery) * ftd_prob * math.exp(-discount_rate * maturity / 2)
    return protection / max(annuity, 1e-10)


def hawkes_vs_copula(
    mu: np.ndarray,
    alpha_matrix: np.ndarray,
    maturity: float = 5.0,
    recoveries: np.ndarray | None = None,
    rho_copula: float = 0.3,
    n_paths: int = 10_000,
    seed: int = 42,
) -> dict:
    """Side-by-side comparison: Hawkes vs Gaussian copula.

    Same baseline PDs but different dependence structure.
    """
    from pricebook.credit.basket_cds import simulate_defaults_copula
    from pricebook.core.survival_curve import SurvivalCurve
    from datetime import date
    from dateutil.relativedelta import relativedelta

    N = len(mu)
    if recoveries is None:
        recoveries = np.full(N, 0.4)

    # Hawkes defaults
    hawkes_result = hawkes_basket_defaults(
        mu, alpha_matrix, HawkesKernel.EXPONENTIAL,
        {"beta": 1.0}, maturity, recoveries, n_paths, seed,
    )

    # Copula defaults — build survival curves from mu (flat hazard)
    ref = date(2024, 1, 1)
    surv_curves = []
    for i in range(N):
        dates = [ref + relativedelta(years=y) for y in range(1, int(maturity) + 1)]
        probs = [math.exp(-mu[i] * y) for y in range(1, int(maturity) + 1)]
        surv_curves.append(SurvivalCurve(ref, dates, probs))

    copula_defaults = simulate_defaults_copula(surv_curves, maturity, rho_copula, n_paths, seed)
    copula_n_defaults = copula_defaults.sum(axis=1)
    copula_losses = np.zeros(n_paths)
    for i in range(N):
        copula_losses += (1 - recoveries[i]) * copula_defaults[:, i] / N

    # Compare
    hawkes_defaulted = hawkes_result.default_times <= maturity
    hawkes_n_defaults = hawkes_defaulted.sum(axis=1)

    return {
        "n_names": N,
        "maturity": maturity,
        "hawkes": {
            "loss_mean": hawkes_result.loss_mean,
            "loss_std": hawkes_result.loss_std,
            "n_defaults_mean": hawkes_result.n_defaults_mean,
            "tail_loss_95": float(np.percentile(hawkes_result.loss_distribution, 95)),
            "tail_loss_99": float(np.percentile(hawkes_result.loss_distribution, 99)),
            "zero_loss_pct": float(np.mean(hawkes_result.loss_distribution == 0) * 100),
        },
        "copula": {
            "loss_mean": float(np.mean(copula_losses)),
            "loss_std": float(np.std(copula_losses)),
            "n_defaults_mean": float(np.mean(copula_n_defaults)),
            "tail_loss_95": float(np.percentile(copula_losses, 95)),
            "tail_loss_99": float(np.percentile(copula_losses, 99)),
            "zero_loss_pct": float(np.mean(copula_losses == 0) * 100),
        },
        "comparison": {
            "hawkes_tail_fatter": float(np.percentile(hawkes_result.loss_distribution, 99)) >
                                 float(np.percentile(copula_losses, 99)),
            "hawkes_more_clustered": hawkes_result.n_defaults_std >
                                    float(np.std(copula_n_defaults)),
        },
    }
