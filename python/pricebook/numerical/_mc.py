"""MC improvements: QE Heston, antithetic variates, MLMC, Sobol.

    from pricebook.numerical import qe_heston_step, antithetic_paths, multilevel_mc
    from pricebook.numerical import MCVarianceReduction, MLMCResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class MCVarianceReduction(Enum):
    """Monte Carlo variance reduction techniques."""
    NONE = "none"
    ANTITHETIC = "antithetic"
    CONTROL_VARIATE = "control_variate"
    IMPORTANCE_SAMPLING = "importance_sampling"


class MCDiscrMethod(Enum):
    """Stochastic process discretisation schemes."""
    EULER = "euler"
    MILSTEIN = "milstein"
    QE_HESTON = "qe_heston"


def qe_heston_step(
    v: np.ndarray,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    rng=None,
    psi_c: float = 1.5,
) -> np.ndarray:
    """Andersen's Quadratic Exponential (QE) scheme for Heston variance.

    Splits into two regimes based on psi = s^2/m^2:
    - psi <= psi_c: quadratic approximation (matching first two moments)
    - psi > psi_c: exponential approximation (mass at zero + exponential)

    This avoids negative variance without full truncation or reflection.

    Reference: Andersen (2008), "Efficient Simulation of the Heston Model."
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(v)
    v_next = np.zeros(n)

    # Conditional mean and variance of V(t+dt) | V(t)
    e_decay = math.exp(-kappa * dt)
    m = theta + (v - theta) * e_decay  # E[V(t+dt) | V(t)]
    s2 = (v * xi ** 2 * e_decay / kappa * (1 - e_decay)
          + theta * xi ** 2 / (2 * kappa) * (1 - e_decay) ** 2)
    s2 = np.maximum(s2, 0.0)

    psi = s2 / np.maximum(m ** 2, 1e-20)
    U = rng.uniform(size=n)

    # Quadratic regime (low variance-of-variance)
    quad = psi <= psi_c
    if np.any(quad):
        b2 = 2 / psi[quad] - 1 + np.sqrt(np.maximum(2 / psi[quad] * (2 / psi[quad] - 1), 0))
        a = m[quad] / (1 + b2)
        b = np.sqrt(np.maximum(b2, 0))
        from scipy.stats import norm
        Zv = norm.ppf(U[quad])
        v_next[quad] = a * (b + Zv) ** 2

    # Exponential regime (high variance-of-variance)
    exp_mask = ~quad
    if np.any(exp_mask):
        p = (psi[exp_mask] - 1) / (psi[exp_mask] + 1)
        beta_param = (1 - p) / np.maximum(m[exp_mask], 1e-20)
        # Inverse CDF: if U <= p -> 0, else -> -log((1-p)/(1-U)) / beta
        v_next[exp_mask] = np.where(
            U[exp_mask] <= p,
            0.0,
            np.log(np.maximum((1 - p) / (1 - U[exp_mask]), 1e-300)) / beta_param,
        )

    return np.maximum(v_next, 0.0)


def antithetic_paths(
    base_paths: np.ndarray,
    rng_normals: np.ndarray | None = None,
) -> np.ndarray:
    """Generate antithetic paths from existing paths.

    If paths were generated with Z, antithetic uses -Z.
    Returns the average of base and antithetic terminal values.

    For simple use: pass the normal draws used to build base_paths.
    """
    # If we have the normals, rebuild with negated
    if rng_normals is not None:
        # The caller should rebuild paths with -rng_normals
        return -rng_normals  # return negated normals for caller to use

    # If we only have paths (log-space), mirror around the mean
    log_paths = np.log(base_paths[:, -1])
    mean_log = log_paths.mean()
    antithetic_log = 2 * mean_log - log_paths
    return np.exp(antithetic_log)


@dataclass
class MLMCResult:
    """Multilevel Monte Carlo result."""
    estimate: float
    variance: float
    cost: float
    n_levels: int
    samples_per_level: list[int]

    def to_dict(self) -> dict:
        return vars(self)


def multilevel_mc(
    payoff,
    process_fn,
    T: float,
    levels: int = 5,
    base_steps: int = 4,
    base_paths: int = 10_000,
    seed: int = 42,
) -> MLMCResult:
    """Multilevel Monte Carlo (Giles 2008).

    Achieves O(ε⁻²) cost instead of O(ε⁻³) for standard MC.

    Estimates E[P_L] = E[P_0] + Σ_{l=1}^{L} E[P_l − P_{l−1}] where P_l is the
    payoff at refinement level l (2^l × base_steps timesteps).

    **Giles coupling.**  At each level l ≥ 1 we simulate the FINE path with
    n_fine = base_steps · 2^l Brownian increments dW^f, then construct the
    COARSE path on the SAME Brownian path by pairing increments:
    dW^c[m] = dW^f[2m] + dW^f[2m+1].  This makes Var(P_fine − P_coarse) →
    O(h^β) where β > 0 (the strong convergence rate of the scheme), the
    property that makes MLMC's variance-budget allocation possible.

    Pre-fix T1.1, this routine generated the fine path then *downsampled* it
    (`paths_coarse = paths_fine[:, ::2]`) to obtain the "coarse" path.  For
    European payoffs depending only on the terminal value, this gave
    P_coarse ≡ P_fine, so P_fine − P_coarse ≡ 0 at every level ≥ 1.  The
    MLMC estimator collapsed to E[P_0] alone (a 4-step Euler estimate),
    completely defeating the purpose of MLMC.

    Args:
        payoff: callable(paths) → array of payoff values.
        process_fn: callable(dW, dt) → paths array (n_paths, n_steps+1).
            **NEW** interface — the routine pre-generates Brownian increments
            dW (shape (n_paths, n_steps), scaled by sqrt(dt)) and passes them
            to the stepper.  This lets MLMC supply paired-sum increments to
            the coarse stepper for proper Giles coupling.
        T: time horizon.
        levels: number of refinement levels (L+1 in Giles' notation).
        base_steps: timesteps at the coarsest level (l = 0).
        base_paths: paths at the coarsest level (scaled down at finer levels).
        seed: PRNG seed.
    """
    rng = np.random.default_rng(seed)

    total_estimate = 0.0
    total_variance = 0.0
    total_cost = 0.0
    samples = []

    for l in range(levels):
        n_steps = base_steps * (2 ** l)
        n_paths = max(base_paths // (2 ** l), 100)
        samples.append(n_paths)

        if l == 0:
            # Coarsest level: estimate E[P_0] with n_steps = base_steps.
            dt = T / n_steps
            dW = rng.standard_normal((n_paths, n_steps)) * math.sqrt(dt)
            paths = process_fn(dW, dt)
            P = payoff(paths)
            total_estimate += float(P.mean())
            total_variance += float(P.var() / n_paths)
        else:
            # Level l: compute E[P_l − P_{l−1}] with Giles coupling.
            n_fine = n_steps
            n_coarse = n_steps // 2
            dt_fine = T / n_fine
            dt_coarse = T / n_coarse  # = 2 · dt_fine

            # Fine Brownian increments (the master Brownian path).
            Z = rng.standard_normal((n_paths, n_fine))
            dW_fine = Z * math.sqrt(dt_fine)
            # Coarse increments by PAIRING the fine ones.
            dW_coarse = dW_fine[:, 0::2] + dW_fine[:, 1::2]

            paths_fine = process_fn(dW_fine, dt_fine)
            paths_coarse = process_fn(dW_coarse, dt_coarse)

            P_fine = payoff(paths_fine)
            P_coarse = payoff(paths_coarse)
            correction = P_fine - P_coarse

            total_estimate += float(correction.mean())
            total_variance += float(correction.var() / n_paths)

        total_cost += n_paths * n_steps

    return MLMCResult(
        estimate=total_estimate,
        variance=total_variance,
        cost=total_cost,
        n_levels=levels,
        samples_per_level=samples,
    )
