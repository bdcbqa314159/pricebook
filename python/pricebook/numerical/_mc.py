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


def antithetic_normals(rng_normals: np.ndarray) -> np.ndarray:
    """Return the antithetic counterparts of a normal-draw array: -Z.

    Antithetic variates exploit the symmetry of the standard normal
    distribution: if Z ~ N(0, I) drives a Monte Carlo path, then -Z also
    has the right distribution and is perfectly negatively correlated
    with Z.  Averaging payoffs from both runs gives an unbiased
    estimator with strictly lower variance than two independent runs.

    Usage:
        Z = rng.standard_normal(...)
        paths_pos = simulate(Z)
        paths_neg = simulate(antithetic_normals(Z))
        estimator = 0.5 * (payoff(paths_pos) + payoff(paths_neg))

    Fix T4-MC2: pre-fix this function was named `antithetic_paths` and
    had a confusingly bi-modal interface that produced WRONG results in
    one branch:

      (i) when called with `rng_normals` (not None), it just returned
          ``-rng_normals`` — a misleading no-op given the function's
          name suggested it returned paths.

      (ii) when called WITHOUT `rng_normals`, it did a "mirror around
          log-mean of terminal values" — NOT a valid antithetic.  That
          transformation depends on the sample mean (a random quantity),
          making the result biased and not actually antithetic.  It also
          ran ``np.log`` on terminal spots, silently producing NaNs for
          paths with non-positive terminals (Bachelier, OU, etc.).

    Post-fix the function is renamed to `antithetic_normals` to reflect
    what it actually does, requires the normal draws explicitly, and
    only performs the valid -Z operation.
    """
    return -np.asarray(rng_normals)


# Backwards-compatibility alias (deprecated): the old `antithetic_paths`
# name now points to `antithetic_normals`.  Any caller passing the second
# (no-longer-supported) path-mirroring branch will hit a TypeError because
# `rng_normals` is now positional and required.
antithetic_paths = antithetic_normals


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
