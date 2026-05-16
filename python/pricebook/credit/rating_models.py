"""Rating-based credit models: generator calibration, TTC/PIT, momentum.

Extends :mod:`pricebook.rating_transition` with:

* :func:`calibrate_generator` — fit Q from observed transition frequencies.
* :func:`ttc_to_pit` — through-the-cycle to point-in-time PD conversion.
* :func:`pit_to_ttc` — reverse conversion.
* :class:`MomentumTransitionMatrix` — rating momentum (downgrades beget downgrades).
* :func:`time_varying_generator` — economic-cycle-adjusted Q(t).

References:
    Jarrow, Lando & Turnbull, *A Markov Model for the Term Structure of
    Credit Risk Spreads*, RFS, 1997.
    Lando, *Credit Risk Modeling*, Princeton, 2004, Ch. 7.
    Nickell, Perraudin & Varotto, *Stability of Rating Transitions*,
    J. Banking & Finance, 2000.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import logm, expm
from scipy.optimize import minimize

from pricebook.rating_transition import RatingTransitionMatrix


# ---- Generator calibration ----

@dataclass
class CalibrationResult:
    """Result of generator matrix calibration."""
    generator: RatingTransitionMatrix
    residual: float
    converged: bool


def calibrate_generator(
    ratings: list[str],
    observed_P: np.ndarray | list[list[float]],
    horizon: float = 1.0,
) -> CalibrationResult:
    """Calibrate generator Q from an observed transition matrix P.

    Solves the inverse problem: find Q such that exp(Q × horizon) ≈ P.

    Uses matrix logarithm as initial estimate, then projects onto
    the space of valid generators (non-negative off-diagonal, rows
    sum to zero, absorbing default state).

    Args:
        ratings: rating labels (last = default).
        observed_P: (n, n) observed transition probability matrix.
        horizon: time horizon over which P was observed (typically 1Y).

    Reference:
        Israel, Rosenthal & Wei, *Finding Generators for Markov Chains*,
        Linear Algebra and its Applications, 2001.
    """
    P = np.asarray(observed_P, dtype=float)
    n = len(ratings)

    # Initial estimate via matrix logarithm
    try:
        Q_init = logm(P).real / horizon
    except Exception:
        Q_init = np.zeros((n, n))

    # Project onto valid generator space
    Q = _project_to_generator(Q_init, n)

    # Refine via optimisation: minimise ||exp(Q*h) - P||_F
    def objective(q_flat):
        Q_try = _unflatten_generator(q_flat, n)
        P_try = expm(Q_try * horizon)
        return float(np.sum((P_try - P) ** 2))

    q0 = _flatten_generator(Q, n)
    result = minimize(objective, q0, method='L-BFGS-B',
                      bounds=[(0, None)] * len(q0))

    Q_final = _unflatten_generator(result.x, n)
    residual = float(np.linalg.norm(expm(Q_final * horizon) - P, 'fro'))

    return CalibrationResult(
        RatingTransitionMatrix(ratings, Q_final),
        residual,
        residual < 0.05,
    )


def _project_to_generator(Q: np.ndarray, n: int) -> np.ndarray:
    """Project a matrix onto the space of valid generators."""
    Q = Q.copy()
    # Off-diagonal: must be non-negative
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = max(Q[i, j], 0.0)
    # Default state (last row) is absorbing
    Q[-1, :] = 0.0
    # Diagonal: rows sum to zero
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def _flatten_generator(Q: np.ndarray, n: int) -> np.ndarray:
    """Extract off-diagonal elements (excluding default row) for optimisation."""
    params = []
    for i in range(n - 1):
        for j in range(n):
            if i != j:
                params.append(Q[i, j])
    return np.array(params)


def _unflatten_generator(params: np.ndarray, n: int) -> np.ndarray:
    """Rebuild generator from off-diagonal parameters."""
    Q = np.zeros((n, n))
    idx = 0
    for i in range(n - 1):
        for j in range(n):
            if i != j:
                Q[i, j] = max(params[idx], 0.0)
                idx += 1
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


# ---- TTC / PIT conversion ----

@dataclass
class TTCPITResult:
    """Through-the-cycle / point-in-time PD conversion."""
    ttc_pd: float
    pit_pd: float
    cycle_factor: float
    regime: str  # "expansion", "contraction", "neutral"


def ttc_to_pit(
    ttc_pd: float,
    cycle_factor: float,
) -> TTCPITResult:
    """Convert through-the-cycle PD to point-in-time PD.

    PIT PD = Φ(Φ⁻¹(TTC PD) + cycle_factor)

    where cycle_factor > 0 means downturn (higher PD),
    cycle_factor < 0 means expansion (lower PD).

    The Merton/Vasicek framework: the systematic factor Z shifts
    the conditional PD. TTC PD = unconditional; PIT PD = conditional
    on the current economic state.

    Args:
        ttc_pd: through-the-cycle (long-run average) PD.
        cycle_factor: macro adjustment (σ × Z, where Z is the current
            systematic factor). Positive = stress.
    """
    from scipy.stats import norm

    z_ttc = norm.ppf(min(max(ttc_pd, 1e-10), 1 - 1e-10))
    z_pit = z_ttc + cycle_factor
    pit_pd = float(norm.cdf(z_pit))

    if cycle_factor > 0.1:
        regime = "contraction"
    elif cycle_factor < -0.1:
        regime = "expansion"
    else:
        regime = "neutral"

    return TTCPITResult(ttc_pd, pit_pd, cycle_factor, regime)


def pit_to_ttc(
    pit_pd: float,
    cycle_factor: float,
) -> TTCPITResult:
    """Convert point-in-time PD back to through-the-cycle PD.

    TTC PD = Φ(Φ⁻¹(PIT PD) − cycle_factor)
    """
    from scipy.stats import norm

    z_pit = norm.ppf(min(max(pit_pd, 1e-10), 1 - 1e-10))
    z_ttc = z_pit - cycle_factor
    ttc_pd = float(norm.cdf(z_ttc))

    if cycle_factor > 0.1:
        regime = "contraction"
    elif cycle_factor < -0.1:
        regime = "expansion"
    else:
        regime = "neutral"

    return TTCPITResult(ttc_pd, pit_pd, cycle_factor, regime)


# ---- Momentum transition matrix ----

class MomentumTransitionMatrix:
    """Rating transition model with momentum (history-dependent).

    A downgraded name has a higher probability of further downgrade
    than a name that has been stable. This is modeled by maintaining
    two generators: Q_stable and Q_downgraded, with a mixing rule.

    λ_down(t) = λ_base + momentum × 1_{downgraded in last period}

    Args:
        base: the standard (memoryless) generator.
        momentum_factor: multiplicative increase in downgrade intensity
            after a recent downgrade (e.g. 1.5 = 50% more likely).
    """

    def __init__(
        self,
        base: RatingTransitionMatrix,
        momentum_factor: float = 1.5,
    ):
        self.base = base
        self.momentum_factor = momentum_factor
        self.ratings = base.ratings
        self.n = base.n

        # Build the "post-downgrade" generator
        Q_mom = base.Q.copy()
        for i in range(self.n - 1):
            for j in range(i + 1, self.n):  # downgrade transitions
                Q_mom[i, j] *= momentum_factor
        # Re-balance diagonal
        np.fill_diagonal(Q_mom, 0.0)
        np.fill_diagonal(Q_mom, -Q_mom.sum(axis=1))
        self._Q_momentum = Q_mom

    def simulate_paths(
        self,
        initial_rating: str,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Simulate with momentum: use Q_momentum if last move was a downgrade."""
        rng = np.random.default_rng(seed)
        dt = T / n_steps

        P_base = expm(self.base.Q * dt)
        P_mom = expm(self._Q_momentum * dt)

        # Normalise
        P_base = np.maximum(P_base, 0.0)
        P_base /= P_base.sum(axis=1, keepdims=True)
        P_mom = np.maximum(P_mom, 0.0)
        P_mom /= P_mom.sum(axis=1, keepdims=True)

        idx0 = self.base.rating_index(initial_rating)
        paths = np.zeros((n_paths, n_steps + 1), dtype=int)
        paths[:, 0] = idx0

        for i in range(n_steps):
            for p in range(n_paths):
                current = paths[p, i]
                if current == self.base.default_state:
                    paths[p, i + 1] = current
                    continue

                # Check if last transition was a downgrade
                if i > 0 and paths[p, i] > paths[p, i - 1]:
                    P = P_mom  # use momentum generator
                else:
                    P = P_base

                paths[p, i + 1] = rng.choice(self.n, p=P[current])

        return paths

    def default_probability_mc(
        self,
        initial_rating: str,
        T: float,
        n_steps: int = 100,
        n_paths: int = 50_000,
        seed: int = 42,
    ) -> float:
        """MC default probability with momentum."""
        paths = self.simulate_paths(initial_rating, T, n_steps, n_paths, seed)
        defaulted = np.any(paths == self.base.default_state, axis=1)
        return float(defaulted.mean())


# ---- Time-varying generator ----

def time_varying_generator(
    base: RatingTransitionMatrix,
    cycle_factors: list[tuple[float, float]],
) -> list[tuple[float, RatingTransitionMatrix]]:
    """Build a sequence of generators adjusted for the economic cycle.

    Scales downgrade intensities by (1 + cycle_factor) and upgrade
    intensities by (1 − cycle_factor), re-balancing the diagonal.

    Args:
        base: the base (neutral) generator.
        cycle_factors: list of (time, factor) pairs. factor > 0 = stress.

    Returns:
        list of (time, adjusted_generator) pairs.
    """
    result = []
    for t, factor in cycle_factors:
        Q_adj = base.Q.copy()
        for i in range(base.n - 1):
            for j in range(base.n):
                if i == j:
                    continue
                if j > i:  # downgrade
                    Q_adj[i, j] *= (1 + factor)
                else:  # upgrade
                    Q_adj[i, j] *= max(1 - factor, 0.1)
        np.fill_diagonal(Q_adj, 0.0)
        np.fill_diagonal(Q_adj, -Q_adj.sum(axis=1))
        result.append((t, RatingTransitionMatrix(base.ratings, Q_adj)))
    return result
