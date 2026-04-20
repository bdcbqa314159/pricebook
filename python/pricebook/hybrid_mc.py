"""Multi-factor hybrid MC: 3+ factor simulation framework.

* :class:`HybridMCEngine` — N-factor correlated simulation.
* :func:`hybrid_payoff_evaluate` — evaluate path-dependent payoff.

References:
    Piterbarg, *Smiling Hybrids*, Risk, 2006.
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np


@dataclass
class HybridFactor:
    """Definition of one factor in the hybrid."""
    name: str
    initial: float
    vol: float
    mean_reversion: float       # 0 for GBM, >0 for OU
    long_run: float             # θ for OU
    factor_type: str            # "gbm", "ou", "lognormal"


@dataclass
class HybridMCResult:
    """Multi-factor hybrid simulation result."""
    paths: dict[str, np.ndarray]    # {factor_name: (n_paths, n_steps+1)}
    n_factors: int
    n_paths: int
    n_steps: int
    T: float


class HybridMCEngine:
    """N-factor correlated MC engine for hybrid products."""

    def __init__(self, factors: list[HybridFactor], correlations: np.ndarray):
        self.factors = factors
        self.n = len(factors)
        corr = np.array(correlations, dtype=float)
        eigvals = np.linalg.eigvalsh(corr)
        if eigvals.min() < 0:
            corr += (-eigvals.min() + 1e-6) * np.eye(self.n)
        self.L = np.linalg.cholesky(corr)

    def simulate(
        self, T: float, n_paths: int = 5_000, n_steps: int = 100,
        seed: int | None = 42,
    ) -> HybridMCResult:
        rng = np.random.default_rng(seed)
        dt = T / n_steps; sqrt_dt = math.sqrt(dt)

        paths = {}
        state = np.zeros((n_paths, self.n))
        for i, f in enumerate(self.factors):
            arr = np.full((n_paths, n_steps + 1), float(f.initial))
            paths[f.name] = arr
            state[:, i] = f.initial

        for step in range(n_steps):
            Z = rng.standard_normal((n_paths, self.n)) @ self.L.T

            for i, f in enumerate(self.factors):
                if f.factor_type == "gbm":
                    drift = -0.5 * f.vol**2 * dt
                    state[:, i] = state[:, i] * np.exp(drift + f.vol * Z[:, i] * sqrt_dt)
                elif f.factor_type == "ou":
                    state[:, i] += f.mean_reversion * (f.long_run - state[:, i]) * dt \
                                   + f.vol * Z[:, i] * sqrt_dt
                elif f.factor_type == "lognormal":
                    drift = (f.mean_reversion - 0.5 * f.vol**2) * dt
                    state[:, i] = state[:, i] * np.exp(drift + f.vol * Z[:, i] * sqrt_dt)
                else:
                    state[:, i] += f.vol * Z[:, i] * sqrt_dt

                paths[f.name][:, step + 1] = state[:, i]

        return HybridMCResult(paths, self.n, n_paths, n_steps, T)


@dataclass
class HybridPayoffResult:
    """Evaluated hybrid payoff."""
    price: float
    std_error: float
    n_paths: int

def hybrid_payoff_evaluate(
    mc_result: HybridMCResult,
    payoff_fn,
    rate: float = 0.0,
) -> HybridPayoffResult:
    """Evaluate a path-dependent payoff on hybrid MC paths.

    Args:
        payoff_fn: callable(paths_dict) → (n_paths,) array of payoffs.
        rate: risk-free rate for discounting.
    """
    payoffs = payoff_fn(mc_result.paths)
    df = math.exp(-rate * mc_result.T)
    discounted = df * np.array(payoffs)
    price = float(discounted.mean())
    se = float(discounted.std() / math.sqrt(mc_result.n_paths))
    return HybridPayoffResult(price, se, mc_result.n_paths)
