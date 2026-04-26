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
    mean_reversion: float       # 0 for GBM, >0 for OU; drift rate for lognormal
    long_run: float             # θ for OU
    factor_type: str            # "gbm", "ou", "lognormal"
    drift: float = 0.0          # risk-neutral drift for GBM (r - q)


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
                    drift = (f.drift - 0.5 * f.vol**2) * dt
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


# ---- 2-factor local-vol MC under Q^T (Pucci 2012b) ----

@dataclass
class LocalVolHybridResult:
    """2D local-vol hybrid MC result."""
    price: float
    std_error: float
    n_paths: int
    n_steps: int
    mean_F: float
    mean_U: float


def simulate_2d_local_vol(
    F0: float,
    U0: float,
    sigma_F,
    sigma_U,
    rho: float,
    T: float,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate two correlated local-vol Q^T-martingales (Pucci Eq 8-10).

    dF/F = sigma_F(t, F) dW^F,  dU/U = sigma_U(t, U) dW^U
    d<W^F, W^U> = rho dt

    sigma_F, sigma_U: callable(t, x) -> vol, or float for flat vol.

    Returns (F_T, U_T) terminal values of shape (n_paths,).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    if not -1 <= rho <= 1:
        raise ValueError(f"rho must be in [-1, 1], got {rho}")
    sqrt_1_rho2 = math.sqrt(1 - rho**2)

    F = np.full(n_paths, F0)
    U = np.full(n_paths, U0)

    flat_F = isinstance(sigma_F, (int, float))
    flat_U = isinstance(sigma_U, (int, float))

    for step in range(n_steps):
        t = step * dt
        Z1 = rng.standard_normal(n_paths)
        Z2 = rho * Z1 + sqrt_1_rho2 * rng.standard_normal(n_paths)

        if flat_F:
            vF = sigma_F
        else:
            vF = np.array([sigma_F(t, f) for f in F])

        if flat_U:
            vU = sigma_U
        else:
            vU = np.array([sigma_U(t, u) for u in U])

        F = F * np.exp(-0.5 * vF**2 * dt + vF * sqrt_dt * Z1)
        U = U * np.exp(-0.5 * vU**2 * dt + vU * sqrt_dt * Z2)

    return F, U


def local_vol_hybrid_price(
    F0: float,
    U0: float,
    discount_factor: float,
    sigma_F,
    sigma_U,
    rho: float,
    T: float,
    payoff_fn,
    n_paths: int = 50_000,
    n_steps: int = 100,
    seed: int | None = 42,
) -> LocalVolHybridResult:
    """Price a European hybrid via 2D local-vol MC (Pucci 2012b, Eq 7).

    v = D_{0,T} * E^T[payoff(F_T, U_T)]

    Args:
        payoff_fn: callable(F_T, U_T) -> array of payoffs per path.
        sigma_F, sigma_U: callable(t, x) -> vol, or float for flat vol.
    """
    F_T, U_T = simulate_2d_local_vol(
        F0, U0, sigma_F, sigma_U, rho, T, n_paths, n_steps, seed)

    payoffs = payoff_fn(F_T, U_T)
    price = discount_factor * float(payoffs.mean())
    std_err = discount_factor * float(payoffs.std()) / math.sqrt(n_paths)

    return LocalVolHybridResult(
        price=price, std_error=std_err,
        n_paths=n_paths, n_steps=n_steps,
        mean_F=float(F_T.mean()), mean_U=float(U_T.mean()),
    )


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
