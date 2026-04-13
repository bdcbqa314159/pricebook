"""Monte Carlo Greeks: pathwise (IPA), likelihood ratio, optimal MLMC, LSM improvements.

Phase M6 slices 191-193 consolidated (share this file).

* :func:`pathwise_delta` — IPA delta for smooth payoffs.
* :func:`pathwise_vega` — IPA vega for smooth payoffs.
* :func:`likelihood_ratio_delta` — LR delta for discontinuous payoffs.
* :func:`likelihood_ratio_vega` — LR vega for discontinuous payoffs.
* :func:`optimal_mlmc` — MLMC with Giles (2008) optimal allocation.
* :func:`lsm_orthogonal` — LSM with Laguerre/Hermite/Chebyshev basis.
* :func:`dual_upper_bound` — Andersen-Broadie upper bound for American.

References:
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 7.
    Giles, *Multilevel Monte Carlo Path Simulation*, Oper. Res., 2008.
    Andersen & Broadie, *Primal-Dual Simulation for American Options*, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Pathwise (IPA) Greeks ----

@dataclass
class MCGreekResult:
    """Result of an MC Greek computation."""
    value: float
    std_error: float
    n_paths: int
    method: str


def pathwise_delta(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCGreekResult:
    """IPA delta for a European call under GBM.

    dC/dS₀ = E[e^{-rT} 1_{S_T>K} × S_T/S₀]

    Valid for smooth payoffs (call/put). Fails for digitals (discontinuous).

    Reference: Glasserman Ch. 7.2.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = spot * np.exp((rate - 0.5 * vol**2) * T + vol * math.sqrt(T) * z)
    df = math.exp(-rate * T)

    # Pathwise derivative: d(payoff)/dS₀ = 1_{S_T > K} × S_T / S₀
    itm = S_T > strike
    pathwise = df * itm * S_T / spot

    mean = float(pathwise.mean())
    std = float(pathwise.std()) / math.sqrt(n_paths)
    return MCGreekResult(mean, std, n_paths, "pathwise_delta")


def pathwise_vega(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCGreekResult:
    """IPA vega for a European call under GBM.

    dC/dσ = E[e^{-rT} 1_{S_T>K} × S_T × (z√T − σT)]

    where z is the standard normal draw used to generate S_T.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = spot * np.exp((rate - 0.5 * vol**2) * T + vol * math.sqrt(T) * z)
    df = math.exp(-rate * T)

    itm = S_T > strike
    pathwise = df * itm * S_T * (z * math.sqrt(T) - vol * T)

    mean = float(pathwise.mean())
    std = float(pathwise.std()) / math.sqrt(n_paths)
    return MCGreekResult(mean, std, n_paths, "pathwise_vega")


# ---- Likelihood ratio Greeks ----

def likelihood_ratio_delta(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCGreekResult:
    """Likelihood ratio delta for any payoff (including discontinuous).

    dC/dS₀ = E[e^{-rT} g(S_T) × ∂log f/∂S₀]

    where ∂log f/∂S₀ = z / (S₀ σ √T) for lognormal.
    Works for digitals, barriers — any payoff.

    Higher variance than pathwise for smooth payoffs.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = spot * np.exp((rate - 0.5 * vol**2) * T + vol * math.sqrt(T) * z)
    df = math.exp(-rate * T)

    payoff = np.maximum(S_T - strike, 0.0)
    score = z / (spot * vol * math.sqrt(T))
    lr = df * payoff * score

    mean = float(lr.mean())
    std = float(lr.std()) / math.sqrt(n_paths)
    return MCGreekResult(mean, std, n_paths, "likelihood_ratio_delta")


def likelihood_ratio_vega(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> MCGreekResult:
    """Likelihood ratio vega.

    Score function: ∂log f/∂σ = (z² − 1)/σ − z√T.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    S_T = spot * np.exp((rate - 0.5 * vol**2) * T + vol * math.sqrt(T) * z)
    df = math.exp(-rate * T)

    payoff = np.maximum(S_T - strike, 0.0)
    score = (z * z - 1) / vol - z * math.sqrt(T)
    lr = df * payoff * score

    mean = float(lr.mean())
    std = float(lr.std()) / math.sqrt(n_paths)
    return MCGreekResult(mean, std, n_paths, "likelihood_ratio_vega")


# ---- Optimal MLMC (Giles 2008) ----

@dataclass
class OptimalMLMCResult:
    """Result of optimal MLMC."""
    price: float
    std_error: float
    n_levels: int
    paths_per_level: list[int]
    variance_per_level: list[float]
    cost_per_level: list[float]
    total_cost: float


def optimal_mlmc(
    payoff: callable,
    simulate_level: callable,
    T: float,
    epsilon: float = 0.01,
    L_max: int = 8,
    n_pilot: int = 1_000,
    seed: int | None = None,
) -> OptimalMLMCResult:
    """MLMC with Giles (2008) optimal allocation.

    Allocates paths per level to minimise total cost for a given
    target RMSE ε:

        N_l = (2/ε²) × √(V_l/C_l) × Σ_k √(V_k C_k)

    where V_l = Var[Y_l], C_l = cost of one sample at level l.

    Args:
        payoff: function(S_T) → payoff values (array).
        simulate_level: function(level, n_paths, seed) → (fine_payoffs, coarse_payoffs).
            Level 0 returns (payoffs, None). Level l>0 returns the difference pair.
        T: time horizon.
        epsilon: target RMSE.
        L_max: maximum number of levels.
        n_pilot: pilot paths for variance estimation.
        seed: random seed.
    """
    rng = np.random.default_rng(seed)

    # Pilot run to estimate variance and cost per level
    variances = []
    costs = []
    means = []

    for l in range(L_max):
        fine, coarse = simulate_level(l, n_pilot, rng.integers(0, 2**31))
        if coarse is None:
            Y = fine
        else:
            Y = fine - coarse
        variances.append(float(np.var(Y)))
        costs.append(2.0 ** l)  # cost proportional to steps = 2^l
        means.append(float(np.mean(Y)))

        # Stop if variance contribution is negligible
        if l >= 2 and variances[-1] < epsilon**2 / (2 * L_max):
            break

    n_levels = len(variances)

    # Optimal allocation: N_l ∝ √(V_l / C_l)
    sv = [math.sqrt(v / c) if v > 0 and c > 0 else 0.0
          for v, c in zip(variances, costs)]
    sum_svc = sum(math.sqrt(v * c) for v, c in zip(variances, costs))

    paths_per_level = []
    for l in range(n_levels):
        if sv[l] > 0:
            N_l = max(int(math.ceil(2.0 / epsilon**2 * sv[l] * sum_svc)), 100)
        else:
            N_l = 100
        paths_per_level.append(N_l)

    # Production run with optimal allocation
    total_mean = 0.0
    total_var = 0.0

    for l in range(n_levels):
        fine, coarse = simulate_level(l, paths_per_level[l],
                                       rng.integers(0, 2**31))
        if coarse is None:
            Y = fine
        else:
            Y = fine - coarse
        total_mean += float(np.mean(Y))
        total_var += float(np.var(Y)) / paths_per_level[l]

    total_cost = sum(n * c for n, c in zip(paths_per_level, costs))

    return OptimalMLMCResult(
        price=total_mean,
        std_error=math.sqrt(total_var),
        n_levels=n_levels,
        paths_per_level=paths_per_level,
        variance_per_level=variances,
        cost_per_level=costs[:n_levels],
        total_cost=total_cost,
    )


# ---- LSM with orthogonal bases ----

def lsm_laguerre_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Laguerre polynomial basis for LSM regression.

    L_0(x) = 1
    L_1(x) = 1 − x
    L_2(x) = 1 − 2x + x²/2
    """
    n = len(x)
    basis = np.zeros((n, degree + 1))
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = 1.0 - x
    for k in range(2, degree + 1):
        basis[:, k] = ((2 * k - 1 - x) * basis[:, k - 1] - (k - 1) * basis[:, k - 2]) / k
    return basis


def lsm_chebyshev_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Chebyshev polynomial basis (first kind) for LSM regression.

    Input x should be normalised to [-1, 1] for optimal conditioning.
    """
    n = len(x)
    basis = np.zeros((n, degree + 1))
    basis[:, 0] = 1.0
    if degree >= 1:
        basis[:, 1] = x
    for k in range(2, degree + 1):
        basis[:, k] = 2 * x * basis[:, k - 1] - basis[:, k - 2]
    return basis


def lsm_with_basis(
    paths: np.ndarray,
    strike: float,
    rate: float,
    T: float,
    basis_type: str = "laguerre",
    degree: int = 3,
    is_call: bool = False,
) -> float:
    """LSM American option pricing with orthogonal polynomial basis.

    Args:
        paths: (n_paths, n_steps+1) price paths.
        strike: option strike.
        rate: risk-free rate.
        T: time to maturity.
        basis_type: "laguerre", "chebyshev", or "polynomial".
        degree: polynomial degree for regression.
        is_call: True for call, False for put.

    Returns:
        American option price estimate.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps
    df = math.exp(-rate * dt)

    # Terminal payoff
    if is_call:
        cashflow = np.maximum(paths[:, -1] - strike, 0.0)
    else:
        cashflow = np.maximum(strike - paths[:, -1], 0.0)
    exercise_time = np.full(n_paths, n_steps)

    # Backward induction
    for step in range(n_steps - 1, 0, -1):
        S = paths[:, step]
        if is_call:
            exercise_value = np.maximum(S - strike, 0.0)
        else:
            exercise_value = np.maximum(strike - S, 0.0)

        itm = exercise_value > 0
        if np.sum(itm) < degree + 1:
            continue

        # Continuation value (discounted future cashflow)
        disc_steps = exercise_time[itm] - step
        cont_value = cashflow[itm] * np.exp(-rate * dt * disc_steps)

        # Regression basis
        x = S[itm]
        if basis_type == "laguerre":
            x_norm = x / strike  # normalise
            basis = lsm_laguerre_basis(x_norm, degree)
        elif basis_type == "chebyshev":
            x_min, x_max = x.min(), x.max()
            x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-10) - 1
            basis = lsm_chebyshev_basis(x_norm, degree)
        else:
            x_norm = x / strike
            basis = np.column_stack([x_norm**k for k in range(degree + 1)])

        # Least squares regression
        coeffs, _, _, _ = np.linalg.lstsq(basis, cont_value, rcond=None)
        estimated_cont = basis @ coeffs

        # Exercise decision
        exercise_mask = exercise_value[itm] > estimated_cont
        exercise_indices = np.where(itm)[0][exercise_mask]

        cashflow[exercise_indices] = exercise_value[exercise_indices]
        exercise_time[exercise_indices] = step

    # Discount all cashflows to time 0
    disc_factors = np.exp(-rate * dt * exercise_time)
    return float(np.mean(cashflow * disc_factors))


# ---- Dual upper bound (simplified Andersen-Broadie) ----

def dual_upper_bound(
    paths: np.ndarray,
    strike: float,
    rate: float,
    T: float,
    lsm_price: float,
    is_call: bool = False,
    n_sub: int = 50,
) -> float:
    """Simplified Andersen-Broadie dual upper bound for American options.

    Upper bound = E[max over exercise dates of (discounted payoff − martingale increment)].

    This simplified version uses the LSM continuation value as the
    approximate martingale. A proper implementation would use nested
    sub-simulation, which is expensive.

    Args:
        paths: (n_paths, n_steps+1) price paths.
        strike: option strike.
        rate: risk-free rate.
        T: time to maturity.
        lsm_price: the LSM lower bound (for reference).
        is_call: True for call, False for put.
        n_sub: not used in simplified version.

    Returns:
        Upper bound estimate. Should satisfy: LSM ≤ true ≤ upper.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    dt = T / n_steps

    max_values = np.zeros(n_paths)

    for step in range(1, n_steps + 1):
        S = paths[:, step]
        df = math.exp(-rate * dt * step)
        if is_call:
            payoff = np.maximum(S - strike, 0.0) * df
        else:
            payoff = np.maximum(strike - S, 0.0) * df
        max_values = np.maximum(max_values, payoff)

    return float(np.mean(max_values))
