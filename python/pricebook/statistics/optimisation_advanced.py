"""Advanced optimisation: constrained, QP, ADMM, CMA-ES.

Phase M9 slices 199-201 consolidated.

* :func:`quadratic_program` — mean-variance portfolio optimisation.
* :func:`constrained_minimize` — SQP-style with equality/inequality constraints.
* :func:`admm_lasso` — ADMM for L1-regularised least squares.
* :func:`cma_es` — Covariance Matrix Adaptation Evolution Strategy.

References:
    Boyd & Vandenberghe, *Convex Optimization*, Cambridge, 2004.
    Nocedal & Wright, *Numerical Optimization*, Springer, 2006.
    Hansen, *The CMA Evolution Strategy: A Tutorial*, 2016.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- Quadratic Programming ----

@dataclass
class QPResult:
    """Result of a quadratic program."""
    x: np.ndarray
    objective: float
    converged: bool


def quadratic_program(
    H: np.ndarray | list[list[float]],
    c: np.ndarray | list[float],
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
) -> QPResult:
    """Solve min 0.5 x'Hx + c'x subject to A_eq x = b_eq.

    Uses the KKT system for equality-constrained QP:
        [H  A'] [x]   [-c ]
        [A  0 ] [λ] = [ b ]

    Args:
        H: (n, n) positive-definite Hessian.
        c: (n,) linear term.
        A_eq: (m, n) equality constraint matrix (or None).
        b_eq: (m,) equality RHS (or None).

    Returns:
        :class:`QPResult`.
    """
    H = np.asarray(H, dtype=float)
    c = np.asarray(c, dtype=float)
    n = len(c)

    if A_eq is None:
        # Unconstrained: x = -H^{-1} c
        try:
            x = np.linalg.solve(H, -c)
            obj = float(0.5 * x @ H @ x + c @ x)
            return QPResult(x, obj, True)
        except np.linalg.LinAlgError:
            return QPResult(np.zeros(n), 0.0, False)

    A = np.asarray(A_eq, dtype=float)
    b = np.asarray(b_eq, dtype=float)
    m = len(b)

    # KKT system
    K = np.zeros((n + m, n + m))
    K[:n, :n] = H
    K[:n, n:] = A.T
    K[n:, :n] = A

    rhs = np.zeros(n + m)
    rhs[:n] = -c
    rhs[n:] = b

    try:
        sol = np.linalg.solve(K, rhs)
        x = sol[:n]
        obj = float(0.5 * x @ H @ x + c @ x)
        return QPResult(x, obj, True)
    except np.linalg.LinAlgError:
        return QPResult(np.zeros(n), 0.0, False)


def markowitz_portfolio(
    expected_returns: np.ndarray | list[float],
    cov_matrix: np.ndarray | list[list[float]],
    target_return: float | None = None,
) -> QPResult:
    """Mean-variance portfolio optimisation (Markowitz).

    min 0.5 w' Σ w  subject to  1'w = 1 (and optionally μ'w = target).

    Args:
        expected_returns: (n,) expected return per asset.
        cov_matrix: (n, n) covariance matrix.
        target_return: if given, adds the return constraint.

    Returns:
        :class:`QPResult` with optimal weights.
    """
    mu = np.asarray(expected_returns, dtype=float)
    Sigma = np.asarray(cov_matrix, dtype=float)
    n = len(mu)

    if target_return is None:
        A = np.ones((1, n))
        b = np.array([1.0])
    else:
        A = np.vstack([np.ones(n), mu])
        b = np.array([1.0, target_return])

    return quadratic_program(Sigma, np.zeros(n), A, b)


# ---- Constrained minimisation (augmented Lagrangian) ----

@dataclass
class ConstrainedResult:
    """Result of constrained minimisation."""
    x: np.ndarray
    objective: float
    constraint_violation: float
    iterations: int
    converged: bool


def constrained_minimize(
    f: callable,
    x0: np.ndarray | list[float],
    constraints_eq: list[callable] | None = None,
    constraints_ineq: list[callable] | None = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    rho: float = 1.0,
    rho_max: float = 1e6,
) -> ConstrainedResult:
    """Augmented Lagrangian method for constrained optimisation.

    Solves: min f(x) s.t. g_i(x) = 0, h_j(x) ≤ 0.

    Converts to a sequence of unconstrained problems via:
        L_ρ(x, λ) = f(x) + Σ λ_i g_i(x) + (ρ/2) Σ g_i(x)²

    Uses scipy.optimize.minimize for the inner unconstrained problems.

    Args:
        f: objective function.
        x0: initial guess.
        constraints_eq: list of g_i(x) = 0 functions.
        constraints_ineq: list of h_j(x) ≤ 0 functions (converted via slack).
        rho: initial penalty parameter.
    """
    from scipy.optimize import minimize as sp_minimize

    x = np.asarray(x0, dtype=float)
    eq_fns = constraints_eq or []
    ineq_fns = constraints_ineq or []

    n_eq = len(eq_fns)
    n_ineq = len(ineq_fns)
    lam_eq = np.zeros(n_eq)
    lam_ineq = np.zeros(n_ineq)

    for iteration in range(max_iter):
        def augmented(x_flat):
            val = f(x_flat)
            for i, g in enumerate(eq_fns):
                gi = g(x_flat)
                val += lam_eq[i] * gi + 0.5 * rho * gi**2
            for j, h in enumerate(ineq_fns):
                hj = h(x_flat)
                # Penalty on max(h, -λ/ρ)
                slack = max(hj, -lam_ineq[j] / rho)
                val += lam_ineq[j] * slack + 0.5 * rho * slack**2
            return val

        result = sp_minimize(augmented, x, method='L-BFGS-B')
        x = result.x

        # Update multipliers
        violation = 0.0
        for i, g in enumerate(eq_fns):
            gi = g(x)
            lam_eq[i] += rho * gi
            violation += abs(gi)
        for j, h in enumerate(ineq_fns):
            hj = h(x)
            lam_ineq[j] = max(lam_ineq[j] + rho * hj, 0.0)
            violation += max(hj, 0.0)

        if violation < tol:
            return ConstrainedResult(x, float(f(x)), violation, iteration + 1, True)

        rho = min(rho * 2, rho_max)

    return ConstrainedResult(x, float(f(x)), violation, max_iter, False)


# ---- ADMM for LASSO ----

@dataclass
class ADMMResult:
    """Result of ADMM optimisation."""
    x: np.ndarray
    objective: float
    iterations: int
    converged: bool


def admm_lasso(
    A: np.ndarray | list[list[float]],
    b: np.ndarray | list[float],
    lam: float = 1.0,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> ADMMResult:
    """ADMM for LASSO: min 0.5||Ax − b||² + λ||x||₁.

    Splits into:
        x-update: (A'A + ρI)^{-1} (A'b + ρ(z − u))
        z-update: soft-threshold(x + u, λ/ρ)
        u-update: u + x − z

    Args:
        A: (m, n) design matrix.
        b: (m,) observation vector.
        lam: L1 regularisation parameter.
        rho: ADMM penalty parameter.

    Reference:
        Boyd et al., *Distributed Optimization via ADMM*, Found. Trends ML, 2011.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    # Precompute
    AtA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(AtA + rho * np.eye(n))

    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    def soft_threshold(v, kappa):
        return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

    for iteration in range(max_iter):
        # x-update
        rhs = Atb + rho * (z - u)
        x = np.linalg.solve(L.T, np.linalg.solve(L, rhs))

        # z-update
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)

        # u-update
        u = u + x - z

        # Convergence: primal and dual residuals
        r_norm = np.linalg.norm(x - z)
        s_norm = np.linalg.norm(rho * (z - z_old))
        if r_norm < tol and s_norm < tol:
            obj = 0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(x, 1)
            return ADMMResult(x, float(obj), iteration + 1, True)

    obj = 0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(x, 1)
    return ADMMResult(x, float(obj), max_iter, False)


# ---- CMA-ES ----

@dataclass
class CMAESResult:
    """Result of CMA-ES optimisation."""
    x: np.ndarray
    objective: float
    iterations: int
    n_evaluations: int


def cma_es(
    f: callable,
    x0: np.ndarray | list[float],
    sigma0: float = 0.5,
    max_iter: int = 1000,
    pop_size: int | None = None,
    tol: float = 1e-8,
    seed: int | None = None,
) -> CMAESResult:
    """Covariance Matrix Adaptation Evolution Strategy.

    Derivative-free optimiser. Samples a population from a multivariate
    Gaussian, evaluates f, selects the best, and updates the mean and
    covariance. Excellent for non-smooth, multi-modal objectives.

    Simplified (μ, λ)-CMA-ES with rank-1 + rank-μ covariance update.

    Args:
        f: objective function f(x) → float.
        x0: initial mean.
        sigma0: initial step size.
        pop_size: population size (default 4 + 3 ln(n)).
        tol: stop when sigma < tol.
        seed: random seed.

    Reference:
        Hansen, *The CMA Evolution Strategy: A Tutorial*, 2016.
    """
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)

    if pop_size is None:
        pop_size = 4 + int(3 * math.log(n))
    mu = pop_size // 2  # number of parents

    # Weights for recombination
    weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)])
    weights = weights / weights.sum()
    mu_eff = 1.0 / (weights ** 2).sum()

    # Adaptation parameters
    c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_1 = 2 / ((n + 1.3)**2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2)**2 + mu_eff))

    # State
    mean = x0.copy()
    sigma = sigma0
    C = np.eye(n)
    p_sigma = np.zeros(n)
    p_c = np.zeros(n)
    n_eval = 0
    chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

    best_x = x0.copy()
    best_f = f(x0)
    n_eval += 1

    for gen in range(max_iter):
        # Sample population
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C = np.eye(n)
            L = np.eye(n)

        population = []
        fitnesses = []
        for _ in range(pop_size):
            z = rng.standard_normal(n)
            x = mean + sigma * (L @ z)
            fx = f(x)
            n_eval += 1
            population.append((x, z, fx))
            fitnesses.append(fx)

        # Sort by fitness
        population.sort(key=lambda t: t[2])

        # Update best
        if population[0][2] < best_f:
            best_f = population[0][2]
            best_x = population[0][0].copy()

        # Recombination: weighted mean of mu best
        old_mean = mean.copy()
        mean = np.zeros(n)
        for i in range(mu):
            mean += weights[i] * population[i][0]

        # Evolution paths
        C_inv_sqrt = np.linalg.inv(L).T  # C^{-1/2} ≈ L^{-T}
        p_sigma = (1 - c_sigma) * p_sigma + math.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (mean - old_mean) / sigma
        h_sigma = 1 if np.linalg.norm(p_sigma) / math.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * chi_n else 0
        p_c = (1 - c_c) * p_c + h_sigma * math.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

        # Covariance update
        C = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
        for i in range(mu):
            y_i = (population[i][0] - old_mean) / sigma
            C += c_mu * weights[i] * np.outer(y_i, y_i)

        # Step size update
        sigma *= math.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))

        if sigma < tol:
            break

    return CMAESResult(best_x, best_f, gen + 1, n_eval)
