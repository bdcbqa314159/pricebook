"""Second-Order Cone Programming (SOCP) solver.

Solves: min c'x  s.t. ||A_i x + b_i|| ≤ c_i'x + d_i, Fx = g.

Applications: robust portfolio optimisation with norm constraints,
worst-case CVaR, tracking error minimisation.

* :func:`socp_solve` — general SOCP via interior-point.
* :func:`robust_portfolio_socp` — robust MV as SOCP.
* :func:`tracking_error_socp` — min tracking error with constraints.

References:
    Lobo et al., *Applications of Second-Order Cone Programming*, LAA, 1998.
    Goldfarb & Iyengar, *Robust Portfolio Selection Problems*, MOR, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class SOCPResult:
    """SOCP solution result."""
    x: np.ndarray
    objective: float
    feasible: bool
    iterations: int
    dual_gap: float

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "feasible": self.feasible,
            "iterations": self.iterations,
            "n_vars": len(self.x),
        }


def socp_solve(
    c: np.ndarray,
    cone_constraints: list[dict],
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> SOCPResult:
    """General SOCP via barrier method.

    min c'x
    s.t. ||A_i x + b_i||₂ ≤ c_i'x + d_i  for each cone constraint
         A_eq x = b_eq

    Each cone_constraint is a dict:
        {"A": (m, n), "b": (m,), "c_vec": (n,), "d": float}

    The barrier for each cone is:
        -log((c'x + d)² - ||Ax + b||²)

    Args:
        c: objective vector (n,).
        cone_constraints: list of SOC constraint dicts.
        A_eq: equality constraint matrix.
        b_eq: equality constraint RHS.
    """
    n = len(c)

    def barrier(x, t):
        """Log-barrier objective: t × c'x + Σ barrier_i."""
        obj = t * c @ x
        for cone in cone_constraints:
            A_i = cone["A"]
            b_i = cone["b"]
            c_i = cone["c_vec"]
            d_i = cone["d"]

            norm_val = np.linalg.norm(A_i @ x + b_i)
            slack = c_i @ x + d_i
            margin = slack**2 - norm_val**2
            if margin <= 0:
                return 1e20  # infeasible
            obj -= math.log(margin)
        return obj

    # Initial feasible point (heuristic)
    x0 = np.zeros(n)
    if A_eq is not None and b_eq is not None:
        # Least-norm solution to Ax = b
        try:
            x0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass

    # Barrier method: increase t each iteration
    t = 1.0
    mu = 10.0
    m = len(cone_constraints)
    x = x0.copy()

    constraints = []
    if A_eq is not None:
        constraints.append({"type": "eq", "fun": lambda x: A_eq @ x - b_eq})

    for iteration in range(max_iter):
        result = minimize(lambda x: barrier(x, t), x, method="SLSQP",
                         constraints=constraints, options={"maxiter": 100})
        x = result.x

        # Duality gap
        gap = m / t
        if gap < tol:
            break

        t *= mu

    obj = float(c @ x)
    feasible = all(
        np.linalg.norm(cone["A"] @ x + cone["b"]) <= cone["c_vec"] @ x + cone["d"] + 1e-6
        for cone in cone_constraints
    )

    return SOCPResult(x=x, objective=obj, feasible=feasible,
                      iterations=iteration + 1, dual_gap=gap)


def robust_portfolio_socp(
    mu: np.ndarray,
    cov: np.ndarray,
    epsilon: float = 0.1,
    target_return: float | None = None,
    long_only: bool = True,
) -> SOCPResult:
    """Robust mean-variance portfolio as SOCP.

    max μ'w − ε × ||Σ^{1/2} w||
    s.t. 1'w = 1, w ≥ 0

    Reformulated as SOCP:
    min −μ'w + ε × t
    s.t. ||Σ^{1/2} w|| ≤ t
         1'w = 1

    With auxiliary variable t for the cone.

    Args:
        epsilon: uncertainty radius.
        target_return: optional minimum return constraint.
    """
    N = len(mu)

    # Cholesky of covariance
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.eye(N) * np.sqrt(np.diag(cov))

    # Variables: [w_1, ..., w_N, t]
    n_vars = N + 1
    c_obj = np.zeros(n_vars)
    c_obj[:N] = -mu  # maximize return → minimize negative
    c_obj[N] = epsilon  # penalise risk

    # Cone: ||L'w|| ≤ t  →  A = L', b = 0, c_vec = [0,...,0, 1], d = 0
    cone = {
        "A": np.hstack([L.T, np.zeros((N, 1))]),
        "b": np.zeros(N),
        "c_vec": np.zeros(n_vars),
        "d": 0.0,
    }
    cone["c_vec"][N] = 1.0  # t is the cone slack

    # Equality: Σw = 1
    A_eq = np.zeros((1, n_vars))
    A_eq[0, :N] = 1.0
    b_eq = np.array([1.0])

    if target_return is not None:
        # Add: μ'w ≥ target  →  -μ'w ≤ -target
        A_ret = np.zeros((1, n_vars))
        A_ret[0, :N] = mu
        A_eq = np.vstack([A_eq, A_ret])
        b_eq = np.append(b_eq, target_return)

    # Solve via SLSQP (SOCP reformulation)
    def neg_obj(x):
        w = x[:N]
        t = x[N]
        return -mu @ w + epsilon * t

    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x[:N]) - 1},
        {"type": "ineq", "fun": lambda x: x[N] - np.linalg.norm(L.T @ x[:N])},  # t ≥ ||L'w||
    ]
    if target_return is not None:
        constraints.append({"type": "ineq", "fun": lambda x: mu @ x[:N] - target_return})

    bounds = [(0 if long_only else None, None)] * N + [(0, None)]

    x0 = np.ones(n_vars) / N
    x0[N] = 1.0
    result = minimize(neg_obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    x = result.x
    return SOCPResult(
        x=x[:N], objective=float(neg_obj(x)),
        feasible=result.success, iterations=result.nit, dual_gap=0.0,
    )


def tracking_error_socp(
    mu: np.ndarray,
    cov: np.ndarray,
    benchmark_weights: np.ndarray,
    max_te: float = 0.02,
    long_only: bool = True,
) -> SOCPResult:
    """Minimise tracking error vs benchmark as SOCP.

    min ||Σ^{1/2}(w − w_b)||
    s.t. 1'w = 1, w ≥ 0

    Args:
        benchmark_weights: target/benchmark weights.
        max_te: maximum tracking error constraint.
    """
    N = len(mu)

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        L = np.eye(N) * np.sqrt(np.diag(cov))

    # Minimize ||L'(w - w_b)||²  subject to constraints
    def te_objective(w):
        diff = w - benchmark_weights
        return float(diff @ cov @ diff)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    if max_te is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_te**2 - float((w - benchmark_weights) @ cov @ (w - benchmark_weights)),
        })

    bounds = [(0, 1)] * N if long_only else [(-1, 1)] * N
    x0 = benchmark_weights.copy()

    result = minimize(te_objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    w = result.x if result.success else benchmark_weights
    te = float(np.sqrt(max((w - benchmark_weights) @ cov @ (w - benchmark_weights), 0)))

    return SOCPResult(x=w, objective=te, feasible=result.success,
                      iterations=result.nit, dual_gap=0.0)
