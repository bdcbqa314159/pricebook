"""Convexity verification and cardinality-constrained optimisation.

* :func:`is_convex` — check if objective is convex via Hessian sampling.
* :func:`verify_kkt` — verify KKT conditions at a solution.
* :func:`cardinality_portfolio` — portfolio with max N assets.

References:
    Boyd & Vandenberghe, *Convex Optimization*, Ch. 3-5.
    Bertsimas & Shioda, *Algorithm for Cardinality-Constrained QP*, COR, 2009.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class ConvexityCheckResult:
    """Convexity verification result."""
    is_convex: bool
    min_eigenvalue: float       # smallest eigenvalue of sampled Hessians
    n_samples: int
    failed_at: np.ndarray | None  # point where non-convexity detected

    def to_dict(self) -> dict:
        return {
            "is_convex": self.is_convex,
            "min_eigenvalue": self.min_eigenvalue,
            "n_samples": self.n_samples,
        }


def is_convex(
    f,
    domain: list[tuple[float, float]],
    n_samples: int = 100,
    tol: float = -1e-6,
    seed: int = 42,
) -> ConvexityCheckResult:
    """Check if f is convex by sampling Hessian eigenvalues.

    f is convex iff H(x) ≽ 0 (positive semidefinite) for all x in domain.
    We sample random points and check minimum eigenvalue ≥ 0.

    Args:
        f: scalar function ℝⁿ → ℝ.
        domain: list of (lower, upper) bounds per dimension.
        n_samples: number of random Hessian evaluations.
        tol: eigenvalue tolerance (< 0 allows small numerical error).
    """
    from pricebook.numerical._differentiate import hessian

    rng = np.random.default_rng(seed)
    n = len(domain)
    min_eig = float('inf')
    failed = None

    for _ in range(n_samples):
        x = np.array([rng.uniform(lo, hi) for lo, hi in domain])
        H_result = hessian(f, x)
        H_matrix = H_result.value if hasattr(H_result, 'value') else H_result
        eigs = np.linalg.eigvalsh(H_matrix)
        min_e = float(np.min(eigs))

        if min_e < min_eig:
            min_eig = min_e
        if min_e < tol:
            failed = x.copy()
            return ConvexityCheckResult(False, min_eig, n_samples, failed)

    return ConvexityCheckResult(True, min_eig, n_samples, None)


@dataclass
class KKTCheckResult:
    """KKT condition verification."""
    stationarity_violation: float
    complementarity_violation: float
    primal_feasibility: float
    dual_feasibility: float
    all_satisfied: bool

    def to_dict(self) -> dict:
        return dict(vars(self))


def verify_kkt(
    grad_f: np.ndarray,
    x: np.ndarray,
    constraints_ineq: list[dict] | None = None,
    constraints_eq: list[dict] | None = None,
    lambdas_ineq: np.ndarray | None = None,
    lambdas_eq: np.ndarray | None = None,
    tol: float = 1e-4,
) -> KKTCheckResult:
    """Verify KKT conditions at a candidate solution.

    KKT conditions:
    1. Stationarity: ∇f + Σ λ_i ∇g_i + Σ μ_j ∇h_j = 0
    2. Primal feasibility: g_i(x) ≤ 0, h_j(x) = 0
    3. Dual feasibility: λ_i ≥ 0
    4. Complementary slackness: λ_i g_i(x) = 0

    Args:
        grad_f: gradient of objective at x.
        x: candidate solution.
        constraints_ineq: list of {"fun": g_i, "grad": ∇g_i} (g_i ≤ 0).
        constraints_eq: list of {"fun": h_j, "grad": ∇h_j} (h_j = 0).
        lambdas_ineq: dual variables for inequality constraints.
        lambdas_eq: dual variables for equality constraints.
    """
    n = len(x)
    stat = grad_f.copy()

    # Add constraint gradients
    if constraints_ineq and lambdas_ineq is not None:
        for i, c in enumerate(constraints_ineq):
            if "grad" in c:
                stat += lambdas_ineq[i] * c["grad"](x)

    if constraints_eq and lambdas_eq is not None:
        for j, c in enumerate(constraints_eq):
            if "grad" in c:
                stat += lambdas_eq[j] * c["grad"](x)

    stationarity = float(np.linalg.norm(stat))

    # Primal feasibility
    pf = 0.0
    if constraints_ineq:
        for c in constraints_ineq:
            pf = max(pf, max(c["fun"](x), 0))
    if constraints_eq:
        for c in constraints_eq:
            pf = max(pf, abs(c["fun"](x)))

    # Dual feasibility
    df = 0.0
    if lambdas_ineq is not None:
        df = max(0, -float(np.min(lambdas_ineq)))

    # Complementary slackness
    cs = 0.0
    if constraints_ineq and lambdas_ineq is not None:
        for i, c in enumerate(constraints_ineq):
            cs = max(cs, abs(lambdas_ineq[i] * c["fun"](x)))

    all_ok = stationarity < tol and pf < tol and df < tol and cs < tol

    return KKTCheckResult(stationarity, cs, pf, df, all_ok)


@dataclass
class CardinalityPortfolioResult:
    """Cardinality-constrained portfolio result."""
    weights: np.ndarray
    n_active: int
    objective: float
    selected_assets: list[int]

    def to_dict(self) -> dict:
        return {
            "n_active": self.n_active,
            "objective": self.objective,
            "selected_assets": self.selected_assets,
        }


def cardinality_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    max_assets: int,
    risk_aversion: float = 1.0,
    long_only: bool = True,
) -> CardinalityPortfolioResult:
    """Portfolio with maximum N assets (cardinality constraint).

    Solved via greedy forward selection + local optimisation.
    Not globally optimal but fast and practical.

    Args:
        max_assets: maximum number of non-zero weights.
        risk_aversion: λ in mean-variance.
    """
    N = len(mu)
    max_k = min(max_assets, N)

    # Greedy forward selection: add one asset at a time
    selected = []
    best_obj = -float('inf')

    for _ in range(max_k):
        best_add = -1
        best_val = -float('inf')

        for j in range(N):
            if j in selected:
                continue
            trial = selected + [j]
            w, obj = _solve_subset(mu, cov, trial, risk_aversion, long_only)
            if obj > best_val:
                best_val = obj
                best_add = j

        if best_add >= 0:
            selected.append(best_add)
            best_obj = best_val

    # Final optimisation on selected subset
    w_sub, obj = _solve_subset(mu, cov, selected, risk_aversion, long_only)

    weights = np.zeros(N)
    for i, idx in enumerate(selected):
        weights[idx] = w_sub[i]

    return CardinalityPortfolioResult(
        weights=weights,
        n_active=int(np.sum(np.abs(weights) > 1e-8)),
        objective=obj,
        selected_assets=sorted(selected),
    )


def _solve_subset(mu, cov, indices, risk_aversion, long_only):
    """Solve MV on a subset of assets."""
    k = len(indices)
    mu_sub = mu[indices]
    cov_sub = cov[np.ix_(indices, indices)]

    def neg_obj(w):
        return -(float(mu_sub @ w) - 0.5 * risk_aversion * float(w @ cov_sub @ w))

    w0 = np.ones(k) / k
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * k if long_only else [(-1, 1)] * k

    result = minimize(neg_obj, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    w = result.x if result.success else w0
    return w, -result.fun if result.success else 0.0
