"""Frank-Wolfe (conditional gradient) method.

For smooth objectives over polytopes. Uses linear minimisation
oracle instead of projection.

* :func:`frank_wolfe` — standard Frank-Wolfe.
* :func:`frank_wolfe_portfolio` — portfolio optimisation via FW.

References:
    Frank & Wolfe, *An Algorithm for Quadratic Programming*, NRL, 1956.
    Jaggi, *Revisiting Frank-Wolfe*, ICML, 2013.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class FWResult:
    """Frank-Wolfe result."""
    x: np.ndarray
    objective: float
    duality_gap: float
    iterations: int
    converged: bool

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "duality_gap": self.duality_gap,
            "iterations": self.iterations,
            "converged": self.converged,
        }


def frank_wolfe(
    grad_f,
    lmo,
    x0: np.ndarray,
    f=None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> FWResult:
    """Frank-Wolfe (conditional gradient) method.

    min f(x)  s.t. x ∈ C (convex compact set)

    Each iteration:
    1. Compute gradient ∇f(x_k)
    2. Solve linear minimisation oracle: s_k = argmin_{s∈C} ⟨∇f(x_k), s⟩
    3. Update: x_{k+1} = x_k + γ_k (s_k − x_k)

    Step size: γ_k = 2/(k+2) (standard) or line search.

    Args:
        grad_f: gradient callable(x) → ∇f(x).
        lmo: linear minimisation oracle callable(d) → argmin_{s∈C} d's.
        x0: initial feasible point.
        f: objective (for gap computation).
        max_iter: maximum iterations.
        tol: duality gap tolerance.
    """
    x = x0.copy()
    n = len(x)

    for k in range(max_iter):
        g = grad_f(x)

        # Linear minimisation oracle
        s = lmo(g)

        # Duality gap: ⟨∇f(x), x − s⟩
        gap = float(g @ (x - s))

        if gap < tol:
            obj = float(f(x)) if f else 0
            return FWResult(x, obj, gap, k + 1, True)

        # Step size: 2/(k+2) or line search
        gamma = 2.0 / (k + 2)

        # Update
        x = x + gamma * (s - x)

    obj = float(f(x)) if f else 0
    return FWResult(x, obj, gap, max_iter, False)


def simplex_lmo(d: np.ndarray) -> np.ndarray:
    """Linear minimisation over probability simplex.

    argmin_{s∈Δ} d's = e_j where j = argmin d_j.
    (Put all weight on the cheapest coordinate.)
    """
    s = np.zeros(len(d))
    s[np.argmin(d)] = 1.0
    return s


def box_simplex_lmo(d: np.ndarray, upper: float = 1.0) -> np.ndarray:
    """LMO over {x ≥ 0, Σx = 1, x ≤ upper}.

    Greedy: fill cheapest coordinates up to upper bound.
    """
    n = len(d)
    s = np.zeros(n)
    order = np.argsort(d)
    remaining = 1.0
    for i in order:
        alloc = min(upper, remaining)
        s[i] = alloc
        remaining -= alloc
        if remaining <= 1e-12:
            break
    return s


def frank_wolfe_portfolio(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 1.0,
    max_weight: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> FWResult:
    """Portfolio optimisation via Frank-Wolfe.

    min −μ'w + (λ/2) w'Σw  s.t. w ∈ Δ, w ≤ max_weight

    Advantage over SLSQP: each iteration is O(n) not O(n³).
    Works well with many constraints (sector limits, ESG, etc).

    Args:
        mu: expected returns.
        cov: covariance matrix.
        risk_aversion: λ.
        max_weight: per-asset cap.
    """
    n = len(mu)

    def f(w):
        return -float(mu @ w) + 0.5 * risk_aversion * float(w @ cov @ w)

    def grad_f(w):
        return -mu + risk_aversion * cov @ w

    def lmo(d):
        return box_simplex_lmo(d, max_weight)

    w0 = np.ones(n) / n
    return frank_wolfe(grad_f, lmo, w0, f, max_iter, tol)
