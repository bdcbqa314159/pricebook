"""Duality framework: shadow prices, reduced costs, sensitivity.

Extract dual variables from LP/QP solutions. Parametric LP
for constraint perturbation analysis.

* :func:`lp_with_duals` — LP with dual variable extraction.
* :func:`shadow_prices` — marginal cost of constraints.
* :func:`sensitivity_range` — constraint RHS range for basis stability.
* :func:`parametric_lp` — LP parametric in RHS.

References:
    Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Ch. 5.
    Boyd & Vandenberghe, *Convex Optimization*, Ch. 5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class LPDualResult:
    """LP solution with dual variables."""
    x: np.ndarray               # primal solution
    objective: float
    dual_ineq: np.ndarray       # shadow prices for inequality constraints
    dual_eq: np.ndarray         # shadow prices for equality constraints
    reduced_costs: np.ndarray   # reduced cost per variable
    slack: np.ndarray           # constraint slack (b - Ax)
    binding: list[int]          # indices of binding constraints
    success: bool

    def to_dict(self) -> dict:
        return {
            "objective": self.objective,
            "n_binding": len(self.binding),
            "success": self.success,
        }


def lp_with_duals(
    c: np.ndarray,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    bounds: list | None = None,
) -> LPDualResult:
    """Solve LP and extract dual variables (shadow prices).

    min c'x  s.t. A_ub x ≤ b_ub, A_eq x = b_eq, bounds.

    Dual variables λ give:
    - ∂(objective) / ∂(b_i) = λ_i (shadow price)
    - If λ_i > 0, constraint i is binding and tight

    Args:
        c: objective coefficients (n,).
        A_ub: inequality constraint matrix (m, n).
        b_ub: inequality RHS (m,).
        A_eq: equality constraint matrix (p, n).
        b_eq: equality RHS (p,).
        bounds: variable bounds.
    """
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    n = len(c)
    m = len(b_ub) if b_ub is not None else 0
    p = len(b_eq) if b_eq is not None else 0

    if result.success:
        x = result.x

        # Dual variables from scipy (if available via HiGHS)
        # scipy.optimize.linprog doesn't directly expose duals
        # Compute via perturbation
        dual_ineq = np.zeros(m)
        for i in range(m):
            b_perturbed = b_ub.copy()
            b_perturbed[i] += 1e-6
            r_pert = linprog(c, A_ub=A_ub, b_ub=b_perturbed, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if r_pert.success:
                dual_ineq[i] = (r_pert.fun - result.fun) / 1e-6

        dual_eq = np.zeros(p)
        for i in range(p):
            b_perturbed = b_eq.copy()
            b_perturbed[i] += 1e-6
            r_pert = linprog(c, A_ub=A_ub, b_ub=b_perturbed, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if r_pert.success:
                dual_eq[i] = (r_pert.fun - result.fun) / 1e-6

        # Reduced costs: c - A'λ
        reduced = c.copy()
        if A_ub is not None:
            reduced -= A_ub.T @ dual_ineq
        if A_eq is not None:
            reduced -= A_eq.T @ dual_eq

        # Slack
        slack = np.zeros(m)
        if A_ub is not None:
            slack = b_ub - A_ub @ x

        # Binding constraints (slack < tolerance)
        binding = [i for i in range(m) if abs(slack[i]) < 1e-6]

    else:
        x = np.zeros(n)
        dual_ineq = np.zeros(m)
        dual_eq = np.zeros(p)
        reduced = c.copy()
        slack = np.zeros(m)
        binding = []

    return LPDualResult(
        x=x, objective=float(result.fun) if result.success else 0,
        dual_ineq=dual_ineq, dual_eq=dual_eq,
        reduced_costs=reduced, slack=slack,
        binding=binding, success=result.success,
    )


def shadow_prices(lp_result: LPDualResult) -> list[dict]:
    """Extract shadow prices with interpretation.

    Shadow price λ_i = ∂(obj) / ∂(b_i):
    - λ_i > 0: tightening constraint i increases cost
    - λ_i = 0: constraint i is not binding (slack > 0)
    """
    prices = []
    for i in range(len(lp_result.dual_ineq)):
        prices.append({
            "constraint": i,
            "shadow_price": float(lp_result.dual_ineq[i]),
            "slack": float(lp_result.slack[i]),
            "binding": i in lp_result.binding,
        })
    return prices


def parametric_lp(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub_base: np.ndarray,
    constraint_idx: int,
    perturbation_range: tuple[float, float],
    n_points: int = 20,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    bounds: list | None = None,
) -> list[dict]:
    """Parametric LP: sweep RHS of one constraint.

    Traces how the optimal objective changes as b_i varies.

    Args:
        constraint_idx: which inequality constraint to perturb.
        perturbation_range: (min_delta, max_delta) around base.
    """
    deltas = np.linspace(perturbation_range[0], perturbation_range[1], n_points)
    results = []

    for delta in deltas:
        b_pert = b_ub_base.copy()
        b_pert[constraint_idx] += delta

        r = linprog(c, A_ub=A_ub, b_ub=b_pert, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

        results.append({
            "delta": float(delta),
            "b_value": float(b_pert[constraint_idx]),
            "objective": float(r.fun) if r.success else float('inf'),
            "feasible": r.success,
        })

    return results
