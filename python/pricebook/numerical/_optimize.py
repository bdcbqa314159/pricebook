"""Unified optimisation: unconstrained, LP, QP, interior-point, proximal.

    from pricebook.numerical import minimize, linprog, qp, interior_point
    from pricebook.numerical import proximal_gradient, projection_simplex

Single entry point for all optimisation — scipy is the backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OptimizeResult:
    """Unified optimisation result."""
    x: np.ndarray
    fun: float
    iterations: int
    converged: bool
    method: str
    n_evaluations: int = 0

    def to_dict(self) -> dict:
        return {"method": self.method, "fun": self.fun,
                "iterations": self.iterations, "converged": self.converged}


# ═══════════════════════════════════════════════════════════════
# Unconstrained / box-constrained
# ═══════════════════════════════════════════════════════════════

def minimize(
    objective,
    x0: np.ndarray,
    method: str = "bfgs",
    bounds=None,
    gradient=None,
    tol: float = 1e-8,
    maxiter: int = 1000,
    **kwargs,
) -> OptimizeResult:
    """Unified minimiser.

    Methods: nelder_mead, bfgs, l_bfgs_b, cg, newton_cg,
             differential_evolution, basin_hopping, cma_es.
    """
    from scipy.optimize import minimize as _minimize, differential_evolution as _de, basinhopping as _bh

    method_map = {
        "nelder_mead": "Nelder-Mead",
        "bfgs": "BFGS",
        "l_bfgs_b": "L-BFGS-B",
        "cg": "CG",
        "newton_cg": "Newton-CG",
    }

    x0 = np.asarray(x0, dtype=float)

    if method == "cma_es":
        from pricebook.optimisation_advanced import cma_es
        r = cma_es(objective, x0, sigma0=kwargs.get("sigma0", 0.5),
                   tol=tol, max_iter=maxiter)
        return OptimizeResult(r.x, r.objective, r.iterations, r.converged,
                              "cma_es", r.n_evaluations)

    if method == "differential_evolution":
        if bounds is None:
            bounds = [(-10, 10)] * len(x0)
        r = _de(objective, bounds, tol=tol, maxiter=maxiter, seed=kwargs.get("seed", 42))
        return OptimizeResult(r.x, float(r.fun), r.nit, r.success,
                              "differential_evolution", r.nfev)

    if method == "basin_hopping":
        r = _bh(objective, x0, niter=maxiter, T=kwargs.get("T", 1.0),
                seed=kwargs.get("seed", 42))
        return OptimizeResult(r.x, float(r.fun), r.nit, True, "basin_hopping")

    scipy_method = method_map.get(method, method)
    r = _minimize(objective, x0, method=scipy_method, jac=gradient,
                  bounds=bounds, tol=tol, options={"maxiter": maxiter})
    return OptimizeResult(r.x, float(r.fun), r.nit if hasattr(r, 'nit') else 0,
                          r.success, method, r.nfev if hasattr(r, 'nfev') else 0)


# ═══════════════════════════════════════════════════════════════
# Linear programming
# ═══════════════════════════════════════════════════════════════

@dataclass
class LPResult:
    """Linear programming result."""
    x: np.ndarray
    fun: float
    converged: bool
    method: str = "highs"

    def to_dict(self) -> dict:
        return {"fun": self.fun, "converged": self.converged, "method": self.method}


def linprog(
    c: np.ndarray,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    bounds: list | None = None,
) -> LPResult:
    """Linear programming: min c'x s.t. A_ub x <= b_ub, A_eq x = b_eq.

    Uses scipy's HiGHS solver (industrial-strength LP/MIP).
    """
    from scipy.optimize import linprog as _linprog

    r = _linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                 bounds=bounds, method="highs")
    return LPResult(r.x, float(r.fun), r.success)


# ═══════════════════════════════════════════════════════════════
# Quadratic programming (with inequality constraints)
# ═══════════════════════════════════════════════════════════════

@dataclass
class QPResult:
    """Quadratic programming result."""
    x: np.ndarray
    fun: float
    converged: bool
    iterations: int

    def to_dict(self) -> dict:
        return {"fun": self.fun, "converged": self.converged, "iterations": self.iterations}


def qp(
    H: np.ndarray,
    c: np.ndarray,
    A_eq: np.ndarray | None = None,
    b_eq: np.ndarray | None = None,
    A_ub: np.ndarray | None = None,
    b_ub: np.ndarray | None = None,
    bounds: list | None = None,
) -> QPResult:
    """Quadratic programming: min 0.5 x'Hx + c'x s.t. constraints.

    Supports both equality and inequality constraints.
    Uses interior-point reduction to scipy.minimize for inequality case.
    """
    H = np.atleast_2d(H)
    c = np.asarray(c, dtype=float)
    n = len(c)

    # If no inequalities, use direct KKT (faster)
    if A_ub is None and bounds is None:
        from pricebook.optimisation_advanced import quadratic_program
        r = quadratic_program(H, c, A_eq, b_eq)
        return QPResult(r.x, r.objective, r.converged, 1)

    # With inequalities: use scipy SLSQP
    from scipy.optimize import minimize as _minimize

    def obj(x):
        return 0.5 * x @ H @ x + c @ x

    def grad(x):
        return H @ x + c

    constraints = []
    if A_eq is not None:
        constraints.append({"type": "eq", "fun": lambda x: A_eq @ x - b_eq,
                           "jac": lambda x: A_eq})
    if A_ub is not None:
        constraints.append({"type": "ineq", "fun": lambda x: b_ub - A_ub @ x,
                           "jac": lambda x: -A_ub})

    x0 = np.zeros(n)
    r = _minimize(obj, x0, jac=grad, method="SLSQP", bounds=bounds,
                  constraints=constraints, options={"maxiter": 500})

    return QPResult(r.x, float(r.fun), r.success, r.nit if hasattr(r, 'nit') else 0)


# ═══════════════════════════════════════════════════════════════
# Interior-point method (barrier)
# ═══════════════════════════════════════════════════════════════

def interior_point(
    objective,
    x0: np.ndarray,
    inequality_constraints=None,
    equality_constraints=None,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> OptimizeResult:
    """Interior-point method via logarithmic barrier.

    Converts inequality g_i(x) <= 0 to barrier: f(x) - (1/t) sum log(-g_i(x)).
    Increases t (barrier parameter) until constraints are tight.
    """
    from scipy.optimize import minimize as _minimize

    x = np.asarray(x0, dtype=float).copy()
    t = 1.0
    mu = 10.0  # barrier growth rate

    for outer in range(maxiter):
        # Barrier subproblem
        def barrier_obj(x_inner):
            val = t * objective(x_inner)
            if inequality_constraints:
                for g in inequality_constraints:
                    gi = g(x_inner)
                    if gi >= 0:
                        return 1e15
                    val -= math.log(-gi)
            return val

        constraints = []
        if equality_constraints:
            for h in equality_constraints:
                constraints.append({"type": "eq", "fun": h})

        r = _minimize(barrier_obj, x, method="BFGS",
                      options={"maxiter": 50, "gtol": tol / t})
        x = r.x

        # Check convergence
        n_ineq = len(inequality_constraints) if inequality_constraints else 0
        if n_ineq > 0 and n_ineq / t < tol:
            break

        t *= mu

    return OptimizeResult(x, float(objective(x)), outer + 1, True,
                          "interior_point")


# ═══════════════════════════════════════════════════════════════
# Proximal methods
# ═══════════════════════════════════════════════════════════════

def proximal_gradient(
    grad_f,
    prox_g,
    x0: np.ndarray,
    step_size: float = 0.01,
    maxiter: int = 1000,
    tol: float = 1e-6,
    accelerated: bool = True,
) -> OptimizeResult:
    """Proximal gradient descent (ISTA / FISTA).

    Solves: min f(x) + g(x) where f is smooth (gradient available)
    and g has a computable proximal operator.

    Args:
        grad_f: gradient of the smooth part.
        prox_g: proximal operator of the non-smooth part: prox_{t*g}(x).
        accelerated: True for FISTA (Nesterov acceleration).
    """
    x = np.asarray(x0, dtype=float).copy()
    y = x.copy()
    t_k = 1.0

    for k in range(maxiter):
        x_old = x.copy()
        g = np.asarray(grad_f(y))
        x = np.asarray(prox_g(y - step_size * g, step_size))

        if np.max(np.abs(x - x_old)) < tol:
            break

        if accelerated:
            t_new = (1 + math.sqrt(1 + 4 * t_k ** 2)) / 2
            y = x + (t_k - 1) / t_new * (x - x_old)
            t_k = t_new
        else:
            y = x

    return OptimizeResult(x, 0.0, k + 1, True,
                          "fista" if accelerated else "ista")


# ═══════════════════════════════════════════════════════════════
# Projection operators
# ═══════════════════════════════════════════════════════════════

def projection_simplex(x: np.ndarray) -> np.ndarray:
    """Project onto the probability simplex: {x >= 0, sum(x) = 1}.

    Michelot (1986) algorithm: O(n log n).
    """
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u * np.arange(1, n + 1) > cssv)[0][-1]
    theta = cssv[rho] / (rho + 1)
    return np.maximum(x - theta, 0)


def projection_l1_ball(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Project onto the L1 ball: {x : ||x||_1 <= radius}."""
    if np.sum(np.abs(x)) <= radius:
        return x
    u = np.abs(x)
    proj = projection_simplex(u / radius) * radius
    return np.sign(x) * proj


def soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    """Soft thresholding (proximal operator of L1 norm)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)
