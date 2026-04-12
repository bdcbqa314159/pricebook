"""Multi-dimensional root finding: Newton, Broyden, damped Newton.

Extends the 1D solvers in :mod:`pricebook.solvers` to N dimensions.
Essential for simultaneous curve calibration, multi-parameter model
fitting, and nonlinear system solving.

* :func:`newton_nd` — Newton-Raphson with analytical or finite-difference Jacobian.
* :func:`broyden` — Broyden's method (quasi-Newton, rank-1 Jacobian updates).
* :func:`damped_newton` — Newton with Armijo backtracking line search.
* :func:`finite_difference_jacobian` — numerical Jacobian computation.

References:
    Nocedal & Wright, *Numerical Optimization*, Springer, 2006, Ch. 11.
    Dennis & Schnabel, *Numerical Methods for Unconstrained Optimization
    and Nonlinear Equations*, SIAM, 1996.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class NDSolverResult:
    """Result of a multi-dimensional root-finding solve."""
    x: np.ndarray
    residual: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float


# ---- Finite-difference Jacobian ----

def finite_difference_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute the Jacobian of f at x via central finite differences.

    J[i,j] = ∂f_i / ∂x_j ≈ (f(x + e_j ε) − f(x − e_j ε)) / (2ε)

    Args:
        f: vector function R^n → R^m.
        x: point at which to evaluate the Jacobian.
        eps: perturbation size.

    Returns:
        (m, n) Jacobian matrix.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x), dtype=float)
    n = len(x)
    m = len(f0)
    J = np.zeros((m, n))
    for j in range(n):
        x_up = x.copy()
        x_dn = x.copy()
        x_up[j] += eps
        x_dn[j] -= eps
        J[:, j] = (np.asarray(f(x_up)) - np.asarray(f(x_dn))) / (2.0 * eps)
    return J


# ---- Newton-Raphson ----

def newton_nd(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | list[float],
    jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    fd_eps: float = 1e-8,
) -> NDSolverResult:
    """Multi-dimensional Newton-Raphson.

    Solves f(x) = 0 by iterating:
        J(x_k) · δ = −f(x_k)
        x_{k+1} = x_k + δ

    Args:
        f: vector function R^n → R^n.
        x0: initial guess.
        jacobian: function returning the (n, n) Jacobian at x.
            If None, uses finite differences.
        tol: convergence tolerance on ||f(x)||.
        max_iter: maximum iterations.
        fd_eps: perturbation for finite-difference Jacobian.

    Returns:
        :class:`NDSolverResult`.
    """
    x = np.asarray(x0, dtype=float).copy()

    for iteration in range(max_iter):
        fx = np.asarray(f(x), dtype=float)
        norm = float(np.linalg.norm(fx))
        if norm < tol:
            return NDSolverResult(x, fx, iteration, True, norm)

        if jacobian is not None:
            J = np.asarray(jacobian(x), dtype=float)
        else:
            J = finite_difference_jacobian(f, x, fd_eps)

        try:
            delta = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            break

        x = x + delta

    fx = np.asarray(f(x), dtype=float)
    return NDSolverResult(x, fx, max_iter, False, float(np.linalg.norm(fx)))


# ---- Broyden's method ----

def broyden(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | list[float],
    tol: float = 1e-10,
    max_iter: int = 100,
    fd_eps: float = 1e-8,
) -> NDSolverResult:
    """Broyden's method (quasi-Newton with rank-1 Jacobian updates).

    Avoids recomputing the full Jacobian at each step. Instead, updates
    an approximation B_k using the secant condition:

        B_{k+1} = B_k + (Δf − B_k Δx) Δx' / (Δx' Δx)

    Starts from a finite-difference Jacobian at x0.

    Args:
        f: vector function R^n → R^n.
        x0: initial guess.
        tol: convergence tolerance on ||f(x)||.
        max_iter: maximum iterations.
        fd_eps: perturbation for initial Jacobian.

    Returns:
        :class:`NDSolverResult`.
    """
    x = np.asarray(x0, dtype=float).copy()
    fx = np.asarray(f(x), dtype=float)
    norm = float(np.linalg.norm(fx))
    if norm < tol:
        return NDSolverResult(x, fx, 0, True, norm)

    # Initial Jacobian via finite differences
    B = finite_difference_jacobian(f, x, fd_eps)

    for iteration in range(max_iter):
        try:
            delta = np.linalg.solve(B, -fx)
        except np.linalg.LinAlgError:
            break

        x_new = x + delta
        fx_new = np.asarray(f(x_new), dtype=float)
        norm = float(np.linalg.norm(fx_new))

        if norm < tol:
            return NDSolverResult(x_new, fx_new, iteration + 1, True, norm)

        # Broyden rank-1 update
        df = fx_new - fx
        denom = delta @ delta
        if denom > 1e-30:
            B = B + np.outer(df - B @ delta, delta) / denom

        x = x_new
        fx = fx_new

    return NDSolverResult(x, fx, max_iter, False, float(np.linalg.norm(fx)))


# ---- Damped Newton (Armijo line search) ----

def damped_newton(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray | list[float],
    jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    fd_eps: float = 1e-8,
    alpha: float = 1e-4,
    rho: float = 0.5,
    max_backtrack: int = 20,
) -> NDSolverResult:
    """Newton with Armijo backtracking line search.

    At each step, finds the Newton direction δ, then backtracks the
    step size t until the Armijo condition is satisfied:

        ||f(x + t·δ)||² ≤ ||f(x)||² + 2·α·t·f(x)'·J·δ

    This prevents divergence when the Newton step is too aggressive.

    Args:
        f: vector function R^n → R^n.
        x0: initial guess.
        jacobian: analytical Jacobian (or None for FD).
        tol: convergence tolerance on ||f(x)||.
        max_iter: maximum outer iterations.
        alpha: Armijo sufficient decrease parameter.
        rho: backtracking contraction factor.
        max_backtrack: maximum line-search steps.

    Returns:
        :class:`NDSolverResult`.
    """
    x = np.asarray(x0, dtype=float).copy()

    for iteration in range(max_iter):
        fx = np.asarray(f(x), dtype=float)
        norm_sq = float(fx @ fx)
        if np.sqrt(norm_sq) < tol:
            return NDSolverResult(x, fx, iteration, True, np.sqrt(norm_sq))

        if jacobian is not None:
            J = np.asarray(jacobian(x), dtype=float)
        else:
            J = finite_difference_jacobian(f, x, fd_eps)

        try:
            delta = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            break

        # Directional derivative: f' · J · δ = f' · (-f) = -||f||²
        slope = float(fx @ (J @ delta))

        # Armijo backtracking
        t = 1.0
        for _ in range(max_backtrack):
            x_trial = x + t * delta
            fx_trial = np.asarray(f(x_trial), dtype=float)
            if float(fx_trial @ fx_trial) <= norm_sq + 2 * alpha * t * slope:
                break
            t *= rho

        x = x + t * delta

    fx = np.asarray(f(x), dtype=float)
    return NDSolverResult(x, fx, max_iter, False, float(np.linalg.norm(fx)))
