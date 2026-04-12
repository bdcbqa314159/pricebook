"""Numerical safety: CFL checking, martingale testing, convergence diagnostics.

Prevents silent numerical failures in PDE solvers and Monte Carlo schemes.

* :func:`check_cfl` — verify CFL condition for explicit FD schemes.
* :func:`check_feller` — verify Feller condition for CIR/Heston variance.
* :func:`martingale_test` — verify E[e^{-rT} S_T] = S_0 for MC schemes.
* :func:`convergence_rate` — empirical convergence order from multiple resolutions.
* :func:`strong_convergence_test` — compare SDE paths at fine vs coarse steps.
* :func:`weak_convergence_test` — compare terminal distributions at multiple dt.

References:
    Lax & Richtmyer, *Survey of the Stability of Linear Finite Difference
    Equations*, Comm. Pure Appl. Math., 1956.
    Feller, *Two Singular Diffusion Problems*, Ann. Math., 1951.
    Glasserman, *Monte Carlo Methods in Financial Engineering*, Ch. 6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---- CFL condition ----

@dataclass
class CFLResult:
    """CFL condition check result."""
    dt: float
    dx: float
    dt_max: float
    ratio: float          # dt / dt_max (< 1 is stable)
    is_stable: bool
    recommendation: str


def check_cfl(
    vol: float,
    rate: float,
    dt: float,
    dx: float,
    div_yield: float = 0.0,
) -> CFLResult:
    """Check the CFL stability condition for an explicit FD scheme.

    For the Black-Scholes PDE in log-spot space, the explicit scheme
    is stable when:

        Δt ≤ Δx² / (σ² + |μ|·Δx)

    where μ = r − q − σ²/2. A stricter sufficient condition is:

        Δt ≤ Δx² / σ²    (ignoring advection)

    Args:
        vol: volatility σ.
        rate: risk-free rate r.
        dt: time step Δt.
        dx: spatial step Δx (in log-spot).
        div_yield: continuous dividend yield q.

    Returns:
        :class:`CFLResult` with stability diagnosis.
    """
    if vol <= 0 or dx <= 0:
        return CFLResult(dt, dx, float("inf"), 0.0, True,
                         "degenerate inputs — no diffusion")

    mu = abs(rate - div_yield - 0.5 * vol * vol)
    alpha = vol * vol
    dt_max = dx * dx / (alpha + mu * dx)

    ratio = dt / dt_max if dt_max > 0 else float("inf")
    stable = ratio <= 1.0

    if stable:
        rec = f"stable (dt/dt_max = {ratio:.3f})"
    else:
        rec = (
            f"UNSTABLE: dt={dt:.2e} exceeds CFL limit dt_max={dt_max:.2e}. "
            f"Reduce dt or increase dx. Ratio = {ratio:.1f}x."
        )

    return CFLResult(dt, dx, dt_max, ratio, stable, rec)


# ---- Feller condition ----

@dataclass
class FellerResult:
    """Feller condition check for CIR/Heston variance process."""
    kappa: float
    theta: float
    xi: float
    lhs: float       # 2κθ
    rhs: float       # ξ²
    is_satisfied: bool
    recommendation: str


def check_feller(
    kappa: float,
    theta: float,
    xi: float,
) -> FellerResult:
    """Check the Feller condition: 2κθ ≥ ξ².

    When satisfied, the CIR/Heston variance process stays strictly
    positive. When violated, the process can hit zero, causing
    numerical issues in Euler discretisation (negative variance).

    Args:
        kappa: mean-reversion speed.
        theta: long-run variance level.
        xi: vol-of-vol.
    """
    lhs = 2 * kappa * theta
    rhs = xi * xi
    satisfied = lhs >= rhs

    if satisfied:
        rec = f"Feller satisfied: 2κθ={lhs:.4f} ≥ ξ²={rhs:.4f}"
    else:
        rec = (
            f"Feller VIOLATED: 2κθ={lhs:.4f} < ξ²={rhs:.4f}. "
            "Variance can hit zero — use QE scheme or exact simulation."
        )

    return FellerResult(kappa, theta, xi, lhs, rhs, satisfied, rec)


# ---- Martingale test ----

@dataclass
class MartingaleTestResult:
    """Result of a martingale test on an MC scheme."""
    expected: float         # S_0
    simulated_mean: float   # E[e^{-rT} S_T]
    relative_error: float
    passed: bool
    n_paths: int
    std_error: float


def martingale_test(
    terminal_values: np.ndarray | list[float],
    spot: float,
    rate: float,
    T: float,
    tol: float = 0.01,
) -> MartingaleTestResult:
    """Verify the fundamental martingale property: E[e^{-rT} S_T] = S_0.

    This is the single most useful debugging tool for MC implementations.
    It catches drift errors, discretisation bias, and measure mistakes.

    Args:
        terminal_values: array of S_T values from MC simulation.
        spot: initial spot S_0.
        rate: risk-free rate r.
        T: time to maturity.
        tol: relative tolerance for pass/fail (default 1%).

    Returns:
        :class:`MartingaleTestResult`.
    """
    S_T = np.asarray(terminal_values, dtype=float)
    n = len(S_T)
    df = math.exp(-rate * T)
    discounted = df * S_T
    mean = float(discounted.mean())
    std = float(discounted.std()) / math.sqrt(n) if n > 1 else 0.0

    rel_err = abs(mean - spot) / spot if spot != 0 else abs(mean)
    passed = rel_err < tol

    return MartingaleTestResult(
        expected=spot,
        simulated_mean=mean,
        relative_error=rel_err,
        passed=passed,
        n_paths=n,
        std_error=std,
    )


# ---- Convergence rate estimation ----

@dataclass
class ConvergenceResult:
    """Empirical convergence rate from multiple resolutions."""
    resolutions: list[float]
    errors: list[float]
    estimated_order: float
    is_consistent: bool    # estimated order close to expected?
    expected_order: float


def convergence_rate(
    resolutions: list[float],
    errors: list[float],
    expected_order: float | None = None,
    order_tol: float = 0.3,
) -> ConvergenceResult:
    """Estimate the convergence order from errors at multiple resolutions.

    Fits log(error) = order × log(h) + const via linear regression.

    Args:
        resolutions: list of step sizes h (decreasing).
        errors: list of corresponding errors (should decrease).
        expected_order: if given, checks whether estimated order is close.
        order_tol: tolerance for consistency check (default ±0.3).

    Returns:
        :class:`ConvergenceResult`.
    """
    h = np.log(np.asarray(resolutions, dtype=float))
    e = np.log(np.maximum(np.asarray(errors, dtype=float), 1e-300))

    if len(h) < 2:
        return ConvergenceResult(
            list(resolutions), list(errors), 0.0, False,
            expected_order or 0.0,
        )

    # Linear regression: e = order * h + const
    A = np.vstack([h, np.ones_like(h)]).T
    result = np.linalg.lstsq(A, e, rcond=None)
    order = float(result[0][0])

    exp = expected_order or order
    consistent = abs(order - exp) < order_tol

    return ConvergenceResult(
        list(resolutions), list(errors), order, consistent, exp,
    )


# ---- Strong convergence test ----

def strong_convergence_test(
    fine_paths: np.ndarray,
    coarse_paths: np.ndarray,
) -> float:
    """Compute strong error: E[|X_fine(T) - X_coarse(T)|].

    Both arrays should have shape (n_paths,) with terminal values
    from the same Brownian draws at different step sizes.

    Returns:
        Mean absolute difference (strong error estimate).
    """
    fine = np.asarray(fine_paths, dtype=float)
    coarse = np.asarray(coarse_paths, dtype=float)
    return float(np.mean(np.abs(fine - coarse)))


# ---- Weak convergence test ----

def weak_convergence_test(
    terminal_values: np.ndarray,
    reference: float,
) -> float:
    """Compute weak error: |E[f(X_T)] - reference|.

    Args:
        terminal_values: array of terminal values from MC.
        reference: analytical or fine-grid reference value.

    Returns:
        Absolute weak error.
    """
    return abs(float(np.mean(terminal_values)) - reference)
