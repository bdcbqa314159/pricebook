"""
Optimization toolkit for model calibration.

Wraps scipy optimizers with consistent interface, result types,
and registration in the pricebook registry.

    from pricebook.optimization import minimize, OptimizerResult

    result = minimize(objective, x0=[0.5, -0.1, 0.3], method="nelder_mead")
    print(result.x, result.fun, result.converged)

    result = minimize_least_squares(residuals, x0=[...], method="lm")
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize as sp_opt


@dataclass
class OptimizerResult:
    """Result of an optimization run."""

    x: np.ndarray
    fun: float
    iterations: int
    converged: bool
    method: str
    n_evaluations: int = 0


def minimize(
    objective,
    x0: list[float] | np.ndarray,
    method: str = "nelder_mead",
    bounds: list[tuple[float, float]] | None = None,
    tol: float = 1e-8,
    maxiter: int = 2000,
    gradient=None,
    seed: int | None = 42,
) -> OptimizerResult:
    """Minimize a scalar objective function.

    Args:
        objective: f(x) → float, where x is a 1D array.
        x0: initial guess.
        method: "nelder_mead", "bfgs", "l_bfgs_b", "differential_evolution",
                "basin_hopping".
        bounds: [(lo, hi), ...] per parameter. Required for DE and L-BFGS-B.
        tol: convergence tolerance.
        maxiter: maximum iterations.
        gradient: f'(x) → array, for BFGS.
        seed: random seed (for stochastic methods).
    """
    x0 = np.asarray(x0, dtype=float)

    if method == "nelder_mead":
        res = sp_opt.minimize(
            objective, x0, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": tol, "fatol": tol},
        )
        return OptimizerResult(
            x=res.x, fun=float(res.fun), iterations=res.nit,
            converged=res.success, method=method, n_evaluations=res.nfev,
        )

    elif method == "bfgs":
        res = sp_opt.minimize(
            objective, x0, method="BFGS", jac=gradient,
            options={"maxiter": maxiter, "gtol": tol},
        )
        return OptimizerResult(
            x=res.x, fun=float(res.fun), iterations=res.nit,
            converged=res.success, method=method, n_evaluations=res.nfev,
        )

    elif method == "l_bfgs_b":
        res = sp_opt.minimize(
            objective, x0, method="L-BFGS-B", jac=gradient, bounds=bounds,
            options={"maxiter": maxiter, "ftol": tol},
        )
        return OptimizerResult(
            x=res.x, fun=float(res.fun), iterations=res.nit,
            converged=res.success, method=method, n_evaluations=res.nfev,
        )

    elif method == "differential_evolution":
        if bounds is None:
            raise ValueError("bounds required for differential_evolution")
        res = sp_opt.differential_evolution(
            objective, bounds, seed=seed, maxiter=maxiter, tol=tol,
        )
        return OptimizerResult(
            x=res.x, fun=float(res.fun), iterations=res.nit,
            converged=res.success, method=method, n_evaluations=res.nfev,
        )

    elif method == "basin_hopping":
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds} if bounds else {}
        res = sp_opt.basinhopping(
            objective, x0, niter=maxiter, seed=seed,
            minimizer_kwargs=minimizer_kwargs,
        )
        lowest = res.lowest_optimization_result
        return OptimizerResult(
            x=res.x, fun=float(res.fun),
            iterations=res.nit if hasattr(res, 'nit') else maxiter,
            converged=lowest.success if hasattr(lowest, 'success') else True,
            method=method,
            n_evaluations=lowest.nfev if hasattr(lowest, 'nfev') else 0,
        )

    else:
        raise ValueError(f"Unknown method '{method}'. Available: nelder_mead, bfgs, "
                         "l_bfgs_b, differential_evolution, basin_hopping")


def minimize_least_squares(
    residuals,
    x0: list[float] | np.ndarray,
    method: str = "lm",
    bounds: tuple | None = None,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> OptimizerResult:
    """Minimize sum of squared residuals (nonlinear least squares).

    Args:
        residuals: f(x) → array of residuals.
        x0: initial guess.
        method: "lm" (Levenberg-Marquardt) or "trf" (Trust Region Reflective).
        bounds: (lower, upper) arrays. Required for "trf".
    """
    x0 = np.asarray(x0, dtype=float)

    if bounds is None:
        bounds_arg = (-np.inf, np.inf)
    else:
        bounds_arg = bounds

    res = sp_opt.least_squares(
        residuals, x0, method=method, bounds=bounds_arg,
        ftol=tol, xtol=tol, max_nfev=maxiter,
    )

    return OptimizerResult(
        x=res.x, fun=float(res.cost), iterations=0,
        converged=res.success, method=method, n_evaluations=res.nfev,
    )
