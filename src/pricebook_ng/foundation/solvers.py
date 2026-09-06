"""Root-finding and least-squares — thin scipy adapters (L0, S17).

`scipy.optimize` behind our own API + provenance: Brent (bracketed root), Newton (with
derivative), secant (without), and Levenberg-Marquardt least-squares for calibration
residuals. This **replaces** the hand-rolled `bisect_root`/`nelder_mead` — no duplicates,
one swap point for the C++ port. Never call scipy from engines/models — call these.

Provenance:
  quarry: python/pricebook/core/ (solvers)
  source: scipy.optimize (brentq, newton, least_squares method='lm')
  oracle: brent/newton/secant find sqrt(2); LM recovers a linear fit's coefficients
  slice:  l0-numerics (Topic 0 gate S17)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from scipy.optimize import brentq as _brentq
from scipy.optimize import least_squares as _least_squares
from scipy.optimize import newton as _newton


def brent(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> float:
    """Root of `f` bracketed in ``[lo, hi]`` (Brent). `f(lo)`, `f(hi)` must differ in sign."""
    # brentq returns a bare float without full_output; the stub types it as a union, so cast.
    return float(cast(float, _brentq(f, lo, hi, xtol=tol, maxiter=max_iter)))


def newton(
    f: Callable[[float], float],
    x0: float,
    fprime: Callable[[float], float],
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Root of `f` from `x0` using its derivative `fprime` (Newton)."""
    return float(_newton(f, x0, fprime=fprime, tol=tol, maxiter=max_iter))


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Root of `f` from two starting points `x0`, `x1` (secant — no derivative needed)."""
    return float(_newton(f, x0, x1=x1, tol=tol, maxiter=max_iter))


def least_squares(
    residual: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    tol: float = 1e-12,
    max_iter: int = 1000,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> list[float]:
    """Minimise ``sum(residual(x)^2)`` from `x0`; returns the solution vector. Unbounded uses
    Levenberg-Marquardt; passing `bounds=(lower, upper)` constrains the solution to a box
    (Trust Region Reflective — LM cannot take bounds). The natural shape for calibration."""
    return root_nd(residual, x0, tol, max_iter, bounds)[0]


def root_nd(
    residual: Callable[[Sequence[float]], Sequence[float]],
    x0: Sequence[float],
    tol: float = 1e-12,
    max_iter: int = 1000,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
) -> tuple[list[float], list[list[float]], bool]:
    """Solve the N-D system ``F(x) = 0`` from `x0` by least-squares. Returns
    ``(solution, jacobian, converged)`` where `jacobian` is ``∂residualᵢ/∂xⱼ`` at the solution
    (rows = residuals, cols = unknowns). Unbounded uses Levenberg-Marquardt; `bounds=(lower,
    upper)` switches to Trust Region Reflective and keeps the solution in the box. Non-convergence
    — including a solver error or a residual that blows up — is returned as ``converged=False``
    (failure is a value), never raised. The single N-D solver for calibration (§7bb)."""
    try:
        if bounds is None:
            result = _least_squares(
                residual, list(x0), method="lm", xtol=tol, ftol=tol, max_nfev=max_iter
            )
        else:
            result = _least_squares(
                residual, list(x0), method="trf",
                bounds=(list(bounds[0]), list(bounds[1])), xtol=tol, ftol=tol, max_nfev=max_iter,
            )
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return list(map(float, x0)), [], False
    x = [float(v) for v in result.x]
    jac = [[float(v) for v in row] for row in np.atleast_2d(np.asarray(result.jac, dtype=float))]
    return x, jac, bool(result.success)
