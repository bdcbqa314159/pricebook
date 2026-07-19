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
    return float(_brentq(f, lo, hi, xtol=tol, maxiter=max_iter))


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
) -> list[float]:
    """Minimise ``sum(residual(x)^2)`` from `x0` by Levenberg-Marquardt; returns the solution
    vector. The natural shape for calibration (fit parameters to market quotes)."""
    result = _least_squares(
        residual, list(x0), method="lm", xtol=tol, ftol=tol, max_nfev=max_iter
    )
    return [float(v) for v in result.x]
