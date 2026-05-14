"""Root finding: unified interface + bisection.

    from pricebook.numerical import bisection, find_root

Extends existing solvers.py with bisection and a unified dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RootResult:
    """Root finding result."""
    root: float
    iterations: int
    converged: bool
    function_value: float
    method: str

    def to_dict(self) -> dict:
        return vars(self)


def bisection(
    f,
    a: float,
    b: float,
    tol: float = 1e-12,
    maxiter: int = 100,
) -> RootResult:
    """Bisection method: guaranteed convergence for continuous f with f(a)f(b) < 0.

    Simplest bracketing method. Linear convergence (halves interval each step).
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.4e} and f(b)={fb:.4e} must have opposite signs")

    for i in range(maxiter):
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) < tol or (b - a) < tol:
            return RootResult(mid, i + 1, True, float(fm), "bisection")
        if fa * fm < 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    mid = 0.5 * (a + b)
    return RootResult(mid, maxiter, False, float(f(mid)), "bisection")


def find_root(
    f,
    x0: float | None = None,
    bracket: tuple[float, float] | None = None,
    method: str = "brent",
    fprime=None,
    tol: float = 1e-12,
    maxiter: int = 100,
) -> RootResult:
    """Unified root finder dispatching to the appropriate method.

    Methods: bisection, brent, newton, secant, halley, itp.

    Args:
        x0: initial guess (for Newton, secant).
        bracket: (a, b) bracketing interval (for bisection, Brent, ITP).
        fprime: derivative (for Newton, Halley).
    """
    from pricebook.solvers import brentq, newton, secant, halley, itp

    if method == "bisection":
        if bracket is None:
            raise ValueError("bisection requires bracket=(a, b)")
        return bisection(f, bracket[0], bracket[1], tol, maxiter)

    if method == "brent":
        if bracket is None:
            raise ValueError("brent requires bracket=(a, b)")
        root = brentq(f, bracket[0], bracket[1], tol, maxiter)
        return RootResult(root, 0, True, float(f(root)), "brent")

    if method == "itp":
        if bracket is None:
            raise ValueError("itp requires bracket=(a, b)")
        r = itp(f, bracket[0], bracket[1], tol, maxiter)
        return RootResult(r.root, r.iterations, r.converged, r.function_value, "itp")

    if method == "newton":
        if x0 is None or fprime is None:
            raise ValueError("newton requires x0 and fprime")
        r = newton(f, fprime, x0, tol, maxiter)
        return RootResult(r.root, r.iterations, r.converged, r.function_value, "newton")

    if method == "secant":
        if x0 is None:
            raise ValueError("secant requires x0")
        r = secant(f, x0, x0 + 0.01, tol, maxiter)
        return RootResult(r.root, r.iterations, r.converged, r.function_value, "secant")

    if method == "halley":
        if x0 is None or fprime is None:
            raise ValueError("halley requires x0, fprime, and fprime2")
        # Halley needs f'' — approximate via FD if not provided
        def fprime2(x):
            h = 1e-5
            return (fprime(x + h) - fprime(x - h)) / (2 * h)
        r = halley(f, fprime, fprime2, x0, tol, maxiter)
        return RootResult(r.root, r.iterations, r.converged, r.function_value, "halley")

    raise ValueError(f"unknown method: {method!r}")
