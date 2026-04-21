"""
Numerical root-finding solvers.

    from pricebook.solvers import newton, brentq, SolverResult

    result = newton(f, fprime, x0=0.5)
    print(result.root, result.iterations, result.converged)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SolverResult:
    """Result of a root-finding solver."""

    root: float
    iterations: int
    converged: bool
    function_value: float


def newton(
    f,
    fprime,
    x0: float,
    tol: float = 1e-12,
    maxiter: int = 50,
) -> SolverResult:
    """Newton-Raphson root finder.

    Args:
        f: function to find root of.
        fprime: derivative of f.
        x0: initial guess.
        tol: convergence tolerance on |f(x)|.
        maxiter: maximum iterations.
    """
    x = x0
    for i in range(maxiter):
        fx = f(x)
        if abs(fx) < tol:
            return SolverResult(root=x, iterations=i, converged=True, function_value=fx)
        dfx = fprime(x)
        if abs(dfx) < 1e-15:
            break
        x = x - fx / dfx
    fx = f(x)
    return SolverResult(root=x, iterations=maxiter, converged=abs(fx) < tol, function_value=fx)


def secant(
    f,
    x0: float,
    x1: float,
    tol: float = 1e-12,
    maxiter: int = 50,
) -> SolverResult:
    """Secant method: Newton-like without explicit derivative.

    Uses finite difference (f(x1)-f(x0))/(x1-x0) as approximate derivative.
    """
    f0 = f(x0)
    f1 = f(x1)
    for i in range(maxiter):
        if abs(f1) < tol:
            return SolverResult(root=x1, iterations=i, converged=True, function_value=f1)
        denom = f1 - f0
        if abs(denom) < 1e-15:
            break
        x2 = x1 - f1 * (x1 - x0) / denom
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
    return SolverResult(root=x1, iterations=maxiter, converged=abs(f1) < tol, function_value=f1)


def halley(
    f,
    fprime,
    fprime2,
    x0: float,
    tol: float = 1e-12,
    maxiter: int = 50,
) -> SolverResult:
    """Halley's method: cubic convergence using f, f', f''.

    x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')
    """
    x = x0
    for i in range(maxiter):
        fx = f(x)
        if abs(fx) < tol:
            return SolverResult(root=x, iterations=i, converged=True, function_value=fx)
        dfx = fprime(x)
        d2fx = fprime2(x)
        denom = 2.0 * dfx * dfx - fx * d2fx
        if abs(denom) < 1e-15:
            break
        x = x - 2.0 * fx * dfx / denom
    fx = f(x)
    return SolverResult(root=x, iterations=maxiter, converged=abs(fx) < tol, function_value=fx)


def itp(
    f,
    a: float,
    b: float,
    tol: float = 1e-12,
    maxiter: int = 100,
    kappa1: float = 0.1,
    kappa2: float = 2.0,
) -> SolverResult:
    """ITP method: Interpolate-Truncate-Project.

    Optimal worst-case bracketing method. Always beats bisection.
    Superlinear average-case convergence.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6e} and f(b)={fb:.6e} must have opposite signs")

    # Ensure fa < 0 < fb
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    ratio = (b - a) / (2 * tol)
    n_half = math.ceil(math.log2(max(ratio, 1.0)))
    n_max = n_half + 1

    for i in range(max(maxiter, n_max)):
        mid = 0.5 * (a + b)
        r = tol * 2.0 ** (n_max - i) - 0.5 * (b - a)

        # Interpolation (regula falsi)
        x_f = (fb * a - fa * b) / (fb - fa)

        # Truncation
        sigma = math.copysign(1.0, mid - x_f)
        delta = kappa1 * (b - a) ** kappa2
        if delta <= abs(mid - x_f):
            x_t = x_f + sigma * delta
        else:
            x_t = mid

        # Projection
        if abs(x_t - mid) <= r:
            x_itp = x_t
        else:
            x_itp = mid - sigma * r

        f_itp = f(x_itp)
        if abs(f_itp) < tol:
            return SolverResult(root=x_itp, iterations=i, converged=True, function_value=f_itp)

        if f_itp < 0:
            a, fa = x_itp, f_itp
        else:
            b, fb = x_itp, f_itp

        if abs(b - a) < 2 * tol:
            root = 0.5 * (a + b)
            froot = f(root)
            return SolverResult(root=root, iterations=i, converged=True, function_value=froot)

    root = 0.5 * (a + b)
    froot = f(root)
    return SolverResult(root=root, iterations=maxiter, converged=abs(froot) < tol, function_value=froot)


def brentq(f, a: float, b: float, tol: float = 1e-12, maxiter: int = 100) -> float:
    """Brent's method for root finding on [a, b].

    f(a) and f(b) must have opposite signs.
    Returns the root directly (float) for backward compatibility.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError(f"f(a)={fa:.6e} and f(b)={fb:.6e} must have opposite signs")

    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b

    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    mflag = True

    for _ in range(maxiter):
        if abs(fb) < tol:
            return b
        if abs(b - a) < tol:
            return b

        if fa != fc and fb != fc:
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
                 + b * fa * fc / ((fb - fa) * (fb - fc))
                 + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = not mflag and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = not mflag and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d = c
        c, fc = b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Convergence check: warn if residual is large
    if abs(fb) > tol * 1000:
        import warnings
        warnings.warn(
            f"brentq: solver may not have converged. "
            f"|f(root)|={abs(fb):.2e}, tol={tol:.2e}, root={b:.6e}",
            RuntimeWarning,
            stacklevel=2,
        )
    return b
