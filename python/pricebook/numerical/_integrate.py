"""Numerical integration — unified framework with method selection.

Adaptive and fixed-order quadrature for 1D, 2D, and semi-infinite domains.
Handles regular, singular, and oscillatory integrands.

    from pricebook.numerical._integrate import (
        integrate, IntegrationMethod, IntegrationResult,
        integrate_2d, integrate_semi_infinite,
    )

    # Simple: adaptive with error control
    result = integrate(f, 0, 1)

    # Method choice
    result = integrate(f, 0, 1, method=IntegrationMethod.GAUSS_LEGENDRE, n=20)

    # Singular integrand
    result = integrate(f, 0, 1, method=IntegrationMethod.TANH_SINH)

References:
    Davis & Rabinowitz (1984). Methods of Numerical Integration.
    Trefethen (2008). Is Gauss Quadrature Better than Clenshaw-Curtis?
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class IntegrationMethod(Enum):
    """Numerical integration method."""
    ADAPTIVE = "adaptive"            # scipy.integrate.quad (Gauss-Kronrod, adaptive)
    GAUSS_LEGENDRE = "gauss_legendre"  # fixed-order Gauss-Legendre
    GAUSS_LAGUERRE = "gauss_laguerre"  # semi-infinite [0, ∞) with exp(-x) weight
    GAUSS_HERMITE = "gauss_hermite"    # infinite (-∞, ∞) with exp(-x²) weight
    TANH_SINH = "tanh_sinh"          # double exponential (endpoint singularities)
    CLENSHAW_CURTIS = "clenshaw_curtis"  # Chebyshev nodes (nested)
    SIMPSON = "simpson"              # composite Simpson's 1/3 rule
    TRAPEZOID = "trapezoid"          # composite trapezoidal rule
    ROMBERG = "romberg"              # Richardson-extrapolated trapezoid


@dataclass
class IntegrationResult:
    """Result of numerical integration."""
    value: float
    error_estimate: float
    n_evaluations: int
    method: str
    converged: bool

    def to_dict(self) -> dict:
        return vars(self)


def integrate(
    f: callable,
    a: float,
    b: float,
    method: IntegrationMethod = IntegrationMethod.ADAPTIVE,
    n: int = 50,
    tol: float = 1e-10,
    args: tuple = (),
) -> IntegrationResult:
    """Integrate f(x) from a to b.

    Args:
        f: integrand f(x) → float.
        a, b: integration limits. Use np.inf for semi-infinite.
        method: quadrature method.
        n: number of points (for fixed-order methods).
        tol: tolerance (for adaptive methods).
        args: extra arguments passed to f.
    """
    func = (lambda x: f(x, *args)) if args else f

    if method == IntegrationMethod.ADAPTIVE:
        return _adaptive(func, a, b, tol)
    elif method == IntegrationMethod.GAUSS_LEGENDRE:
        return _gauss_legendre(func, a, b, n)
    elif method == IntegrationMethod.GAUSS_LAGUERRE:
        return _gauss_laguerre(func, n)
    elif method == IntegrationMethod.GAUSS_HERMITE:
        return _gauss_hermite(func, n)
    elif method == IntegrationMethod.TANH_SINH:
        return _tanh_sinh(func, a, b, n)
    elif method == IntegrationMethod.CLENSHAW_CURTIS:
        return _clenshaw_curtis(func, a, b, n)
    elif method == IntegrationMethod.SIMPSON:
        return _simpson(func, a, b, n)
    elif method == IntegrationMethod.TRAPEZOID:
        return _trapezoid(func, a, b, n)
    elif method == IntegrationMethod.ROMBERG:
        return _romberg(func, a, b, tol)
    else:
        raise ValueError(f"Unknown method: {method}")


def integrate_2d(
    f: callable,
    x_range: tuple[float, float],
    y_range: tuple[float, float] | callable,
    method: IntegrationMethod = IntegrationMethod.ADAPTIVE,
    tol: float = 1e-8,
) -> IntegrationResult:
    """Double integral ∫∫ f(x, y) dy dx.

    Args:
        f: integrand f(x, y) → float.
        x_range: (x_min, x_max).
        y_range: (y_min, y_max) or callable(x) → (y_min, y_max).
    """
    from scipy.integrate import dblquad

    if callable(y_range):
        y_lo = lambda x: y_range(x)[0]
        y_hi = lambda x: y_range(x)[1]
    else:
        y_lo = lambda x: y_range[0]
        y_hi = lambda x: y_range[1]

    # Fix T1.3: scipy.integrate.dblquad expects func(y, x) — y is the INNER
    # variable, x is the OUTER. Our docstring contract is f(x, y), so wrap
    # to swap argument order before passing to scipy. Pre-fix, scipy was
    # calling the user's f as f(y, x), inverting any non-symmetric integrand
    # (e.g. integrate_2d(lambda x, y: x, (0,3), (0,1)) returned 1.5 instead
    # of the correct 4.5).
    def _f_xy(y, x):
        return f(x, y)

    value, error = dblquad(_f_xy, x_range[0], x_range[1], y_lo, y_hi,
                            epsabs=tol, epsrel=tol)

    return IntegrationResult(float(value), float(error), 0, "dblquad", True)


def integrate_semi_infinite(
    f: callable,
    a: float = 0.0,
    method: IntegrationMethod = IntegrationMethod.GAUSS_LAGUERRE,
    n: int = 30,
) -> IntegrationResult:
    """Integrate f(x) from a to ∞.

    For integrands that decay exponentially, Gauss-Laguerre is optimal.
    For general integrands, use adaptive with np.inf.
    """
    if method == IntegrationMethod.GAUSS_LAGUERRE:
        # Shift: ∫_a^∞ f(x)dx = ∫_0^∞ f(x+a) dx ≈ ∫_0^∞ f(x+a) e^x × e^{-x} dx
        # Gauss-Laguerre integrates g(x)e^{-x} directly
        shifted = lambda x: f(x + a) * math.exp(x)
        return _gauss_laguerre(shifted, n)
    else:
        return _adaptive(f, a, np.inf)


def integrate_complex_contour(
    f: callable,
    t_range: tuple[float, float],
    contour: callable,
    contour_derivative: callable,
    n: int = 100,
) -> complex:
    """Complex contour integral ∮ f(z) dz along a parameterised contour.

    ∫ f(z(t)) z'(t) dt from t_range[0] to t_range[1].

    Args:
        f: integrand f(z) → complex.
        t_range: parameter range.
        contour: z(t) → complex.
        contour_derivative: z'(t) → complex.
        n: quadrature points.
    """
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n)
    t_mid = 0.5 * (t_range[0] + t_range[1])
    t_half = 0.5 * (t_range[1] - t_range[0])
    t_mapped = t_mid + t_half * nodes

    total = 0j
    for t, w in zip(t_mapped, weights):
        z = contour(t)
        dz = contour_derivative(t)
        total += w * f(z) * dz * t_half

    return total


# ═══════════════════════════════════════════════════════════════
# Internal implementations
# ═══════════════════════════════════════════════════════════════


def _adaptive(f, a, b, tol=1e-10):
    from scipy.integrate import quad
    value, error = quad(f, a, b, epsabs=tol, epsrel=tol, limit=200)
    return IntegrationResult(float(value), float(error), 0, "adaptive", True)


def _gauss_legendre(f, a, b, n):
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(n)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * nodes
    value = float(half * np.dot(weights, [f(xi) for xi in x]))
    return IntegrationResult(value, 0.0, n, "gauss_legendre", True)


def _gauss_laguerre(f, n):
    from numpy.polynomial.laguerre import laggauss
    nodes, weights = laggauss(n)
    value = float(np.dot(weights, [f(x) for x in nodes]))
    return IntegrationResult(value, 0.0, n, "gauss_laguerre", True)


def _gauss_hermite(f, n):
    from numpy.polynomial.hermite import hermgauss
    nodes, weights = hermgauss(n)
    value = float(np.dot(weights, [f(x) for x in nodes]))
    return IntegrationResult(value, 0.0, n, "gauss_hermite", True)


def _tanh_sinh(f, a, b, n):
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    h = 0.1
    total = 0.0
    for k in range(-n, n + 1):
        t = k * h
        sinh_t = math.sinh(t)
        arg = 0.5 * math.pi * sinh_t
        if abs(arg) > 20:
            continue
        x = math.tanh(arg)
        w = 0.5 * math.pi * math.cosh(t) / (math.cosh(arg) ** 2)
        total += w * f(mid + half * x) * half
    return IntegrationResult(float(total * h), 0.0, 2 * n + 1, "tanh_sinh", True)


def _clenshaw_curtis(f, a, b, n):
    """Clenshaw-Curtis quadrature on n+1 nodes at cos(jπ/n).

    Fix T3.7: pre-fix used the n-EVEN weight formula for ALL n.  Two
    differences for n odd:
      (a) the upper sum limit is (n−1)/2 with NO halved-boundary term
          (the n/2 term doesn't exist when n is odd), and
      (b) the endpoint weights are 1/n² (not 1/(n²−1)).
    Pre-fix, integrating constants for odd n gave the wrong answer (e.g.
    ∫₀¹ 1 dx ≈ 0.9 instead of 1 for n=3).
    """
    theta = np.pi * np.arange(n + 1) / n
    nodes = np.cos(theta)
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    mapped = mid + half * nodes
    fvals = np.array([f(x) for x in mapped])
    weights = np.zeros(n + 1)
    if n <= 1:
        weights[0] = 1.0
        if n == 1:
            weights[1] = 1.0
    elif n % 2 == 0:
        weights[0] = 1.0 / (n * n - 1)
        weights[n] = weights[0]
        for j in range(1, n):
            s = 0.0
            for k in range(1, n // 2 + 1):
                b_k = 1.0 if k == n // 2 else 2.0
                s += b_k * math.cos(2 * k * j * math.pi / n) / (4 * k * k - 1)
            weights[j] = 2.0 / n * (1 - s)
    else:
        # n odd
        weights[0] = 1.0 / (n * n)
        weights[n] = weights[0]
        for j in range(1, n):
            s = 0.0
            for k in range(1, (n - 1) // 2 + 1):
                s += 2.0 * math.cos(2 * k * j * math.pi / n) / (4 * k * k - 1)
            weights[j] = 2.0 / n * (1 - s)
    value = float(np.dot(weights, fvals) * half)
    return IntegrationResult(value, 0.0, n + 1, "clenshaw_curtis", True)


def _simpson(f, a, b, n):
    # Fix T4-INT1: pre-fix `n=0` caused ``h = (b - a) / 0`` →
    # ZeroDivisionError. Validate that we have at least one interval.
    if n < 1:
        raise ValueError(
            f"_simpson: n must be >= 1 (got {n}); need at least one "
            f"subinterval."
        )
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    fvals = np.array([f(xi) for xi in x])
    value = h / 3 * (fvals[0] + fvals[-1] + 4 * fvals[1::2].sum() + 2 * fvals[2:-1:2].sum())
    return IntegrationResult(float(value), 0.0, n + 1, "simpson", True)


def _trapezoid(f, a, b, n):
    # Fix T4-INT1: pre-fix `n=0` caused ``h = (b - a) / 0`` →
    # ZeroDivisionError.
    if n < 1:
        raise ValueError(
            f"_trapezoid: n must be >= 1 (got {n}); need at least one "
            f"subinterval."
        )
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    fvals = np.array([f(xi) for xi in x])
    value = h * (0.5 * fvals[0] + fvals[1:-1].sum() + 0.5 * fvals[-1])
    return IntegrationResult(float(value), 0.0, n + 1, "trapezoid", True)


def _romberg(f, a, b, tol=1e-10, max_levels: int = 20):
    """Romberg integration via Richardson extrapolation of trapezoidal estimates.

    Fix T2.6: pre-fix this wrapped ``scipy.integrate.romberg``, which was
    removed in SciPy 1.15.  Reimplemented natively so the function still
    works without a SciPy version pin.

    The Romberg table is built as:
        R[k, 0] = trapezoidal rule with 2^k intervals
        R[k, j] = (4^j · R[k, j-1] − R[k-1, j-1]) / (4^j − 1)
    Converges when |R[k, k] − R[k-1, k-1]| < tol.
    """
    f0 = f(a)
    fN = f(b)
    R_prev = [(b - a) * 0.5 * (f0 + fN)]
    n_evals = 2

    for k in range(1, max_levels + 1):
        # Trapezoid refinement: add the new midpoint values.
        n = 1 << k
        h = (b - a) / n
        s = 0.0
        for i in range(1, n, 2):
            s += f(a + i * h)
            n_evals += 1
        R_k0 = 0.5 * R_prev[0] + h * s

        R_cur = [R_k0]
        for j in range(1, k + 1):
            R_cur.append(
                (4**j * R_cur[j - 1] - R_prev[j - 1]) / (4**j - 1)
            )

        if abs(R_cur[k] - R_prev[k - 1]) < tol:
            return IntegrationResult(
                float(R_cur[k]), float(abs(R_cur[k] - R_prev[k - 1])),
                n_evals, "romberg", True,
            )
        R_prev = R_cur

    return IntegrationResult(
        float(R_prev[-1]), float(tol), n_evals, "romberg", False,
    )


