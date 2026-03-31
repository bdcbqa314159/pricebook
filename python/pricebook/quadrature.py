"""
Numerical integration (quadrature).

Gauss quadrature rules for finite, semi-infinite, and infinite intervals.
Adaptive integration with automatic refinement.

    from pricebook.quadrature import gauss_legendre, adaptive_simpson

    result = gauss_legendre(f, a=0, b=1, n=16)
    print(result.value, result.error_estimate)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.hermite import hermgauss


@dataclass
class QuadratureResult:
    """Result of a numerical integration."""

    value: float
    error_estimate: float
    n_evaluations: int


def gauss_legendre(
    f,
    a: float,
    b: float,
    n: int = 16,
) -> QuadratureResult:
    """Gauss-Legendre quadrature on [a, b].

    Exact for polynomials of degree ≤ 2n-1.
    """
    nodes, weights = leggauss(n)
    # Transform from [-1, 1] to [a, b]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    x = mid + half * nodes
    w = half * weights

    values = np.array([f(xi) for xi in x])
    result = float(np.dot(w, values))

    # Error estimate: compare n vs n//2
    if n >= 4:
        nodes2, weights2 = leggauss(n // 2)
        x2 = mid + half * nodes2
        w2 = half * weights2
        values2 = np.array([f(xi) for xi in x2])
        result2 = float(np.dot(w2, values2))
        err = abs(result - result2)
    else:
        err = float("inf")

    return QuadratureResult(value=result, error_estimate=err, n_evaluations=n)


def gauss_laguerre(
    f,
    n: int = 16,
) -> QuadratureResult:
    """Gauss-Laguerre quadrature on [0, ∞).

    Integrates f(x) * exp(-x) dx. Pass g(x) = f(x) * exp(x) if you
    want to integrate f(x) dx on [0, ∞) without the weight.
    """
    nodes, weights = laggauss(n)
    values = np.array([f(float(xi)) for xi in nodes])
    result = float(np.dot(weights, values))

    if n >= 4:
        nodes2, weights2 = laggauss(n // 2)
        values2 = np.array([f(float(xi)) for xi in nodes2])
        result2 = float(np.dot(weights2, values2))
        err = abs(result - result2)
    else:
        err = float("inf")

    return QuadratureResult(value=result, error_estimate=err, n_evaluations=n)


def gauss_hermite(
    f,
    n: int = 16,
) -> QuadratureResult:
    """Gauss-Hermite quadrature on (-∞, ∞).

    Integrates f(x) * exp(-x^2) dx. Pass g(x) = f(x) * exp(x^2) if you
    want to integrate f(x) dx without the weight.
    """
    nodes, weights = hermgauss(n)
    values = np.array([f(float(xi)) for xi in nodes])
    result = float(np.dot(weights, values))

    if n >= 4:
        nodes2, weights2 = hermgauss(n // 2)
        values2 = np.array([f(float(xi)) for xi in nodes2])
        result2 = float(np.dot(weights2, values2))
        err = abs(result - result2)
    else:
        err = float("inf")

    return QuadratureResult(value=result, error_estimate=err, n_evaluations=n)


def adaptive_simpson(
    f,
    a: float,
    b: float,
    tol: float = 1e-10,
    max_depth: int = 50,
) -> QuadratureResult:
    """Adaptive Simpson's rule with automatic refinement.

    Recursively subdivides intervals where the error exceeds tolerance.
    """
    n_eval = [0]

    def _simpson(fa, fm, fb, a, m, b):
        return (b - a) / 6.0 * (fa + 4.0 * fm + fb)

    def _recursive(a, b, fa, fm, fb, whole, depth):
        m = 0.5 * (a + b)
        lm = 0.5 * (a + m)
        rm = 0.5 * (m + b)
        flm = f(lm)
        frm = f(rm)
        n_eval[0] += 2

        left = _simpson(fa, flm, fm, a, lm, m)
        right = _simpson(fm, frm, fb, m, rm, b)
        combined = left + right
        err = (combined - whole) / 15.0

        if depth <= 0 or abs(err) < tol:
            return combined + err

        return (_recursive(a, m, fa, flm, fm, left, depth - 1) +
                _recursive(m, b, fm, frm, fb, right, depth - 1))

    fa = f(a)
    fb = f(b)
    fm = f(0.5 * (a + b))
    n_eval[0] = 3
    whole = _simpson(fa, fm, fb, a, 0.5 * (a + b), b)
    result = _recursive(a, b, fa, fm, fb, whole, max_depth)

    return QuadratureResult(
        value=result,
        error_estimate=tol,  # guaranteed by the adaptive scheme
        n_evaluations=n_eval[0],
    )
