"""Extended quadrature: Gauss-Jacobi, tanh-sinh, Clenshaw-Curtis.

    from pricebook.numerical import gauss_jacobi, tanh_sinh, clenshaw_curtis

Extends the existing quadrature.py with additional rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class QuadratureResult:
    """Quadrature result."""
    value: float
    error_estimate: float
    n_points: int
    method: str

    def to_dict(self) -> dict:
        return vars(self)


def gauss_jacobi(
    f,
    n: int = 16,
    alpha: float = 0.0,
    beta: float = 0.0,
    a: float = -1.0,
    b: float = 1.0,
) -> QuadratureResult:
    """Gauss-Jacobi quadrature with weight (1-x)^alpha (1+x)^beta on [-1,1].

    alpha=beta=0 reduces to Gauss-Legendre.
    alpha=beta=-0.5 gives Chebyshev of the first kind.
    """
    from numpy.polynomial.legendre import leggauss

    if abs(alpha) < 1e-10 and abs(beta) < 1e-10:
        nodes, weights = leggauss(n)
    else:
        # Use scipy for general Jacobi
        from scipy.special import roots_jacobi
        nodes, weights = roots_jacobi(n, alpha, beta)

    # Map from [-1,1] to [a,b]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    mapped_nodes = mid + half * nodes
    mapped_weights = weights * half

    value = float(np.sum(mapped_weights * np.array([f(x) for x in mapped_nodes])))

    return QuadratureResult(value, 0.0, n, "gauss_jacobi")


def tanh_sinh(
    f,
    a: float = -1.0,
    b: float = 1.0,
    n: int = 64,
    h: float = 0.1,
) -> QuadratureResult:
    """Tanh-sinh (double exponential) quadrature.

    Handles endpoint singularities gracefully.
    Transform: x = tanh(pi/2 sinh(t)), dx = pi/2 cosh(t) / cosh²(pi/2 sinh(t)).
    """
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    total = 0.0

    for k in range(-n, n + 1):
        t = k * h
        sinh_t = math.sinh(t)
        arg = 0.5 * math.pi * sinh_t
        if abs(arg) > 20:
            continue
        x = math.tanh(arg)
        w = 0.5 * math.pi * math.cosh(t) / (math.cosh(arg) ** 2)
        mapped_x = mid + half * x
        total += w * f(mapped_x) * half

    return QuadratureResult(float(total * h), 0.0, 2 * n + 1, "tanh_sinh")


def clenshaw_curtis(
    f,
    a: float = -1.0,
    b: float = 1.0,
    n: int = 32,
) -> QuadratureResult:
    """Clenshaw-Curtis quadrature via DCT.

    Uses Chebyshev nodes (nested: doubling n reuses previous points).
    Exact for polynomials of degree <= n.
    """
    # Chebyshev nodes on [-1,1]
    theta = np.pi * np.arange(n + 1) / n
    nodes = np.cos(theta)

    # Map to [a,b]
    mid = 0.5 * (a + b)
    half = 0.5 * (b - a)
    mapped = mid + half * nodes
    fvals = np.array([f(x) for x in mapped])

    # Compute weights
    weights = np.zeros(n + 1)
    weights[0] = 1.0 / (n * n - 1) if n > 1 else 1.0
    weights[n] = weights[0]
    for j in range(1, n):
        s = 0.0
        for k in range(1, n // 2 + 1):
            b_k = 1.0 if k == n // 2 else 2.0
            s += b_k * math.cos(2 * k * j * math.pi / n) / (4 * k * k - 1)
        weights[j] = 2.0 / n * (1 - s)

    value = float(np.dot(weights, fvals) * half)

    return QuadratureResult(value, 0.0, n + 1, "clenshaw_curtis")
