"""Spectral methods — Chebyshev collocation and polynomial expansion.

High-accuracy PDE solving and function approximation using polynomial
spectral methods. Exponential convergence for smooth problems.

    from pricebook.numerical._spectral import (
        chebyshev_nodes, chebyshev_diff_matrix, chebyshev_interpolate,
        spectral_solve_bvp, SpectralResult,
    )

References:
    Trefethen (2000). Spectral Methods in MATLAB. SIAM.
    Boyd (2001). Chebyshev and Fourier Spectral Methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SpectralResult:
    """Result of spectral method computation."""
    nodes: np.ndarray            # collocation points
    values: np.ndarray           # solution values at nodes
    coefficients: np.ndarray     # spectral coefficients
    n_points: int
    residual: float

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the spectral approximation at arbitrary points."""
        return chebyshev_evaluate(self.coefficients, x, self.nodes[0], self.nodes[-1])

    def to_dict(self) -> dict:
        return {"n_points": self.n_points, "residual": self.residual}


# ═══════════════════════════════════════════════════════════════
# Chebyshev fundamentals
# ═══════════════════════════════════════════════════════════════


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [a, b].

    x_j = (a+b)/2 + (b-a)/2 × cos(πj/N), j = 0, ..., N.

    Clustered near endpoints → excellent for Runge phenomenon avoidance.
    """
    j = np.arange(n + 1)
    nodes = np.cos(np.pi * j / n)
    return 0.5 * (a + b) + 0.5 * (b - a) * nodes


def chebyshev_diff_matrix(n: int) -> np.ndarray:
    """Chebyshev differentiation matrix D on [-1, 1].

    D[i, j] = derivative of j-th Lagrange interpolant at node i.
    Size: (N+1) × (N+1).
    """
    x = chebyshev_nodes(n)
    N = n
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c *= (-1.0) ** np.arange(N + 1)

    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D


def chebyshev_coefficients(values: np.ndarray) -> np.ndarray:
    """Compute Chebyshev coefficients from values at Chebyshev nodes.

    Uses the discrete cosine transform (DCT).
    """
    n = len(values) - 1
    if n == 0:
        return values.copy()

    # Compute via DCT-I
    V = values.copy()
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        s = 0.0
        for j in range(n + 1):
            w = 1.0
            if j == 0 or j == n:
                w = 0.5
            s += w * V[j] * math.cos(math.pi * k * j / n)
        coeffs[k] = 2.0 * s / n

    coeffs[0] /= 2.0
    coeffs[n] /= 2.0
    return coeffs


def chebyshev_evaluate(coeffs: np.ndarray, x: float | np.ndarray,
                        a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Evaluate Chebyshev expansion at arbitrary points.

    Uses Clenshaw recurrence (numerically stable).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    # Map x from [a, b] to [-1, 1]
    t = 2.0 * (x - a) / (b - a) - 1.0
    t = np.clip(t, -1, 1)

    n = len(coeffs) - 1
    if n == 0:
        return np.full_like(x, coeffs[0])

    # Clenshaw recurrence
    b_k2 = np.zeros_like(x)
    b_k1 = np.zeros_like(x)
    for k in range(n, 0, -1):
        b_k = coeffs[k] + 2.0 * t * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k

    return coeffs[0] + t * b_k1 - b_k2


def chebyshev_interpolate(
    f: callable,
    n: int,
    a: float = -1.0,
    b: float = 1.0,
) -> SpectralResult:
    """Interpolate a function using Chebyshev polynomials.

    Evaluates f at N+1 Chebyshev nodes and computes spectral coefficients.

    Args:
        f: function to interpolate, f(x) → float.
        n: polynomial degree (N+1 points).
        a, b: interval [a, b].
    """
    nodes = chebyshev_nodes(n, a, b)
    values = np.array([f(x) for x in nodes])
    coeffs = chebyshev_coefficients(values)

    # Residual: how well the last coefficients are decaying
    residual = float(np.abs(coeffs[-3:]).max()) if n > 3 else 0.0

    return SpectralResult(nodes, values, coeffs, n + 1, residual)


# ═══════════════════════════════════════════════════════════════
# Spectral PDE / BVP solver
# ═══════════════════════════════════════════════════════════════


def spectral_solve_bvp(
    L_operator: callable,
    rhs: callable,
    bc_left: float,
    bc_right: float,
    n: int = 32,
    a: float = 0.0,
    b: float = 1.0,
) -> SpectralResult:
    """Solve a linear BVP: L[u](x) = f(x), u(a) = bc_left, u(b) = bc_right.

    The operator L is discretised on Chebyshev nodes using the
    differentiation matrix.

    Args:
        L_operator: callable(D, D2, x_interior) → (N-1, N-1) operator matrix.
            D = first derivative matrix (interior), D2 = second derivative.
            Return the discretised operator on interior nodes.
        rhs: callable(x) → f(x), right-hand side function.
        bc_left, bc_right: Dirichlet boundary conditions.
        n: number of collocation points.
        a, b: domain [a, b].
    """
    # Chebyshev on [a, b]
    nodes = chebyshev_nodes(n, a, b)
    D_ref = chebyshev_diff_matrix(n)
    # Scale derivative for [a, b] domain
    scale = 2.0 / (b - a)
    D = scale * D_ref
    D2 = D @ D

    # Interior nodes (exclude boundaries: first and last)
    interior = slice(1, n)
    x_int = nodes[interior]

    # Build operator matrix
    L = L_operator(D[interior, :][:, interior], D2[interior, :][:, interior], x_int)

    # RHS: f(x) minus boundary contributions
    f = np.array([rhs(x) for x in x_int])

    # Boundary corrections
    f -= bc_right * D2[interior, 0]  # left BC (Chebyshev node 0 = right endpoint)
    f -= bc_left * D2[interior, -1]  # right BC

    # Solve
    try:
        u_int = np.linalg.solve(L, f)
    except np.linalg.LinAlgError:
        u_int = np.linalg.lstsq(L, f, rcond=None)[0]

    # Assemble full solution
    u = np.zeros(n + 1)
    u[0] = bc_right   # Chebyshev: node 0 = cos(0) = 1 → right endpoint
    u[-1] = bc_left    # node N = cos(π) = -1 → left endpoint
    u[interior] = u_int

    coeffs = chebyshev_coefficients(u)
    residual = float(np.max(np.abs(L @ u_int - f)))

    return SpectralResult(nodes, u, coeffs, n + 1, residual)


# ═══════════════════════════════════════════════════════════════
# Gauss quadrature (Legendre)
# ═══════════════════════════════════════════════════════════════


def legendre_nodes_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature nodes and weights on [-1, 1].

    Exact for polynomials of degree ≤ 2n-1.
    """
    from numpy.polynomial.legendre import leggauss
    return leggauss(n)


def spectral_integrate(
    f: callable,
    a: float,
    b: float,
    n: int = 20,
) -> float:
    """Integrate f(x) from a to b using Gauss-Legendre quadrature.

    Exact for polynomials of degree ≤ 2n-1.
    """
    nodes, weights = legendre_nodes_weights(n)
    # Map from [-1, 1] to [a, b]
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.dot(weights, [f(xi) for xi in x]))
