"""Approximation theory: Chebyshev interpolation, Padé, error bounds.

Phase M8 slices 197-198 consolidated.

* :func:`chebyshev_interpolate` — near-optimal polynomial interpolation.
* :func:`chebyshev_evaluate` — Clenshaw evaluation at arbitrary points.
* :func:`pade_approximant` — rational approximation from Taylor coefficients.
* :func:`richardson_table` — full Richardson extrapolation table.
* :func:`bspline_basis` — B-spline basis functions for curve construction.

References:
    Trefethen, *Approximation Theory and Approximation Practice*, SIAM, 2013.
    Baker & Graves-Morris, *Padé Approximants*, Cambridge, 1996.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---- Chebyshev interpolation ----

@dataclass
class ChebyshevInterpolant:
    """Chebyshev polynomial interpolant on [a, b]."""
    coefficients: np.ndarray  # Chebyshev coefficients c_k
    a: float
    b: float
    degree: int

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the interpolant at x via Clenshaw algorithm."""
        x = np.asarray(x, dtype=float)
        # Map x from [a, b] to [-1, 1]
        xi = (2 * x - self.a - self.b) / (self.b - self.a)
        return _clenshaw(self.coefficients, xi)

    def max_coeff_magnitude(self) -> float:
        """Magnitude of the last few coefficients (convergence diagnostic)."""
        return float(np.max(np.abs(self.coefficients[-3:])))


def chebyshev_interpolate(
    f: Callable[[float | np.ndarray], float | np.ndarray],
    a: float,
    b: float,
    n: int = 20,
) -> ChebyshevInterpolant:
    """Interpolate f on [a, b] using n+1 Chebyshev-Lobatto points.

    Near-minimax: the interpolation error is within a factor
    O(log n) of the best polynomial approximation.

    Args:
        f: function to interpolate.
        a, b: interval.
        n: polynomial degree.

    Returns:
        :class:`ChebyshevInterpolant`.

    Reference:
        Trefethen, ATAP, Ch. 4-5.
    """
    # Chebyshev-Lobatto points on [-1, 1]
    k = np.arange(n + 1)
    xi = np.cos(k * math.pi / n)

    # Map to [a, b]
    x = 0.5 * (a + b) + 0.5 * (b - a) * xi
    fx = np.asarray(f(x), dtype=float)

    # Chebyshev coefficients via DCT
    coeffs = np.zeros(n + 1)
    for j in range(n + 1):
        s = 0.0
        for i in range(n + 1):
            w = 1.0 if (i == 0 or i == n) else 2.0
            s += w * fx[i] * math.cos(j * i * math.pi / n)
        coeffs[j] = s / n
    coeffs[0] /= 2
    coeffs[n] /= 2

    return ChebyshevInterpolant(coeffs, a, b, n)


def _clenshaw(coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Clenshaw algorithm for evaluating a Chebyshev series.

    Computes Σ c_k T_k(xi) using the recurrence T_{k+1} = 2xi T_k - T_{k-1}.
    """
    n = len(coeffs) - 1
    if n == 0:
        return np.full_like(xi, coeffs[0])

    b_k1 = np.zeros_like(xi)
    b_k2 = np.zeros_like(xi)

    for k in range(n, 0, -1):
        b_k = coeffs[k] + 2 * xi * b_k1 - b_k2
        b_k2 = b_k1
        b_k1 = b_k

    return coeffs[0] + xi * b_k1 - b_k2


# ---- Padé approximant ----

@dataclass
class PadeApproximant:
    """Padé [L/M] rational approximation: P_L(x) / Q_M(x)."""
    numerator: np.ndarray   # coefficients of P (degree L)
    denominator: np.ndarray # coefficients of Q (degree M), Q[0] = 1
    L: int
    M: int

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the Padé approximant at x."""
        x = np.asarray(x, dtype=float)
        num = np.polyval(self.numerator[::-1], x)
        den = np.polyval(self.denominator[::-1], x)
        return num / den


def pade_approximant(
    taylor_coeffs: list[float] | np.ndarray,
    L: int,
    M: int,
) -> PadeApproximant:
    """Construct a Padé [L/M] approximant from Taylor series coefficients.

    Given f(x) = Σ c_k x^k, find polynomials P_L and Q_M such that
    P_L(x) / Q_M(x) matches the Taylor series to order L + M.

    Args:
        taylor_coeffs: Taylor coefficients c_0, c_1, ..., c_{L+M}.
        L: numerator degree.
        M: denominator degree.

    Reference:
        Baker & Graves-Morris, Padé Approximants, Ch. 1.
    """
    c = np.asarray(taylor_coeffs, dtype=float)
    n = L + M

    if len(c) < n + 1:
        c = np.pad(c, (0, n + 1 - len(c)))

    # Solve for denominator coefficients q_1, ..., q_M
    # from the system: Σ_{j=0}^{M} q_j c_{L+1-j+k} = 0 for k=0,...,M-1
    # with q_0 = 1.
    if M > 0:
        A = np.zeros((M, M))
        b_vec = np.zeros(M)
        for i in range(M):
            for j in range(M):
                idx = L + 1 + i - j - 1
                A[i, j] = c[idx] if 0 <= idx <= n else 0.0
            b_vec[i] = -c[L + 1 + i] if L + 1 + i <= n else 0.0

        try:
            q = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            q = np.zeros(M)

        denom = np.zeros(M + 1)
        denom[0] = 1.0
        denom[1:] = q
    else:
        denom = np.array([1.0])

    # Numerator: p_k = Σ_{j=0}^{min(k,M)} q_j c_{k-j}
    numer = np.zeros(L + 1)
    for k in range(L + 1):
        for j in range(min(k, M) + 1):
            numer[k] += denom[j] * c[k - j]

    return PadeApproximant(numer, denom, L, M)


# ---- Richardson extrapolation table ----

@dataclass
class RichardsonTable:
    """Full Richardson extrapolation table."""
    table: np.ndarray  # (n, n) triangular table
    best_estimate: float
    estimates: list[float]  # diagonal entries


def richardson_table(
    values: list[float],
    order: int = 2,
) -> RichardsonTable:
    """Build a full Richardson extrapolation table.

    Given estimates f(h), f(h/2), f(h/4), ..., build the
    triangular table where each column cancels one more order
    of the leading error term.

    Args:
        values: estimates at h, h/2, h/4, ... (must have at least 2).
        order: base convergence order p.

    Returns:
        :class:`RichardsonTable` with the full table and best estimate.
    """
    n = len(values)
    T = np.zeros((n, n))
    T[:, 0] = values

    for j in range(1, n):
        r = 2 ** (order * j)
        for i in range(j, n):
            T[i, j] = (r * T[i, j - 1] - T[i - 1, j - 1]) / (r - 1)

    diag = [float(T[i, i]) for i in range(n)]

    return RichardsonTable(T, float(T[n - 1, n - 1]), diag)


# ---- B-spline basis ----

def bspline_basis(
    x: float,
    knots: np.ndarray | list[float],
    degree: int,
    i: int,
) -> float:
    """Evaluate the i-th B-spline basis function of given degree at x.

    Uses the Cox-de Boor recursion:
        B_{i,0}(x) = 1 if knots[i] ≤ x < knots[i+1], else 0
        B_{i,k}(x) = w_{i,k} B_{i,k-1}(x) + (1−w_{i+1,k}) B_{i+1,k-1}(x)

    where w_{i,k} = (x − knots[i]) / (knots[i+k] − knots[i]).

    Args:
        x: evaluation point.
        knots: knot vector (sorted, with multiplicity).
        degree: spline degree (0 = piecewise constant, 3 = cubic).
        i: basis function index.
    """
    t = np.asarray(knots, dtype=float)

    if degree == 0:
        if t[i] <= x < t[i + 1]:
            return 1.0
        return 0.0

    left = 0.0
    d1 = t[i + degree] - t[i]
    if d1 > 0:
        left = (x - t[i]) / d1 * bspline_basis(x, t, degree - 1, i)

    right = 0.0
    d2 = t[i + degree + 1] - t[i + 1]
    if d2 > 0:
        right = (t[i + degree + 1] - x) / d2 * bspline_basis(x, t, degree - 1, i + 1)

    return left + right
