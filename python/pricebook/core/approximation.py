"""Approximation theory: Chebyshev interpolation, Padé, error bounds.

Phase M8 slices 197-198 consolidated. Also the single source of truth for the
Chebyshev fundamentals (nodes, DCT coefficients, Clenshaw evaluation): the
spectral-methods layer (``numerical._spectral``) imports them from here rather
than re-implementing — ``core`` is the lower layer, so the dependency points
downward.

* :func:`chebyshev_nodes` — Chebyshev-Gauss-Lobatto collocation points.
* :func:`chebyshev_coefficients` — DCT coefficients from node values.
* :func:`chebyshev_evaluate` — Clenshaw evaluation at arbitrary points.
* :func:`chebyshev_interpolate` — near-optimal polynomial interpolation.
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


def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [a, b].

    x_j = (a+b)/2 + (b-a)/2 · cos(πj/n), j = 0, ..., n. Clustered near the
    endpoints, which avoids the Runge phenomenon.

    Requires n >= 1 (a Lobatto set needs at least two endpoints); n=0 is a
    degree-0 constant, handled by the interpolators directly, not here.
    """
    if n < 1:
        raise ValueError(f"chebyshev_nodes requires n >= 1, got {n}")
    j = np.arange(n + 1)
    return 0.5 * (a + b) + 0.5 * (b - a) * np.cos(np.pi * j / n)


def chebyshev_coefficients(values: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients c_k from values at the Lobatto nodes (DCT-I).

    `values` must be f sampled at ``chebyshev_nodes(len(values) - 1)``, in node
    order. The endpoint weighting and the c_0 / c_n halving give the standard
    interpolation coefficients.
    """
    values = np.asarray(values, dtype=float)
    n = len(values) - 1
    if n == 0:
        return values.copy()

    coeffs = np.zeros(n + 1)
    for j in range(n + 1):
        s = 0.0
        for i in range(n + 1):
            w = 1.0 if (i == 0 or i == n) else 2.0
            s += w * values[i] * math.cos(j * i * math.pi / n)
        coeffs[j] = s / n
    coeffs[0] /= 2
    coeffs[n] /= 2
    return coeffs


def chebyshev_evaluate(
    coeffs: np.ndarray,
    x: float | np.ndarray,
    a: float = -1.0,
    b: float = 1.0,
) -> float | np.ndarray:
    """Evaluate a Chebyshev series Σ c_k T_k on [a, b] via Clenshaw recurrence.

    Numerically stable. Queries outside [a, b] are clamped to the boundary
    rather than extrapolated (polynomial extrapolation blows up fast).

    Return contract: scalar ``x`` → Python ``float``; array ``x`` → ``np.ndarray``.
    """
    x_arr = np.asarray(x, dtype=float)
    # Map [a, b] → [-1, 1], clamped to guard against overshoot / extrapolation.
    xi = np.clip((2 * x_arr - a - b) / (b - a), -1.0, 1.0)
    out = _clenshaw(np.asarray(coeffs, dtype=float), xi)
    return float(out) if x_arr.ndim == 0 else out


def _clenshaw(coeffs: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Clenshaw recurrence for Σ c_k T_k(xi), xi ∈ [-1, 1].

    Uses T_{k+1} = 2·xi·T_k - T_{k-1}.
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


@dataclass
class ChebyshevInterpolant:
    """Chebyshev polynomial interpolant on [a, b]."""

    coefficients: np.ndarray  # Chebyshev coefficients c_k
    a: float
    b: float
    degree: int

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the interpolant at x (Clenshaw; out-of-domain clamped)."""
        return chebyshev_evaluate(self.coefficients, x, self.a, self.b)

    def max_coeff_magnitude(self) -> float:
        """Magnitude of the last few coefficients (convergence diagnostic)."""
        return float(np.max(np.abs(self.coefficients[-3:])))

    def to_dict(self) -> dict:
        return dict(vars(self))


def chebyshev_interpolate(
    f: Callable[[float | np.ndarray], float | np.ndarray],
    a: float,
    b: float,
    n: int = 20,
) -> ChebyshevInterpolant:
    """Interpolate f on [a, b] using n+1 Chebyshev-Lobatto points.

    Near-minimax: the interpolation error is within a factor O(log n) of the
    best polynomial approximation. f is called once on the node array, so it
    must accept a NumPy array.

    Args:
        f: function to interpolate (vectorised over a NumPy array).
        a, b: interval.
        n: polynomial degree.

    Returns:
        :class:`ChebyshevInterpolant`.

    Reference:
        Trefethen, ATAP, Ch. 4-5.
    """
    if b == a:
        raise ValueError(f"degenerate interval: a == b == {a}")
    if n == 0:
        # Degree-0: constant interpolant at the interval midpoint.
        mid = 0.5 * (a + b)
        return ChebyshevInterpolant(np.array([float(f(mid))]), a, b, 0)

    nodes = chebyshev_nodes(n, a, b)
    coeffs = chebyshev_coefficients(np.asarray(f(nodes), dtype=float))
    return ChebyshevInterpolant(coeffs, a, b, n)


# ---- Padé approximant ----


@dataclass
class PadeApproximant:
    """Padé [L/M] rational approximation: P_L(x) / Q_M(x)."""

    numerator: np.ndarray  # coefficients of P (degree L)
    denominator: np.ndarray  # coefficients of Q (degree M), Q[0] = 1
    L: int
    M: int

    def evaluate(self, x: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the Padé approximant at x.

        Scalar ``x`` → Python ``float``; array ``x`` → ``np.ndarray``.
        """
        x_arr = np.asarray(x, dtype=float)
        num = np.polyval(self.numerator[::-1], x_arr)
        den = np.polyval(self.denominator[::-1], x_arr)
        out = num / den
        return float(out) if x_arr.ndim == 0 else out

    def to_dict(self) -> dict:
        return dict(vars(self))


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

        # The denominator system is classically ill-conditioned for high M.
        # A singular (or near-singular) system means no Padé [L/M] exists for
        # these coefficients — fail loud rather than silently returning a
        # degree-L Taylor truncation that violates the order-(L+M) contract.
        cond = np.linalg.cond(A)
        if not np.isfinite(cond) or cond > 1e12:
            raise ValueError(
                f"Padé [{L}/{M}] denominator system is singular or "
                f"ill-conditioned; reduce M or check the Taylor coefficients."
            )
        q = np.linalg.solve(A, b_vec)

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

    def to_dict(self) -> dict:
        return dict(vars(self))


def richardson_table(
    values: list[float],
    order: int = 2,
) -> RichardsonTable:
    """Build a full Richardson extrapolation table.

    Given estimates f(h), f(h/2), f(h/4), ..., build the
    triangular table where each column cancels one more order
    of the leading error term.

    Assumes the error expansion is *geometric* in the base order, i.e.
    f(h) = A + c_1 h^p + c_2 h^{2p} + ... (column j eliminates the h^{jp}
    term via the factor 2^{p·j}). For a series with consecutive-order terms
    (e.g. h^p, h^{p+1}, ...) this cancels only the h^{jp} terms and is
    sub-optimal — supply estimates whose error is geometric in `order`.

    Args:
        values: estimates at h, h/2, h/4, ... (at least one; ≥2 to extrapolate).
        order: base convergence order p.

    Returns:
        :class:`RichardsonTable` with the full table and best estimate.
    """
    if not values:
        raise ValueError("richardson_table requires at least one estimate")
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

    # Guard the index: a negative i would silently wrap via numpy indexing
    # and return a plausible-but-wrong value; an out-of-range i must error.
    if not 0 <= i <= len(t) - degree - 2:
        raise IndexError(
            f"basis index i={i} out of range for {len(t)} knots, degree {degree} "
            f"(valid: 0..{len(t) - degree - 2})"
        )

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
