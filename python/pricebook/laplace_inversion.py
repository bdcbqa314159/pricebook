"""Laplace transform inversion: Talbot, Euler acceleration, Gaver-Stehfest.

Numerical inversion of Laplace transforms for pricing problems where the
Laplace-domain solution is known analytically (affine models, first-passage
times, CIR bond pricing).

* :func:`talbot_inversion` — deformed Bromwich contour (Talbot 1979).
* :func:`euler_inversion` — Euler summation with Richardson acceleration.
* :func:`gaver_stehfest` — real-valued inversion (Stehfest 1970).

References:
    Abate & Whitt, *Numerical Inversion of Laplace Transforms of
    Probability Distributions*, ORSA J. Computing, 1995.
    Talbot, *The Accurate Numerical Inversion of Laplace Transforms*,
    J. Inst. Maths Applics, 1979.
    Stehfest, *Algorithm 368: Numerical Inversion of Laplace Transforms*,
    Comm. ACM, 1970.
"""

from __future__ import annotations

import math
import cmath
from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---- Talbot inversion ----

def talbot_inversion(
    F: Callable[[complex], complex],
    t: float,
    N: int = 32,
) -> float:
    """Talbot's method: deformed Bromwich contour for Laplace inversion.

    Uses the fixed Talbot contour. The contour wraps around the
    negative real axis, giving exponential convergence for smooth f.

    The method evaluates:
        f(t) ≈ (2/5t) Σ_{k=0}^{N-1} Re[ exp(s_k t) F(s_k) γ_k ]

    with the contour s_k parametrised by θ_k = kπ/N.

    Args:
        F: Laplace transform F(s) accepting complex s.
        t: time at which to evaluate f(t). Must be > 0.
        N: number of quadrature points (higher = more accurate).

    Returns:
        Approximate f(t).

    Reference:
        Talbot, J. Inst. Maths Applics, 1979.
        Weideman, SIAM J. Numer. Anal., 2006.
    """
    if t <= 0:
        return 0.0

    r = 2.0 * N / (5.0 * t)

    # k = 0: s_0 = r, weight = 0.5
    ans = 0.5 * (cmath.exp(r * t) * F(complex(r, 0))).real

    for k in range(1, N):
        theta = k * math.pi / N
        cot_t = math.cos(theta) / math.sin(theta)
        sin_t = math.sin(theta)

        # Contour: s = r (1 + θ cot θ − θ/θ + iθ) ... simplified:
        # Use the standard parametrisation:
        #   s(θ) = r θ (cot θ + i)
        s = r * theta * complex(cot_t, 1.0)

        # Weight: ds/dθ = r (cot θ + i − θ/sin²θ + i) ... simplified:
        #   ds/dθ = r (cot θ − θ/sin²θ + i)
        # Actually for the trapezoidal sum the weight is just 1 (uniform θ spacing).
        # The Talbot method needs: Re[exp(st) F(s) (1 + i(θ + θ cot²θ − cot θ))]
        inner = 1.0 + 1j * (theta + theta * cot_t * cot_t - cot_t)

        ans += (cmath.exp(s * t) * F(s) * inner).real

    return (r / N) * ans


# ---- Euler inversion (Abate-Whitt) ----

def euler_inversion(
    F: Callable[[complex], complex],
    t: float,
    N: int = 32,
    M: int = 11,
    A: float = 18.4,
) -> float:
    """Euler summation with Richardson acceleration for Laplace inversion.

    The Bromwich integral is discretised on the line Re(s) = A/(2t),
    giving a Fourier series. Euler summation accelerates convergence.

    Args:
        F: Laplace transform F(s).
        t: time at which to evaluate f(t). Must be > 0.
        N: number of terms in the Euler sum.
        M: number of terms for Richardson extrapolation.
        A: shift parameter (default 18.4 from Abate-Whitt).

    Returns:
        Approximate f(t).

    Reference:
        Abate & Whitt, ORSA J. Computing 7(1), 1995, Algorithm 1.
    """
    if t <= 0:
        return 0.0

    h = A / (2 * t)

    # Compute partial sums
    total = 0.5 * F(complex(h, 0)).real
    for k in range(1, N + M + 1):
        s = complex(h, k * math.pi / t)
        total += (-1) ** k * F(s).real

    # Euler summation for acceleration
    terms = []
    partial = 0.5 * F(complex(h, 0)).real
    for k in range(1, N + M + 1):
        s = complex(h, k * math.pi / t)
        partial += (-1) ** k * F(s).real
        if k >= N:
            terms.append(partial)

    # Binomial weights for Euler acceleration
    result = 0.0
    for j in range(M + 1):
        weight = math.comb(M, j) / (2 ** M)
        result += weight * terms[j] if j < len(terms) else 0.0

    return math.exp(A / 2) / t * result


# ---- Gaver-Stehfest (real-valued) ----

def gaver_stehfest(
    F: Callable[[float], float],
    t: float,
    N: int = 12,
) -> float:
    """Gaver-Stehfest inversion using only real evaluations of F(s).

    Useful when F(s) is only defined for real s (e.g. from simulation
    or empirical data). Less accurate than Talbot/Euler but needs no
    complex arithmetic.

    Args:
        F: Laplace transform F(s) accepting real s > 0.
        t: time at which to evaluate f(t). Must be > 0.
        N: number of terms (must be even; typical 8-14).

    Returns:
        Approximate f(t).

    Reference:
        Stehfest, Comm. ACM 13(1), 1970.
    """
    if t <= 0:
        return 0.0
    if N % 2 != 0:
        N += 1

    # Gaver-Stehfest uses the Gaver functional:
    #   f_N(t) = ln2/t × Σ V_k × F(k ln2/t)
    # where V_k are combinatorial weights. We compute via the
    # simpler Gaver sequence with Salzer acceleration.
    ln2 = math.log(2)
    half = N // 2

    # Direct computation of Stehfest coefficients via the recursion
    # from the original paper (corrected sign convention).
    V = [0.0] * (N + 1)
    for i in range(1, N + 1):
        vi = 0.0
        for k in range(int(math.floor((i + 1) / 2)), min(i, half) + 1):
            vi += (
                math.pow(k, half) * math.factorial(2 * k)
                / (math.factorial(half - k)
                   * math.factorial(k)
                   * math.factorial(k - 1)
                   * math.factorial(i - k)
                   * math.factorial(2 * k - i))
            )
        V[i] = math.pow(-1, i + half) * vi

    result = 0.0
    for i in range(1, N + 1):
        result += V[i] * F(i * ln2 / t)

    return ln2 / t * result


# ---- Convenience: CIR bond price via Laplace ----

@dataclass
class LaplaceInversionResult:
    """Result with value and method info."""
    value: float
    method: str
    n_evaluations: int


def invert(
    F: Callable,
    t: float,
    method: str = "talbot",
    N: int = 32,
) -> LaplaceInversionResult:
    """Unified Laplace inversion interface.

    Args:
        F: Laplace transform (complex-valued for talbot/euler,
           real-valued for stehfest).
        t: evaluation time.
        method: "talbot", "euler", or "stehfest".
        N: number of terms.
    """
    if method == "talbot":
        val = talbot_inversion(F, t, N)
        return LaplaceInversionResult(val, "talbot", N)
    elif method == "euler":
        val = euler_inversion(F, t, N)
        return LaplaceInversionResult(val, "euler", N + 11)
    elif method == "stehfest":
        val = gaver_stehfest(F, t, N)
        return LaplaceInversionResult(val, "stehfest", N)
    else:
        raise ValueError(f"Unknown method: {method}")
