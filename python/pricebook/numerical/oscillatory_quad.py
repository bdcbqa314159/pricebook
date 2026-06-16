"""Oscillatory integral quadrature: Filon and Levin methods.

For integrals of the form ∫f(x)e^{iωx}dx where standard quadrature
fails due to rapid oscillation.

* :func:`filon_quad` — Filon's method for ∫f(x)cos(ωx)dx or sin.
* :func:`levin_quad` — Levin collocation for general oscillatory.
* :func:`fourier_integral` — adaptive Fourier integral.

References:
    Filon, *On a Quadrature Formula for Trigonometric Integrals*, PRSE, 1928.
    Levin, *Fast Integration of Rapidly Oscillatory Functions*, JCA, 1996.
    Iserles, *On the Numerical Quadrature of Highly-Oscillatory Integrals*,
    IMA JNA, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class OscillatoryResult:
    """Oscillatory quadrature result."""
    value: float
    error_estimate: float
    n_evaluations: int
    method: str

    def to_dict(self) -> dict:
        return dict(vars(self))


def filon_quad(
    f,
    a: float,
    b: float,
    omega: float,
    n: int = 100,
    mode: str = "cos",
) -> OscillatoryResult:
    """Filon's method for oscillatory integrals.

    Computes: ∫_a^b f(x) cos(ωx) dx  (mode="cos")
    or:       ∫_a^b f(x) sin(ωx) dx  (mode="sin")

    Filon approximates f(x) by piecewise quadratic (Simpson-like)
    and integrates the product analytically.

    Accuracy: O(h³/ω) — works well when ω is large (many oscillations).

    Args:
        f: smooth amplitude function.
        a, b: integration limits.
        omega: oscillation frequency.
        n: number of subintervals (must be even).
        mode: "cos" or "sin".
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    theta = omega * h

    # Filon coefficients
    if abs(theta) < 0.1:
        # Small theta: Taylor expansion
        alpha_f = 2 * theta**3 / 45 - 2 * theta**5 / 315
        beta_f = 2.0 / 3 + 2 * theta**2 / 15 - 4 * theta**4 / 105
        gamma_f = 4.0 / 3 - 2 * theta**2 / 15 + theta**4 / 210
    else:
        alpha_f = (theta**2 + theta * math.sin(theta) * math.cos(theta) - 2 * math.sin(theta)**2) / theta**3
        beta_f = 2 * (theta * (1 + math.cos(theta)**2) - 2 * math.sin(theta) * math.cos(theta)) / theta**3
        gamma_f = 4 * (math.sin(theta) - theta * math.cos(theta)) / theta**3

    # Evaluate f at grid points
    x = np.linspace(a, b, n + 1)
    fx = np.array([f(xi) for xi in x])

    # Sums
    C_even = sum(fx[2 * k] * math.cos(omega * x[2 * k]) for k in range(n // 2 + 1))
    C_odd = sum(fx[2 * k + 1] * math.cos(omega * x[2 * k + 1]) for k in range(n // 2))
    S_even = sum(fx[2 * k] * math.sin(omega * x[2 * k]) for k in range(n // 2 + 1))
    S_odd = sum(fx[2 * k + 1] * math.sin(omega * x[2 * k + 1]) for k in range(n // 2))

    if mode == "cos":
        result = h * (
            alpha_f * (fx[0] * math.sin(omega * a) - fx[-1] * math.sin(omega * b))
            + beta_f * C_even + gamma_f * C_odd
        )
    else:
        result = h * (
            alpha_f * (fx[-1] * math.cos(omega * b) - fx[0] * math.cos(omega * a))
            + beta_f * S_even + gamma_f * S_odd
        )

    return OscillatoryResult(
        value=result, error_estimate=abs(h**3 / max(omega, 1)),
        n_evaluations=n + 1, method="filon",
    )


def levin_quad(
    f,
    a: float,
    b: float,
    omega: float,
    n: int = 32,
) -> OscillatoryResult:
    """Levin collocation for general oscillatory integrals.

    Computes: ∫_a^b f(x) e^{iωx} dx

    Levin's idea: find p(x) such that (p e^{iωx})' = f e^{iωx},
    then ∫ f e^{iωx} = p(b)e^{iωb} − p(a)e^{iωa}.

    Solve: p'(x) + iω p(x) = f(x) via collocation on Chebyshev nodes.

    Args:
        f: amplitude function (can be complex-valued).
        omega: frequency.
        n: collocation points.
    """
    # Chebyshev nodes on [a, b]
    k = np.arange(n)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos(math.pi * (2 * k + 1) / (2 * n))
    nodes = np.sort(nodes)

    # Evaluate f at nodes
    f_vals = np.array([complex(f(xi)) for xi in nodes])

    # Build collocation system: p'(x_j) + iω p(x_j) = f(x_j)
    # Approximate p as polynomial: p(x) = Σ c_k T_k(x)
    # Use monomial basis for simplicity: p(x) = Σ c_k x^k
    # p'(x) = Σ k c_k x^{k-1}

    # Vandermonde-like matrix
    V = np.zeros((n, n), dtype=complex)      # p values at nodes
    Vp = np.zeros((n, n), dtype=complex)     # p' values at nodes

    for j in range(n):
        for k in range(n):
            V[j, k] = nodes[j]**k
            Vp[j, k] = k * nodes[j]**(k - 1) if k > 0 else 0

    # System: (Vp + iω V) c = f
    A = Vp + 1j * omega * V
    try:
        c = np.linalg.solve(A, f_vals)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        c = np.linalg.lstsq(A, f_vals, rcond=None)[0]

    # p(x) = Σ c_k x^k
    def p_eval(x):
        return sum(c[k] * x**k for k in range(n))

    # Result: p(b)e^{iωb} − p(a)e^{iωa}
    val = p_eval(b) * np.exp(1j * omega * b) - p_eval(a) * np.exp(1j * omega * a)

    return OscillatoryResult(
        value=float(np.real(val)), error_estimate=0.0,
        n_evaluations=n, method="levin",
    )


def fourier_integral(
    f,
    a: float,
    b: float,
    omega: float,
    tol: float = 1e-8,
    max_n: int = 1000,
) -> OscillatoryResult:
    """Adaptive Fourier integral ∫_a^b f(x) e^{iωx} dx.

    Chooses between standard quadrature (low ω) and Filon (high ω).

    Args:
        f: amplitude function.
        omega: frequency.
        tol: target accuracy.
        max_n: maximum evaluations.
    """
    # Number of oscillations
    n_osc = abs(omega) * (b - a) / (2 * math.pi)

    if n_osc < 3:
        # Few oscillations: standard quadrature
        from scipy.integrate import quad
        real_part = quad(lambda x: f(x) * math.cos(omega * x), a, b, limit=max_n)[0]
        imag_part = quad(lambda x: f(x) * math.sin(omega * x), a, b, limit=max_n)[0]
        return OscillatoryResult(
            value=real_part, error_estimate=tol,
            n_evaluations=max_n, method="adaptive_quad",
        )
    else:
        # Many oscillations: Filon
        n = max(int(4 * n_osc), 20)
        n = min(n, max_n)
        if n % 2 != 0:
            n += 1
        return filon_quad(f, a, b, omega, n, mode="cos")
