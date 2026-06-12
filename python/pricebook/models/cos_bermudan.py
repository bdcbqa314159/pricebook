"""COS method for Bermudan/American options via backward iteration.

Extends the European COS method (:mod:`pricebook.cos_method`) to
early-exercise options by iterating backward through exercise dates,
using the COS expansion to compute the continuation value at each step.

* :func:`cos_bermudan` — Bermudan option price via backward COS iteration.
* :func:`cos_american` — American approximation (many exercise dates).

References:
    Fang & Oosterlee, *Pricing Early-Exercise and Discrete Barrier
    Options by Fourier-Cosine Series Expansions*, Numer. Math., 2009.
"""

from __future__ import annotations

import math
import cmath
from typing import Callable

import numpy as np

from pricebook.models.black76 import OptionType


# ---- COS coefficients for payoff ----

def _chi(k: int, a: float, b: float, c: float, d: float) -> float:
    """χ_k(c,d) = ∫_c^d exp(x) cos(kπ(x-a)/(b-a)) dx."""
    w = k * math.pi / (b - a)
    if k == 0:
        return math.exp(d) - math.exp(c)
    denom = 1 + w * w
    return (
        (math.exp(d) * (math.cos(w * (d - a)) + w * math.sin(w * (d - a)))
         - math.exp(c) * (math.cos(w * (c - a)) + w * math.sin(w * (c - a))))
        / denom
    )


def _psi(k: int, a: float, b: float, c: float, d: float) -> float:
    """ψ_k(c,d) = ∫_c^d cos(kπ(x-a)/(b-a)) dx."""
    if k == 0:
        return d - c
    w = k * math.pi / (b - a)
    return (math.sin(w * (d - a)) - math.sin(w * (c - a))) / w


def _payoff_coefficients(
    N: int,
    a: float,
    b: float,
    strike: float,
    option_type: OptionType,
) -> np.ndarray:
    """COS coefficients V_k for the payoff function."""
    V = np.zeros(N)
    for k in range(N):
        if option_type == OptionType.CALL:
            V[k] = 2.0 / (b - a) * (
                _chi(k, a, b, 0, b) - _psi(k, a, b, 0, b)
            )
        else:
            V[k] = 2.0 / (b - a) * (
                -_chi(k, a, b, a, 0) + _psi(k, a, b, a, 0)
            )
    V[0] *= 0.5
    return V


# ---- Backward COS iteration ----

def cos_bermudan(
    char_func_dt: Callable,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    n_exercise: int,
    option_type: OptionType = OptionType.PUT,
    N: int = 128,
    L: float = 10.0,
) -> float:
    """Bermudan option price via backward COS iteration.

    At each exercise date (from maturity backward), computes the
    continuation value via COS expansion and compares it to the
    immediate exercise value.

    Args:
        char_func_dt: function that returns the characteristic function
            φ(u) of log(S_{t+dt}/S_t) for one time step dt = T/n_exercise.
            Called as ``char_func_dt(u)`` returning complex.
        spot: current spot price.
        strike: option strike.
        rate: risk-free rate.
        T: total time to maturity.
        n_exercise: number of equally-spaced exercise dates.
        option_type: CALL or PUT.
        N: number of COS terms.
        L: truncation parameter.

    Returns:
        Bermudan option price.

    Reference:
        Fang & Oosterlee, Numer. Math. 114, 2009, Algorithm 1.
    """
    dt = T / n_exercise
    df_step = math.exp(-rate * dt)
    x = math.log(spot / strike)

    # Estimate truncation range from the char func
    eps = 1e-4
    phi0 = char_func_dt(0.0)
    phi_p = char_func_dt(eps)
    phi_m = char_func_dt(-eps)
    ln0 = cmath.log(phi0) if abs(phi0) > 1e-20 else 0
    ln_p = cmath.log(phi_p)
    ln_m = cmath.log(phi_m)
    c1 = float((ln_p - ln_m).imag / (2 * eps))
    c2 = max(float(-(ln_p + ln_m - 2 * ln0).real / eps ** 2), 0.001)

    # Total range for the full time horizon
    a = -L * math.sqrt(c2 * n_exercise)
    b = L * math.sqrt(c2 * n_exercise)

    # Build x-grid for physical-space early exercise comparison
    n_grid = 2 * N
    x_grid = np.linspace(a, b, n_grid)

    # Payoff on x-grid: g(x) = (exp(x) - 1)^+ for call, (1 - exp(x))^+ for put
    if option_type == OptionType.CALL:
        payoff_grid = np.maximum(np.exp(x_grid) - 1.0, 0.0) * strike
    else:
        payoff_grid = np.maximum(1.0 - np.exp(x_grid), 0.0) * strike

    # Value at maturity = payoff
    value_grid = payoff_grid.copy()

    # Pre-compute φ(u_k) (constant across backward steps), and the cos / sin
    # basis matrices over x_grid. Fix T4-CB1: pre-fix the continuation
    # recursion was `c_new[k] = df · Re(φ(u_k) · c[k])` (with c[k] real),
    # which reduces to `df · Re(φ) · c[k]` and then evaluates
    # `cont = Σ c_new[k] · cos(u_k(x_i − a))` — DROPPING the Im(φ)·sin term
    # entirely.  The Fang-Oosterlee (2009) Bermudan recursion (their eq 2.10)
    # actually evaluates `Σ Re[φ(u_k) · exp(i·u_k·(x_i − a))] · V_k`, which
    # equals `Σ [Re(φ_k) · cos(u_k(x − a)) − Im(φ_k) · sin(u_k(x − a))] · V_k`.
    # Pre-fix the sin terms were missing, so any drifted process (e.g. BS
    # with r ≠ 0, or any jump model with non-zero mean jump) was mispriced.
    # On a vanilla BS American put (S=K=100, r=5%, σ=20%, T=1y) the pre-fix
    # COS price was ~15 % HIGHER than the PDE benchmark.
    u_vec = np.array([k * math.pi / (b - a) for k in range(N)])
    phi_vec = np.array([char_func_dt(uk) for uk in u_vec], dtype=complex)
    cos_basis = np.cos(np.outer(u_vec, x_grid - a))  # shape (N, n_grid)
    sin_basis = np.sin(np.outer(u_vec, x_grid - a))

    # Backward iteration through exercise dates
    for step in range(n_exercise - 1, -1, -1):
        # Step 1: compute COS coefficients V_k of current value via
        # trapezoidal quadrature on x_grid.
        V = np.zeros(N)
        for k in range(N):
            integrand = value_grid * cos_basis[k]
            V[k] = 2.0 / (b - a) * np.trapezoid(integrand, x_grid)
        V[0] *= 0.5

        # Step 2 + 3: combined evaluation —
        # cont_grid[i] = df · Σ_k V[k] · Re[φ(u_k) · exp(i u_k (x_i − a))]
        #             = df · Σ_k V[k] · [Re(φ_k) · cos_basis[k,i] − Im(φ_k) · sin_basis[k,i]]
        weighted_re = V * phi_vec.real
        weighted_im = V * phi_vec.imag
        cont_grid = df_step * (
            weighted_re @ cos_basis - weighted_im @ sin_basis
        )

        # Step 4: early exercise: value = max(continuation, payoff)
        value_grid = np.maximum(cont_grid, payoff_grid)

    # Interpolate at spot
    idx = np.searchsorted(x_grid, x) - 1
    idx = max(0, min(idx, n_grid - 2))
    w = (x - x_grid[idx]) / (x_grid[idx + 1] - x_grid[idx])
    return value_grid[idx] * (1 - w) + value_grid[idx + 1] * w


# ---- American approximation ----

def cos_american(
    char_func_dt: Callable,
    spot: float,
    strike: float,
    rate: float,
    T: float,
    option_type: OptionType = OptionType.PUT,
    n_exercise: int = 100,
    N: int = 128,
    L: float = 10.0,
) -> float:
    """American option price approximated as Bermudan with many exercise dates.

    Uses ``n_exercise=100`` by default (exercise every T/100).
    """
    return cos_bermudan(
        char_func_dt, spot, strike, rate, T, n_exercise,
        option_type, N, L,
    )
