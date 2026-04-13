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

from pricebook.black76 import OptionType


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

    # Backward iteration through exercise dates
    for step in range(n_exercise - 1, -1, -1):
        # Compute continuation value at each grid point via COS:
        # C(x_i) = df × Σ_{k=0}^{N-1} Re[φ(u_k) c_k] cos(u_k(x_i - a))
        # where c_k are the COS coefficients of the current value function.

        # Step 1: compute COS coefficients of current value via DCT-like sum
        c = np.zeros(N)
        dx_grid = (b - a) / (n_grid - 1)
        for k in range(N):
            u_k = k * math.pi / (b - a)
            # Trapezoidal integration: c_k = (2/(b-a)) ∫ v(x) cos(u_k(x-a)) dx
            integrand = value_grid * np.cos(u_k * (x_grid - a))
            c[k] = 2.0 / (b - a) * np.trapezoid(integrand, x_grid)
        c[0] *= 0.5

        # Step 2: multiply by CF and discount
        for k in range(N):
            u_k = k * math.pi / (b - a)
            phi_k = char_func_dt(u_k)
            c[k] = df_step * (phi_k * c[k]).real

        # Step 3: evaluate continuation at grid points
        cont_grid = np.zeros(n_grid)
        for i in range(n_grid):
            for k in range(N):
                u_k = k * math.pi / (b - a)
                cont_grid[i] += c[k] * math.cos(u_k * (x_grid[i] - a))

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
