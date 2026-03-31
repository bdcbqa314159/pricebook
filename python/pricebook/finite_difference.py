"""
Finite difference option pricing on the Black-Scholes PDE.

Solves: dV/dt + 0.5*vol^2*S^2*d2V/dS2 + (r-q)*S*dV/dS - r*V = 0

in log-spot space x = ln(S), backward from T to 0.

Schemes:
    - Explicit: forward Euler (conditionally stable)
    - Implicit: backward Euler (unconditionally stable, 1st order)
    - Crank-Nicolson: average of explicit+implicit (unconditionally stable, 2nd order)

    price = fd_european(spot=100, strike=105, rate=0.05, vol=0.20,
                        T=1.0, option_type=OptionType.CALL, scheme="cn")
"""

from __future__ import annotations

import math
from enum import Enum

import numpy as np

from pricebook.black76 import OptionType


class FDScheme(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CRANK_NICOLSON = "cn"


def _thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Tridiagonal solver (Thomas algorithm). Modifies inputs in place."""
    n = len(d)
    c = c.copy()
    d = d.copy()
    for i in range(1, n):
        m = a[i] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
    x = np.empty(n)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def fd_european(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_spot: int = 200,
    n_time: int = 200,
    spot_range: float = 4.0,
    scheme: FDScheme | str = FDScheme.CRANK_NICOLSON,
) -> float:
    """European option price via finite difference.

    Args:
        spot: initial price.
        strike: option strike.
        rate: risk-free rate.
        vol: lognormal volatility.
        T: time to expiry.
        option_type: CALL or PUT.
        div_yield: continuous dividend yield.
        n_spot: number of spatial grid points.
        n_time: number of time steps.
        spot_range: grid extends spot_range * vol * sqrt(T) in log-space.
        scheme: "explicit", "implicit", or "cn" (Crank-Nicolson).
    """
    if isinstance(scheme, str):
        scheme = FDScheme(scheme)

    mu = rate - div_yield
    dt = T / n_time
    x0 = math.log(spot)
    width = spot_range * vol * math.sqrt(T)
    x_min, x_max = x0 - width, x0 + width
    dx = (x_max - x_min) / n_spot
    x = np.linspace(x_min, x_max, n_spot + 1)
    S = np.exp(x)

    # Terminal payoff
    if option_type == OptionType.CALL:
        V = np.maximum(S - strike, 0.0)
    else:
        V = np.maximum(strike - S, 0.0)

    # PDE coefficients in log-space: dV/dt + alpha*d2V/dx2 + beta*dV/dx - r*V = 0
    alpha = 0.5 * vol**2
    beta = mu - 0.5 * vol**2

    # Tridiagonal coefficients (interior points only: 1..n_spot-1)
    n_int = n_spot - 1  # interior point count

    a_coef = (alpha / dx**2 - beta / (2 * dx))
    b_coef = (-2 * alpha / dx**2 - rate)
    c_coef = (alpha / dx**2 + beta / (2 * dx))

    # Pre-allocate tridiagonal arrays (copied per step since Thomas modifies them)
    cn_theta = 0.5
    imp_a = np.full(n_int, -dt * a_coef)
    imp_b = np.full(n_int, 1 - dt * b_coef)
    imp_c = np.full(n_int, -dt * c_coef)
    cn_a = np.full(n_int, -(1 - cn_theta) * dt * a_coef)
    cn_b = np.full(n_int, 1 - (1 - cn_theta) * dt * b_coef)
    cn_c = np.full(n_int, -(1 - cn_theta) * dt * c_coef)

    for _ in range(n_time):
        # Boundary conditions at each time step
        tau = T - _ * dt  # remaining time (approximate)
        if option_type == OptionType.CALL:
            V[0] = 0.0
            V[-1] = S[-1] - strike * math.exp(-rate * max(tau - dt, 0))
        else:
            V[-1] = 0.0
            V[0] = strike * math.exp(-rate * max(tau - dt, 0)) - S[0]

        if scheme == FDScheme.EXPLICIT:
            V_new = V.copy()
            V_new[1:n_spot] = V[1:n_spot] + dt * (
                a_coef * V[0:n_spot - 1] + b_coef * V[1:n_spot] + c_coef * V[2:n_spot + 1]
            )
            V = V_new

        elif scheme == FDScheme.IMPLICIT:
            d_arr = V[1:n_spot].copy()
            d_arr[0] += dt * a_coef * V[0]
            d_arr[-1] += dt * c_coef * V[-1]
            V[1:n_spot] = _thomas(imp_a.copy(), imp_b.copy(), imp_c.copy(), d_arr)

        else:  # Crank-Nicolson
            rhs = V[1:n_spot] + cn_theta * dt * (
                a_coef * V[0:n_spot - 1] + b_coef * V[1:n_spot] + c_coef * V[2:n_spot + 1]
            )
            rhs[0] += (1 - cn_theta) * dt * a_coef * V[0]
            rhs[-1] += (1 - cn_theta) * dt * c_coef * V[-1]
            V[1:n_spot] = _thomas(cn_a.copy(), cn_b.copy(), cn_c.copy(), rhs)

    # Interpolate at spot
    idx = np.searchsorted(x, x0) - 1
    idx = max(0, min(idx, n_spot - 1))
    w = (x0 - x[idx]) / dx
    return float(V[idx] * (1 - w) + V[idx + 1] * w)


def fd_american(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_spot: int = 200,
    n_time: int = 200,
    spot_range: float = 4.0,
) -> float:
    """American option price via Crank-Nicolson with early exercise.

    At each time step: V = max(continuation, exercise).
    """
    mu = rate - div_yield
    dt = T / n_time
    x0 = math.log(spot)
    width = spot_range * vol * math.sqrt(T)
    x_min, x_max = x0 - width, x0 + width
    dx = (x_max - x_min) / n_spot
    x = np.linspace(x_min, x_max, n_spot + 1)
    S = np.exp(x)

    if option_type == OptionType.CALL:
        payoff = np.maximum(S - strike, 0.0)
    else:
        payoff = np.maximum(strike - S, 0.0)

    V = payoff.copy()

    alpha = 0.5 * vol**2
    beta = mu - 0.5 * vol**2
    a_coef = (alpha / dx**2 - beta / (2 * dx))
    b_coef = (-2 * alpha / dx**2 - rate)
    c_coef = (alpha / dx**2 + beta / (2 * dx))

    n_int = n_spot - 1
    theta = 0.5

    cn_a = np.full(n_int, -(1 - theta) * dt * a_coef)
    cn_b = np.full(n_int, 1 - (1 - theta) * dt * b_coef)
    cn_c = np.full(n_int, -(1 - theta) * dt * c_coef)

    for step in range(n_time):
        tau = T - step * dt
        if option_type == OptionType.CALL:
            V[0] = 0.0
            V[-1] = S[-1] - strike * math.exp(-rate * max(tau - dt, 0))
        else:
            V[-1] = 0.0
            V[0] = strike * math.exp(-rate * max(tau - dt, 0)) - S[0]

        rhs = V[1:n_spot] + theta * dt * (
            a_coef * V[0:n_spot - 1] + b_coef * V[1:n_spot] + c_coef * V[2:n_spot + 1]
        )
        rhs[0] += (1 - theta) * dt * a_coef * V[0]
        rhs[-1] += (1 - theta) * dt * c_coef * V[-1]

        V[1:n_spot] = _thomas(cn_a.copy(), cn_b.copy(), cn_c.copy(), rhs)

        # Early exercise
        V[1:n_spot] = np.maximum(V[1:n_spot], payoff[1:n_spot])

    idx = np.searchsorted(x, x0) - 1
    idx = max(0, min(idx, n_spot - 1))
    w = (x0 - x[idx]) / dx
    return float(V[idx] * (1 - w) + V[idx + 1] * w)
