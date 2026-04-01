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
from dataclasses import dataclass

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


@dataclass
class _FDGrid:
    """Shared finite difference grid and coefficients."""

    x0: float
    dx: float
    x: np.ndarray
    S: np.ndarray
    V: np.ndarray
    dt: float
    n_spot: int
    n_time: int
    n_int: int
    rate: float
    a_coef: float
    b_coef: float
    c_coef: float
    option_type: OptionType
    strike: float
    # Pre-allocated CN tridiagonal arrays
    cn_a: np.ndarray
    cn_b: np.ndarray
    cn_c: np.ndarray


def _build_grid(
    spot: float, strike: float, rate: float, vol: float, T: float,
    option_type: OptionType, div_yield: float,
    n_spot: int, n_time: int, spot_range: float,
) -> _FDGrid:
    mu = rate - div_yield
    dt = T / n_time
    x0 = math.log(spot)
    width = spot_range * vol * math.sqrt(T)
    x_min, x_max = x0 - width, x0 + width
    dx = (x_max - x_min) / n_spot
    x = np.linspace(x_min, x_max, n_spot + 1)
    S = np.exp(x)

    if option_type == OptionType.CALL:
        V = np.maximum(S - strike, 0.0)
    else:
        V = np.maximum(strike - S, 0.0)

    alpha = 0.5 * vol**2
    beta = mu - 0.5 * vol**2
    n_int = n_spot - 1

    a_coef = alpha / dx**2 - beta / (2 * dx)
    b_coef = -2 * alpha / dx**2 - rate
    c_coef = alpha / dx**2 + beta / (2 * dx)

    theta = 0.5
    cn_a = np.full(n_int, -(1 - theta) * dt * a_coef)
    cn_b = np.full(n_int, 1 - (1 - theta) * dt * b_coef)
    cn_c = np.full(n_int, -(1 - theta) * dt * c_coef)

    return _FDGrid(
        x0=x0, dx=dx, x=x, S=S, V=V, dt=dt,
        n_spot=n_spot, n_time=n_time, n_int=n_int,
        rate=rate, a_coef=a_coef, b_coef=b_coef, c_coef=c_coef,
        option_type=option_type, strike=strike,
        cn_a=cn_a, cn_b=cn_b, cn_c=cn_c,
    )


def _apply_boundary(g: _FDGrid, tau: float) -> None:
    """Apply boundary conditions in place."""
    if g.option_type == OptionType.CALL:
        g.V[0] = 0.0
        g.V[-1] = g.S[-1] - g.strike * math.exp(-g.rate * max(tau, 0))
    else:
        g.V[-1] = 0.0
        g.V[0] = g.strike * math.exp(-g.rate * max(tau, 0)) - g.S[0]


def _implicit_step(g: _FDGrid, imp_a: np.ndarray, imp_b: np.ndarray, imp_c: np.ndarray) -> None:
    """One fully-implicit time step in place."""
    d_arr = g.V[1:g.n_spot].copy()
    d_arr[0] += g.dt * g.a_coef * g.V[0]
    d_arr[-1] += g.dt * g.c_coef * g.V[-1]
    g.V[1:g.n_spot] = _thomas(imp_a, imp_b.copy(), imp_c, d_arr)


def _cn_step(g: _FDGrid) -> None:
    """One Crank-Nicolson time step in place."""
    theta = 0.5
    n = g.n_spot
    rhs = g.V[1:n] + theta * g.dt * (
        g.a_coef * g.V[0:n - 1] + g.b_coef * g.V[1:n] + g.c_coef * g.V[2:n + 1]
    )
    rhs[0] += (1 - theta) * g.dt * g.a_coef * g.V[0]
    rhs[-1] += (1 - theta) * g.dt * g.c_coef * g.V[-1]
    g.V[1:n] = _thomas(g.cn_a, g.cn_b.copy(), g.cn_c, rhs)


def _interpolate_at_spot(g: _FDGrid) -> float:
    """Interpolate the solution at the original spot."""
    idx = int(np.searchsorted(g.x, g.x0)) - 1
    idx = max(0, min(idx, g.n_spot - 1))
    w = (g.x0 - g.x[idx]) / g.dx
    return float(g.V[idx] * (1 - w) + g.V[idx + 1] * w)


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
    """European option price via finite difference."""
    if isinstance(scheme, str):
        scheme = FDScheme(scheme)

    g = _build_grid(spot, strike, rate, vol, T, option_type, div_yield,
                    n_spot, n_time, spot_range)

    # Pre-allocate implicit arrays if needed
    if scheme == FDScheme.IMPLICIT:
        imp_a = np.full(g.n_int, -g.dt * g.a_coef)
        imp_b = np.full(g.n_int, 1 - g.dt * g.b_coef)
        imp_c = np.full(g.n_int, -g.dt * g.c_coef)

    for step in range(g.n_time):
        tau = T - step * g.dt - g.dt
        _apply_boundary(g, tau)

        if scheme == FDScheme.EXPLICIT:
            V_new = g.V.copy()
            n = g.n_spot
            V_new[1:n] = g.V[1:n] + g.dt * (
                g.a_coef * g.V[0:n - 1] + g.b_coef * g.V[1:n] + g.c_coef * g.V[2:n + 1]
            )
            g.V = V_new

        elif scheme == FDScheme.IMPLICIT:
            _implicit_step(g, imp_a, imp_b, imp_c)

        else:
            _cn_step(g)

    return _interpolate_at_spot(g)


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
    """American option price via Crank-Nicolson with early exercise."""
    g = _build_grid(spot, strike, rate, vol, T, option_type, div_yield,
                    n_spot, n_time, spot_range)
    payoff = g.V.copy()

    for step in range(g.n_time):
        tau = T - step * g.dt - g.dt
        _apply_boundary(g, tau)
        _cn_step(g)
        g.V[1:g.n_spot] = np.maximum(g.V[1:g.n_spot], payoff[1:g.n_spot])

    return _interpolate_at_spot(g)


# ---------------------------------------------------------------------------
# Barrier options
# ---------------------------------------------------------------------------

def _apply_barrier(g: _FDGrid, lower: float | None, upper: float | None) -> None:
    """Zero out grid values at barrier levels."""
    if lower is not None:
        g.V[g.S <= lower] = 0.0
    if upper is not None:
        g.V[g.S >= upper] = 0.0


def fd_barrier_knockout(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    barrier_lower: float | None = None,
    barrier_upper: float | None = None,
    n_spot: int = 400,
    n_time: int = 400,
    spot_range: float = 4.0,
    rannacher_steps: int = 0,
) -> float:
    """Knock-out barrier option via Crank-Nicolson.

    Value = 0 if spot touches the barrier. Supports:
    - Down-and-out: barrier_lower set
    - Up-and-out: barrier_upper set
    - Double knock-out: both set

    Args:
        rannacher_steps: number of initial fully-implicit steps for smoothing.
    """
    g = _build_grid(spot, strike, rate, vol, T, option_type, div_yield,
                    n_spot, n_time, spot_range)

    _apply_barrier(g, barrier_lower, barrier_upper)

    if rannacher_steps > 0:
        imp_a = np.full(g.n_int, -g.dt * g.a_coef)
        imp_b = np.full(g.n_int, 1 - g.dt * g.b_coef)
        imp_c = np.full(g.n_int, -g.dt * g.c_coef)

    for step in range(g.n_time):
        tau = T - step * g.dt - g.dt
        _apply_boundary(g, tau)

        if rannacher_steps > 0 and step < rannacher_steps:
            _implicit_step(g, imp_a, imp_b, imp_c)
        else:
            _cn_step(g)

        _apply_barrier(g, barrier_lower, barrier_upper)

    return _interpolate_at_spot(g)


def fd_barrier_knockin(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    barrier_lower: float | None = None,
    barrier_upper: float | None = None,
    n_spot: int = 400,
    n_time: int = 400,
    spot_range: float = 4.0,
) -> float:
    """Knock-in barrier option via in-out parity: knock-in = vanilla - knock-out."""
    vanilla = fd_european(spot, strike, rate, vol, T, option_type, div_yield,
                          n_spot, n_time, spot_range, scheme="cn")
    knockout = fd_barrier_knockout(spot, strike, rate, vol, T, option_type, div_yield,
                                   barrier_lower, barrier_upper, n_spot, n_time, spot_range)
    return vanilla - knockout
