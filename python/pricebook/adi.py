"""
ADI (Alternating Direction Implicit) 2D PDE solver.

Solves two-dimensional PDEs by splitting each time step into
two 1D implicit half-steps (one per spatial dimension). Enables
Heston PDE pricing and two-asset option pricing.

Douglas scheme: first-order splitting, no mixed derivatives.
Craig-Sneyd scheme: handles mixed derivatives (essential for Heston).

    price = heston_pde(spot=100, strike=100, rate=0.05, T=1.0,
                       v0=0.04, kappa=2, theta=0.04, xi=0.3, rho=-0.7)
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.black76 import OptionType
from pricebook.finite_difference import _thomas


def heston_pde(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: OptionType = OptionType.CALL,
    div_yield: float = 0.0,
    n_x: int = 80,
    n_v: int = 40,
    n_time: int = 100,
    x_range: float = 4.0,
    v_max: float = 1.0,
) -> float:
    """Heston PDE solver via Craig-Sneyd ADI.

    Solves in (x, v) space where x = log(S/K).

    Args:
        n_x: grid points in log-spot direction.
        n_v: grid points in variance direction.
        n_time: number of time steps.
        x_range: log-spot range in stdevs.
        v_max: maximum variance on the grid.
    """
    dt = T / n_time
    vol0 = math.sqrt(v0)

    # Grid
    x0 = math.log(spot / strike)
    width = x_range * vol0 * math.sqrt(T)
    x_min, x_max = x0 - max(width, 1.0), x0 + max(width, 1.0)
    dx = (x_max - x_min) / n_x
    dv = v_max / n_v

    x = np.linspace(x_min, x_max, n_x + 1)
    v = np.linspace(0, v_max, n_v + 1)

    S_grid = strike * np.exp(x)

    # Terminal payoff: V[i, j] at (x_i, v_j)
    V = np.zeros((n_x + 1, n_v + 1))
    for i in range(n_x + 1):
        if option_type == OptionType.CALL:
            V[i, :] = max(S_grid[i] - strike, 0.0)
        else:
            V[i, :] = max(strike - S_grid[i], 0.0)

    # Time stepping (backward from T to 0)
    for step in range(n_time):
        V_old = V.copy()

        # --- Craig-Sneyd ADI ---
        # Step 1: implicit in x-direction
        for j in range(1, n_v):
            vj = v[j]
            half_v = 0.5 * vj

            # PDE coefficients for x-direction
            ax = half_v / dx**2 - (rate - div_yield - half_v) / (2 * dx)
            bx = -vj / dx**2 - rate
            cx = half_v / dx**2 + (rate - div_yield - half_v) / (2 * dx)

            # RHS: explicit in x with old V
            n_int = n_x - 1
            rhs = np.zeros(n_int)
            for i in range(n_int):
                ii = i + 1
                rhs[i] = V_old[ii, j] + 0.5 * dt * (
                    ax * V_old[ii - 1, j] + bx * V_old[ii, j] + cx * V_old[ii + 1, j]
                )

                # Mixed derivative (Craig-Sneyd addition)
                if j > 0 and j < n_v and ii > 0 and ii < n_x:
                    mixed = rho * xi * vj / (4 * dx * dv) * (
                        V_old[ii + 1, j + 1] - V_old[ii - 1, j + 1]
                        - V_old[ii + 1, j - 1] + V_old[ii - 1, j - 1]
                    )
                    rhs[i] += 0.5 * dt * mixed

                # v-direction explicit contribution
                if j > 0 and j < n_v:
                    av = 0.5 * xi**2 * vj / dv**2 - kappa * (theta - vj) / (2 * dv)
                    bv = -xi**2 * vj / dv**2
                    cv = 0.5 * xi**2 * vj / dv**2 + kappa * (theta - vj) / (2 * dv)
                    rhs[i] += 0.5 * dt * (
                        av * V_old[ii, j - 1] + bv * V_old[ii, j] + cv * V_old[ii, j + 1]
                    )

            # Boundary terms
            rhs[0] += 0.5 * dt * ax * V_old[0, j]
            rhs[-1] += 0.5 * dt * cx * V_old[n_x, j]

            # Solve tridiagonal: (I - 0.5*dt*Ax) * V_half = rhs
            a_arr = np.full(n_int, -0.5 * dt * ax)
            b_arr = np.full(n_int, 1 - 0.5 * dt * bx)
            c_arr = np.full(n_int, -0.5 * dt * cx)

            V[1:n_x, j] = _thomas(a_arr, b_arr.copy(), c_arr, rhs)

        # Step 2: implicit in v-direction
        for i in range(1, n_x):
            V_half_x = V[i, :].copy()

            n_int_v = n_v - 1
            rhs_v = V_half_x[1:n_v].copy()

            # Solve tridiagonal in v
            vj_arr = v[1:n_v]
            av_arr = 0.5 * xi**2 * vj_arr / dv**2 - kappa * (theta - vj_arr) / (2 * dv)
            bv_arr = -xi**2 * vj_arr / dv**2
            cv_arr = 0.5 * xi**2 * vj_arr / dv**2 + kappa * (theta - vj_arr) / (2 * dv)

            a_v = -0.5 * dt * av_arr
            b_v = (1 - 0.5 * dt * bv_arr).copy()
            c_v = -0.5 * dt * cv_arr

            V[i, 1:n_v] = _thomas(a_v, b_v, c_v, rhs_v)

        # Boundary conditions
        if option_type == OptionType.CALL:
            V[0, :] = 0.0
            V[n_x, :] = S_grid[n_x] - strike * math.exp(-rate * (step + 1) * dt)
            V[:, 0] = np.maximum(S_grid - strike * math.exp(-rate * (step + 1) * dt), 0)
        else:
            V[n_x, :] = 0.0
            V[0, :] = strike * math.exp(-rate * (step + 1) * dt) - S_grid[0]
            V[:, 0] = np.maximum(strike * math.exp(-rate * (step + 1) * dt) - S_grid, 0)

        # v → ∞: linear extrapolation
        V[:, n_v] = 2 * V[:, n_v - 1] - V[:, n_v - 2]

    # Interpolate at (x0, v0)
    ix = int((x0 - x_min) / dx)
    ix = max(0, min(ix, n_x - 1))
    wx = (x0 - x[ix]) / dx

    iv = int(v0 / dv)
    iv = max(0, min(iv, n_v - 1))
    wv = (v0 - v[iv]) / dv

    price = (
        V[ix, iv] * (1 - wx) * (1 - wv)
        + V[ix + 1, iv] * wx * (1 - wv)
        + V[ix, iv + 1] * (1 - wx) * wv
        + V[ix + 1, iv + 1] * wx * wv
    )

    return max(float(price), 0.0)
