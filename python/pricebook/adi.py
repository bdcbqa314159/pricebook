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


# ---------------------------------------------------------------------------
# Two-asset options via Craig-Sneyd ADI
# ---------------------------------------------------------------------------


def two_asset_option(
    spot1: float,
    spot2: float,
    strike: float,
    rate: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    payoff_type: str = "spread",
    weights: tuple[float, float] = (1.0, 1.0),
    div_yield1: float = 0.0,
    div_yield2: float = 0.0,
    n_x: int = 60,
    n_y: int = 60,
    n_time: int = 80,
    x_range: float = 3.5,
) -> float:
    """Two-asset option via Craig-Sneyd ADI on 2D GBM PDE.

    Payoff types:
        "spread": max(S1 - S2 - K, 0)
        "basket": max(w1*S1 + w2*S2 - K, 0)
        "best_of": max(max(S1, S2) - K, 0)

    Both assets follow GBM with correlation rho.
    """
    dt = T / n_time
    w1, w2 = weights

    # Grids in log-spot space: x = ln(S1), y = ln(S2)
    x0 = math.log(spot1)
    y0 = math.log(spot2)

    wx = x_range * vol1 * math.sqrt(T)
    wy = x_range * vol2 * math.sqrt(T)

    x_min, x_max = x0 - max(wx, 0.5), x0 + max(wx, 0.5)
    y_min, y_max = y0 - max(wy, 0.5), y0 + max(wy, 0.5)

    dx = (x_max - x_min) / n_x
    dy = (y_max - y_min) / n_y

    x = np.linspace(x_min, x_max, n_x + 1)
    y = np.linspace(y_min, y_max, n_y + 1)

    S1_grid = np.exp(x)
    S2_grid = np.exp(y)

    # Terminal payoff
    V = np.zeros((n_x + 1, n_y + 1))
    for i in range(n_x + 1):
        for j in range(n_y + 1):
            s1, s2 = S1_grid[i], S2_grid[j]
            if payoff_type == "spread":
                V[i, j] = max(s1 - s2 - strike, 0.0)
            elif payoff_type == "basket":
                V[i, j] = max(w1 * s1 + w2 * s2 - strike, 0.0)
            elif payoff_type == "best_of":
                V[i, j] = max(max(s1, s2) - strike, 0.0)

    mu1 = rate - div_yield1 - 0.5 * vol1**2
    mu2 = rate - div_yield2 - 0.5 * vol2**2
    sig1_sq = vol1**2
    sig2_sq = vol2**2

    for step in range(n_time):
        V_old = V.copy()
        tau = (step + 1) * dt

        # Step 1: implicit in x-direction
        for j in range(1, n_y):
            ax = 0.5 * sig1_sq / dx**2 - mu1 / (2 * dx)
            bx = -sig1_sq / dx**2 - rate
            cx = 0.5 * sig1_sq / dx**2 + mu1 / (2 * dx)

            n_int = n_x - 1
            rhs = np.zeros(n_int)
            for i in range(n_int):
                ii = i + 1
                # x-direction explicit
                rhs[i] = V_old[ii, j] + 0.5 * dt * (
                    ax * V_old[ii - 1, j] + bx * V_old[ii, j] + cx * V_old[ii + 1, j]
                )

                # y-direction explicit
                ay = 0.5 * sig2_sq / dy**2 - mu2 / (2 * dy)
                by = -sig2_sq / dy**2
                cy = 0.5 * sig2_sq / dy**2 + mu2 / (2 * dy)
                if j > 0 and j < n_y:
                    rhs[i] += 0.5 * dt * (
                        ay * V_old[ii, j - 1] + by * V_old[ii, j] + cy * V_old[ii, j + 1]
                    )

                # Mixed derivative (Craig-Sneyd)
                if ii > 0 and ii < n_x and j > 0 and j < n_y:
                    mixed = rho * vol1 * vol2 / (4 * dx * dy) * (
                        V_old[ii + 1, j + 1] - V_old[ii - 1, j + 1]
                        - V_old[ii + 1, j - 1] + V_old[ii - 1, j - 1]
                    )
                    rhs[i] += 0.5 * dt * mixed

            rhs[0] += 0.5 * dt * ax * V_old[0, j]
            rhs[-1] += 0.5 * dt * cx * V_old[n_x, j]

            a_arr = np.full(n_int, -0.5 * dt * ax)
            b_arr = np.full(n_int, 1 - 0.5 * dt * bx)
            c_arr = np.full(n_int, -0.5 * dt * cx)

            V[1:n_x, j] = _thomas(a_arr, b_arr.copy(), c_arr, rhs)

        # Step 2: implicit in y-direction
        for i in range(1, n_x):
            ay = 0.5 * sig2_sq / dy**2 - mu2 / (2 * dy)
            by = -sig2_sq / dy**2
            cy = 0.5 * sig2_sq / dy**2 + mu2 / (2 * dy)

            n_int_y = n_y - 1
            rhs_y = V[i, 1:n_y].copy()

            a_y = np.full(n_int_y, -0.5 * dt * ay)
            b_y = np.full(n_int_y, 1 - 0.5 * dt * by)
            c_y = np.full(n_int_y, -0.5 * dt * cy)

            V[i, 1:n_y] = _thomas(a_y, b_y.copy(), c_y, rhs_y)

        # Boundary: linear extrapolation at edges
        V[0, :] = 2 * V[1, :] - V[2, :]
        V[n_x, :] = 2 * V[n_x - 1, :] - V[n_x - 2, :]
        V[:, 0] = 2 * V[:, 1] - V[:, 2]
        V[:, n_y] = 2 * V[:, n_y - 1] - V[:, n_y - 2]
        V = np.maximum(V, 0.0)

    # Interpolate at (x0, y0)
    ix = int((x0 - x_min) / dx)
    ix = max(0, min(ix, n_x - 1))
    wx_interp = (x0 - x[ix]) / dx

    iy = int((y0 - y_min) / dy)
    iy = max(0, min(iy, n_y - 1))
    wy_interp = (y0 - y[iy]) / dy

    price = (
        V[ix, iy] * (1 - wx_interp) * (1 - wy_interp)
        + V[ix + 1, iy] * wx_interp * (1 - wy_interp)
        + V[ix, iy + 1] * (1 - wx_interp) * wy_interp
        + V[ix + 1, iy + 1] * wx_interp * wy_interp
    )

    return max(float(price), 0.0)
