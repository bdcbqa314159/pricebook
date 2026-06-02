"""American option PDE in 2D: Heston American, two-asset American.

Extends ADI with penalty method for early exercise.

* :func:`heston_american_pde` — American under Heston via ADI + penalty.
* :func:`two_asset_american_pde` — American on two assets.

References:
    Forsyth & Vetzal, *Quadratic Convergence for Valuing American Options
    Using a Penalty Method*, SISC, 2002.
    Ikonen & Toivanen, *Operator Splitting Methods for American Option
    Pricing*, ANM, 2004.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.models.finite_difference import _thomas


def heston_american_pde(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    is_call: bool = False,
    div_yield: float = 0.0,
    n_x: int = 80,
    n_v: int = 40,
    n_time: int = 100,
    penalty_param: float = 1e6,
) -> dict:
    """American option under Heston via ADI + penalty method.

    Penalty method: adds large penalty for V < payoff:
    ∂V/∂t + ℒV + λ × max(g − V, 0) = 0
    where λ is large (penalty parameter).

    This converts the free boundary problem to a fixed-domain PDE
    with a nonlinear source term.

    Args:
        penalty_param: λ — larger = more accurate but stiffer.
    """
    dt = T / n_time
    vol0 = math.sqrt(v0)
    mu = rate - div_yield

    # Grid
    x0 = math.log(spot / strike)
    width = max(4 * vol0 * math.sqrt(T), 1.0)
    x = np.linspace(x0 - width, x0 + width, n_x)
    dx = x[1] - x[0]
    S = strike * np.exp(x)

    v_max = max(5 * v0, 0.5)
    v = np.linspace(0, v_max, n_v)
    dv = v[1] - v[0]

    # Terminal payoff
    V = np.zeros((n_x, n_v))
    payoff = np.zeros((n_x, n_v))
    for i in range(n_x):
        if is_call:
            payoff[i, :] = max(S[i] - strike, 0)
        else:
            payoff[i, :] = max(strike - S[i], 0)
    V[:] = payoff

    for step in range(n_time):
        V_old = V.copy()

        # Step 1: Implicit in x-direction (with penalty)
        for j in range(1, n_v - 1):
            vj = v[j]
            sigma = math.sqrt(max(vj, 1e-8))

            diff_x = 0.5 * vj
            conv_x = (mu - 0.5 * vj)
            a_x = diff_x / dx**2 - conv_x / (2 * dx)
            b_x = -2 * diff_x / dx**2 - rate
            c_x = diff_x / dx**2 + conv_x / (2 * dx)

            rhs = np.zeros(n_x - 2)
            lower = np.zeros(n_x - 2)
            diag = np.zeros(n_x - 2)
            upper = np.zeros(n_x - 2)

            for i in range(1, n_x - 1):
                # Mixed derivative (explicit)
                mixed = 0.0
                if 1 <= i < n_x - 1 and 1 <= j < n_v - 1:
                    mixed = rho * xi * vj * (
                        V_old[i+1, j+1] - V_old[i+1, j-1] - V_old[i-1, j+1] + V_old[i-1, j-1]
                    ) / (4 * dx * dv)

                rhs[i - 1] = V_old[i, j] + 0.5 * dt * (
                    a_x * V_old[i-1, j] + b_x * V_old[i, j] + c_x * V_old[i+1, j] + mixed
                )
                # Add penalty for American exercise
                rhs[i - 1] += dt * penalty_param * max(payoff[i, j] - V_old[i, j], 0)

                lower[i - 1] = -0.5 * dt * a_x
                diag[i - 1] = 1 + dt * diff_x / dx**2 + dt * penalty_param * (1 if V_old[i, j] < payoff[i, j] else 0)
                upper[i - 1] = -0.5 * dt * c_x

            V[1:-1, j] = _thomas(lower, diag, upper, rhs)

        # Step 2: Implicit in v-direction
        V_temp = V.copy()
        for i in range(1, n_x - 1):
            rhs = np.zeros(n_v - 2)
            lower = np.zeros(n_v - 2)
            diag_v = np.zeros(n_v - 2)
            upper = np.zeros(n_v - 2)

            for j in range(1, n_v - 1):
                vj = v[j]
                diff_v = 0.5 * xi**2 * vj
                conv_v = kappa * (theta - vj)
                a_v = diff_v / dv**2 - conv_v / (2 * dv)
                c_v = diff_v / dv**2 + conv_v / (2 * dv)

                rhs[j - 1] = V[i, j]
                lower[j - 1] = -0.5 * dt * a_v
                diag_v[j - 1] = 1 + dt * diff_v / dv**2
                upper[j - 1] = -0.5 * dt * c_v

            V_temp[i, 1:-1] = _thomas(lower, diag_v, upper, rhs)

        V = V_temp

        # Enforce American exercise constraint
        V = np.maximum(V, payoff)

        # Boundary conditions
        V[0, :] = payoff[0, :]
        V[-1, :] = payoff[-1, :]
        V[:, 0] = payoff[:, 0]
        V[:, -1] = V[:, -2]  # extrapolation at high vol

    # Interpolate at (spot, v0)
    i0 = int(np.searchsorted(S, spot))
    i0 = max(1, min(i0, n_x - 2))
    j0 = int(np.searchsorted(v, v0))
    j0 = max(1, min(j0, n_v - 2))

    # Bilinear interpolation
    wx = (spot - S[i0 - 1]) / (S[i0] - S[i0 - 1]) if S[i0] != S[i0 - 1] else 0
    wv = (v0 - v[j0 - 1]) / (v[j0] - v[j0 - 1]) if v[j0] != v[j0 - 1] else 0
    price = (1 - wx) * (1 - wv) * V[i0 - 1, j0 - 1] + wx * (1 - wv) * V[i0, j0 - 1] + \
            (1 - wx) * wv * V[i0 - 1, j0] + wx * wv * V[i0, j0]

    return {
        "price": float(max(price, 0)),
        "method": "heston_american_adi_penalty",
        "n_x": n_x, "n_v": n_v, "n_time": n_time,
    }
