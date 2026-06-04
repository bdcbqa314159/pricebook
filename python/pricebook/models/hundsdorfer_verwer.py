"""Hundsdorfer-Verwer ADI scheme for 2D parabolic PDEs.

More stable than Craig-Sneyd for strong mixed derivatives.
Uses a modified θ-scheme with explicit mixed derivative correction.

* :func:`hv_adi_heston` — Heston PDE via HV ADI.
* :func:`hv_adi_two_asset` — two-asset option via HV ADI.

References:
    Hundsdorfer & Verwer, *Numerical Solution of Time-Dependent
    Advection-Diffusion-Reaction Equations*, Springer, 2003, Ch. IV.5.
    in 't Hout & Foulon, *ADI Finite Difference Schemes for Option
    Pricing in the Heston Model with Correlation*, IJNAM, 2010.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.models.finite_difference import _thomas


def hv_adi_heston(
    spot: float,
    strike: float,
    rate: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    is_call: bool = True,
    div_yield: float = 0.0,
    n_x: int = 80,
    n_v: int = 40,
    n_time: int = 100,
    hv_theta: float = 0.5,
) -> float:
    """Heston PDE via Hundsdorfer-Verwer ADI.

    HV scheme for F_t = (A₀ + A₁ + A₂)F:

    Step 0: Y₀ = Fⁿ + dt(A₀ + A₁ + A₂)Fⁿ           (explicit predictor)
    Step 1: Y₁ = Y₀ + θ dt (A₁ Y₁ − A₁ Fⁿ)          (implicit in x)
    Step 2: Y₂ = Y₁ + θ dt (A₂ Y₂ − A₂ Fⁿ)          (implicit in v)
    Step 3: Ỹ₀ = Y₀ + ½ dt (A₀ Y₂ − A₀ Fⁿ)          (mixed deriv correction)
    Step 4: Ỹ₁ = Ỹ₀ + θ dt (A₁ Ỹ₁ − A₁ Y₁)         (implicit in x again)
    Step 5: F^{n+1} = Ỹ₁ + θ dt (A₂ F^{n+1} − A₂ Y₂) (implicit in v again)

    This double-pass gives O(dt²) accuracy with strong stability.

    Args:
        hv_theta: θ parameter (0.5 = standard HV, recommended).
    """
    dt = T / n_time
    mu = rate - div_yield
    vol0 = math.sqrt(v0)

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
    for i in range(n_x):
        if is_call:
            V[i, :] = max(S[i] - strike, 0)
        else:
            V[i, :] = max(strike - S[i], 0)

    th = hv_theta

    for step in range(n_time):
        F_n = V.copy()

        # A₀ Fⁿ: mixed derivative (explicit)
        A0_F = np.zeros_like(V)
        for i in range(1, n_x - 1):
            for j in range(1, n_v - 1):
                A0_F[i, j] = rho * xi * v[j] * (
                    F_n[i+1, j+1] - F_n[i+1, j-1] - F_n[i-1, j+1] + F_n[i-1, j-1]
                ) / (4 * dx * dv)

        # A₁ Fⁿ: x-direction operator
        A1_F = np.zeros_like(V)
        for j in range(1, n_v - 1):
            vj = v[j]
            diff_x = 0.5 * vj
            conv_x = mu - 0.5 * vj
            for i in range(1, n_x - 1):
                A1_F[i, j] = (diff_x / dx**2) * (F_n[i+1, j] - 2*F_n[i, j] + F_n[i-1, j]) \
                    + (conv_x / (2*dx)) * (F_n[i+1, j] - F_n[i-1, j]) \
                    - 0.5 * rate * F_n[i, j]

        # A₂ Fⁿ: v-direction operator
        A2_F = np.zeros_like(V)
        for i in range(1, n_x - 1):
            for j in range(1, n_v - 1):
                vj = v[j]
                diff_v = 0.5 * xi**2 * vj
                conv_v = kappa * (theta - vj)
                A2_F[i, j] = (diff_v / dv**2) * (F_n[i, j+1] - 2*F_n[i, j] + F_n[i, j-1]) \
                    + (conv_v / (2*dv)) * (F_n[i, j+1] - F_n[i, j-1]) \
                    - 0.5 * rate * F_n[i, j]

        # Step 0: explicit predictor
        Y0 = F_n + dt * (A0_F + A1_F + A2_F)

        # Step 1: implicit in x
        Y1 = Y0.copy()
        for j in range(1, n_v - 1):
            vj = v[j]
            diff_x = 0.5 * vj
            conv_x = mu - 0.5 * vj
            a_x = diff_x / dx**2 - conv_x / (2 * dx)
            b_x = -2 * diff_x / dx**2 - 0.5 * rate
            c_x = diff_x / dx**2 + conv_x / (2 * dx)

            rhs = Y0[1:-1, j] - th * dt * A1_F[1:-1, j]
            lower = np.full(n_x - 2, -th * dt * a_x)
            diag = np.full(n_x - 2, 1 - th * dt * b_x)
            upper = np.full(n_x - 2, -th * dt * c_x)
            Y1[1:-1, j] = _thomas(lower, diag, upper, rhs)

        # Step 2: implicit in v
        Y2 = Y1.copy()
        for i in range(1, n_x - 1):
            rhs_v = np.zeros(n_v - 2)
            lower_v = np.zeros(n_v - 2)
            diag_v = np.zeros(n_v - 2)
            upper_v = np.zeros(n_v - 2)
            for j in range(1, n_v - 1):
                vj = v[j]
                diff_v = 0.5 * xi**2 * vj
                conv_v = kappa * (theta - vj)
                a_v = diff_v / dv**2 - conv_v / (2 * dv)
                b_v = -2 * diff_v / dv**2 - 0.5 * rate
                c_v = diff_v / dv**2 + conv_v / (2 * dv)
                rhs_v[j - 1] = Y1[i, j] - th * dt * A2_F[i, j]
                lower_v[j - 1] = -th * dt * a_v
                diag_v[j - 1] = 1 - th * dt * b_v
                upper_v[j - 1] = -th * dt * c_v
            Y2[i, 1:-1] = _thomas(lower_v, diag_v, upper_v, rhs_v)

        # Step 3: mixed derivative correction
        A0_Y2 = np.zeros_like(V)
        for i in range(1, n_x - 1):
            for j in range(1, n_v - 1):
                A0_Y2[i, j] = rho * xi * v[j] * (
                    Y2[i+1, j+1] - Y2[i+1, j-1] - Y2[i-1, j+1] + Y2[i-1, j-1]
                ) / (4 * dx * dv)
        Y0_tilde = Y0 + 0.5 * dt * (A0_Y2 - A0_F)

        # Step 4: implicit in x (second pass)
        A1_Y1 = np.zeros_like(V)
        for j in range(1, n_v - 1):
            vj = v[j]
            diff_x = 0.5 * vj
            conv_x = mu - 0.5 * vj
            for i in range(1, n_x - 1):
                A1_Y1[i, j] = (diff_x / dx**2) * (Y1[i+1, j] - 2*Y1[i, j] + Y1[i-1, j]) \
                    + (conv_x / (2*dx)) * (Y1[i+1, j] - Y1[i-1, j]) \
                    - 0.5 * rate * Y1[i, j]

        Y1_tilde = Y0_tilde.copy()
        for j in range(1, n_v - 1):
            vj = v[j]
            diff_x = 0.5 * vj
            conv_x = mu - 0.5 * vj
            a_x = diff_x / dx**2 - conv_x / (2 * dx)
            b_x = -2 * diff_x / dx**2 - 0.5 * rate
            c_x = diff_x / dx**2 + conv_x / (2 * dx)
            rhs = Y0_tilde[1:-1, j] - th * dt * A1_Y1[1:-1, j]
            lower = np.full(n_x - 2, -th * dt * a_x)
            diag = np.full(n_x - 2, 1 - th * dt * b_x)
            upper = np.full(n_x - 2, -th * dt * c_x)
            Y1_tilde[1:-1, j] = _thomas(lower, diag, upper, rhs)

        # Step 5: implicit in v (second pass)
        A2_Y2 = np.zeros_like(V)
        for i in range(1, n_x - 1):
            for j in range(1, n_v - 1):
                vj = v[j]
                diff_v = 0.5 * xi**2 * vj
                conv_v = kappa * (theta - vj)
                A2_Y2[i, j] = (diff_v / dv**2) * (Y2[i, j+1] - 2*Y2[i, j] + Y2[i, j-1]) \
                    + (conv_v / (2*dv)) * (Y2[i, j+1] - Y2[i, j-1]) \
                    - 0.5 * rate * Y2[i, j]

        V_new = Y1_tilde.copy()
        for i in range(1, n_x - 1):
            rhs_v = np.zeros(n_v - 2)
            lower_v = np.zeros(n_v - 2)
            diag_v = np.zeros(n_v - 2)
            upper_v = np.zeros(n_v - 2)
            for j in range(1, n_v - 1):
                vj = v[j]
                diff_v = 0.5 * xi**2 * vj
                conv_v = kappa * (theta - vj)
                a_v = diff_v / dv**2 - conv_v / (2 * dv)
                b_v = -2 * diff_v / dv**2 - 0.5 * rate
                c_v = diff_v / dv**2 + conv_v / (2 * dv)
                rhs_v[j - 1] = Y1_tilde[i, j] - th * dt * A2_Y2[i, j]
                lower_v[j - 1] = -th * dt * a_v
                diag_v[j - 1] = 1 - th * dt * b_v
                upper_v[j - 1] = -th * dt * c_v
            V_new[i, 1:-1] = _thomas(lower_v, diag_v, upper_v, rhs_v)

        # Boundary conditions
        V_new[0, :] = 0 if is_call else strike - S[0]
        V_new[-1, :] = S[-1] - strike if is_call else 0
        V_new[:, 0] = V_new[:, 1]
        V_new[:, -1] = V_new[:, -2]

        V = V_new

    # Interpolate
    i0 = max(1, min(int(np.searchsorted(S, spot)), n_x - 2))
    j0 = max(1, min(int(np.searchsorted(v, v0)), n_v - 2))
    wx = (spot - S[i0 - 1]) / (S[i0] - S[i0 - 1]) if S[i0] != S[i0 - 1] else 0
    wv = (v0 - v[j0 - 1]) / (v[j0] - v[j0 - 1]) if v[j0] != v[j0 - 1] else 0
    price = (1 - wx) * (1 - wv) * V[i0 - 1, j0 - 1] + wx * (1 - wv) * V[i0, j0 - 1] + \
            (1 - wx) * wv * V[i0 - 1, j0] + wx * wv * V[i0, j0]

    return max(float(price), 0)
