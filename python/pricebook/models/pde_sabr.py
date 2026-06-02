"""SABR PDE via ADI: 2D in (F, σ) space.

* :func:`sabr_pde` — SABR option pricing via 2D ADI.

References:
    Hagan et al., *Managing Smile Risk*, Wilmott, 2002.
    Le Floc'h, *Finite Difference Techniques for Arbitrage-Free SABR*, 2014.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.models.finite_difference import _thomas


def sabr_pde(
    forward: float,
    strike: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    is_call: bool = True,
    n_f: int = 80,
    n_a: int = 40,
    n_time: int = 100,
) -> float:
    """SABR option pricing via 2D ADI (Craig-Sneyd).

    dF = σ F^β dW₁
    dσ = ν σ dW₂
    dW₁ dW₂ = ρ dt

    Solved in (F, σ) space with absorbing boundary at F=0.

    Args:
        forward: initial forward rate.
        alpha: initial vol (σ₀).
        beta: CEV exponent.
        rho: correlation.
        nu: vol-of-vol.
        n_f: grid points in forward direction.
        n_a: grid points in alpha (vol) direction.
        n_time: time steps.
    """
    dt = T / n_time

    # Forward grid
    f_max = forward * 5.0
    f_min = max(forward * 0.01, 1e-6) if beta < 1 else forward * 0.1
    f = np.linspace(f_min, f_max, n_f)
    df = f[1] - f[0]

    # Vol grid
    a_min = alpha * 0.1
    a_max = alpha * 4.0
    a_grid = np.linspace(a_min, a_max, n_a)
    da = a_grid[1] - a_grid[0]

    # Terminal payoff
    V = np.zeros((n_f, n_a))
    for i in range(n_f):
        if is_call:
            V[i, :] = max(f[i] - strike, 0)
        else:
            V[i, :] = max(strike - f[i], 0)

    for step in range(n_time):
        V_new = V.copy()

        # Mixed derivative term (explicit)
        mixed = np.zeros((n_f, n_a))
        for i in range(1, n_f - 1):
            for j in range(1, n_a - 1):
                mixed[i, j] = rho * nu * a_grid[j] * f[i]**beta * (
                    V[i+1, j+1] - V[i+1, j-1] - V[i-1, j+1] + V[i-1, j-1]
                ) / (4 * df * da)

        # Step 1: Implicit in F-direction
        for j in range(1, n_a - 1):
            sig = a_grid[j]
            # Coefficients for F-direction: 0.5 σ² F^{2β} ∂²V/∂F²
            rhs = np.zeros(n_f - 2)
            lower_f = np.zeros(n_f - 2)
            diag_f = np.zeros(n_f - 2)
            upper_f = np.zeros(n_f - 2)

            for i in range(1, n_f - 1):
                diff_f = 0.5 * sig**2 * f[i]**(2 * beta)
                a_f = diff_f / df**2
                b_f = -2 * diff_f / df**2

                # Explicit RHS (half step)
                rhs[i - 1] = V[i, j] + 0.5 * dt * (
                    a_f * V[i-1, j] + b_f * V[i, j] + a_f * V[i+1, j]
                    + mixed[i, j]
                )

                # Implicit LHS
                lower_f[i - 1] = -0.5 * dt * a_f
                diag_f[i - 1] = 1 + dt * diff_f / df**2
                upper_f[i - 1] = -0.5 * dt * a_f

            V_new[1:-1, j] = _thomas(lower_f, diag_f, upper_f, rhs)

        # Step 2: Implicit in σ-direction
        V_temp = V_new.copy()
        for i in range(1, n_f - 1):
            rhs = np.zeros(n_a - 2)
            lower_a = np.zeros(n_a - 2)
            diag_a = np.zeros(n_a - 2)
            upper_a = np.zeros(n_a - 2)

            for j in range(1, n_a - 1):
                diff_a = 0.5 * nu**2 * a_grid[j]**2
                a_a = diff_a / da**2

                rhs[j - 1] = V_new[i, j]  # from step 1

                lower_a[j - 1] = -0.5 * dt * a_a
                diag_a[j - 1] = 1 + dt * diff_a / da**2
                upper_a[j - 1] = -0.5 * dt * a_a

            V_temp[i, 1:-1] = _thomas(lower_a, diag_a, upper_a, rhs)

        V = V_temp

        # Boundary conditions
        V[0, :] = 0  # F → 0: absorbing
        for j in range(n_a):
            if is_call:
                V[-1, j] = f[-1] - strike  # deep ITM
            else:
                V[-1, j] = 0
        # σ boundaries: extrapolation
        V[:, 0] = V[:, 1]
        V[:, -1] = V[:, -2]

    # Interpolate at (forward, alpha)
    price = float(np.interp(forward, f, V[:, np.searchsorted(a_grid, alpha)]))
    return max(price, 0)
