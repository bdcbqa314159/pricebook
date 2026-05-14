"""PDE improvements: Hundsdorfer-Verwer ADI, 2D PSOR, operator splitting.

    from pricebook.numerical import hundsdorfer_verwer, psor_2d, operator_splitting
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PDEResult:
    """PDE solver result."""
    values: np.ndarray
    grid: np.ndarray
    method: str

    def to_dict(self) -> dict:
        return {"method": self.method, "grid_size": list(self.values.shape)}


def hundsdorfer_verwer(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    terminal: np.ndarray,
    a_x, a_y, a_xy,
    r: float,
    T: float,
    n_time: int = 100,
    theta: float = 0.5,
) -> PDEResult:
    """Hundsdorfer-Verwer ADI scheme for 2D PDE.

    Better stability than Craig-Sneyd for stiff problems.
    Second-order in time, handles mixed derivatives via explicit correction.

    PDE: ∂V/∂t + a_x ∂²V/∂x² + a_y ∂²V/∂y² + a_xy ∂²V/∂x∂y - rV = 0

    Args:
        x_grid, y_grid: spatial grids.
        terminal: terminal condition (n_x, n_y).
        a_x, a_y: diffusion coefficients (can be arrays on grid).
        a_xy: mixed derivative coefficient.
        r: discount rate.
        theta: implicit weight (0.5 = Crank-Nicolson like).
    """
    nx, ny = len(x_grid), len(y_grid)
    dx = np.diff(x_grid)
    dy = np.diff(y_grid)
    dt = T / n_time

    V = terminal.copy()

    for step in range(n_time):
        V_old = V.copy()

        # Explicit predictor (full 2D including mixed derivative)
        Y0 = V_old.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                hy = 0.5 * (dy[j-1] + dy[j])
                dxx = (V_old[i+1, j] - 2*V_old[i, j] + V_old[i-1, j]) / (hx ** 2)
                dyy = (V_old[i, j+1] - 2*V_old[i, j] + V_old[i, j-1]) / (hy ** 2)
                dxy = (V_old[i+1, j+1] - V_old[i+1, j-1] - V_old[i-1, j+1] + V_old[i-1, j-1]) / (4 * hx * hy)
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                a_xy_val = a_xy if isinstance(a_xy, (int, float)) else a_xy[i, j]
                Y0[i, j] = V_old[i, j] + dt * (a_x_val * dxx + a_y_val * dyy + a_xy_val * dxy - r * V_old[i, j])

        # Compute explicit L_x(V_old) and L_y(V_old) for correction RHS
        Lx_Vold = np.zeros((nx, ny))
        Ly_Vold = np.zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                hy = 0.5 * (dy[j-1] + dy[j])
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                Lx_Vold[i, j] = a_x_val * (V_old[i+1, j] - 2*V_old[i, j] + V_old[i-1, j]) / (hx**2)
                Ly_Vold[i, j] = a_y_val * (V_old[i, j+1] - 2*V_old[i, j] + V_old[i, j-1]) / (hy**2)

        # Stage 1: implicit correction in x — solve (I - theta*dt*L_x) Y1 = Y0 - theta*dt*L_x(V_old)
        Y1 = Y0.copy()
        for j in range(1, ny - 1):
            a_coeff = np.zeros(nx)
            b_coeff = np.ones(nx)
            c_coeff = np.zeros(nx)
            d_vec = Y0[:, j] - theta * dt * Lx_Vold[:, j]
            for i in range(1, nx - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                coeff = theta * dt * a_x_val / (hx ** 2)
                a_coeff[i] = -coeff
                b_coeff[i] = 1 + 2 * coeff
                c_coeff[i] = -coeff
            Y1[:, j] = _thomas(a_coeff, b_coeff, c_coeff, d_vec)

        # Stage 2: implicit correction in y — solve (I - theta*dt*L_y) Y2 = Y1 - theta*dt*L_y(V_old)
        Y2 = Y1.copy()
        for i in range(1, nx - 1):
            a_coeff = np.zeros(ny)
            b_coeff = np.ones(ny)
            c_coeff = np.zeros(ny)
            d_vec = Y1[i, :] - theta * dt * Ly_Vold[i, :]
            for j in range(1, ny - 1):
                hy = 0.5 * (dy[j-1] + dy[j])
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                coeff = theta * dt * a_y_val / (hy ** 2)
                a_coeff[j] = -coeff
                b_coeff[j] = 1 + 2 * coeff
                c_coeff[j] = -coeff
            Y2[i, :] = _thomas(a_coeff, b_coeff, c_coeff, d_vec)

        # Stage 3 (second corrector): repeat with Y2 as predictor for full HV
        # Compute explicit operator on Y2
        F_Y2 = np.zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                hy = 0.5 * (dy[j-1] + dy[j])
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                a_xy_val = a_xy if isinstance(a_xy, (int, float)) else a_xy[i, j]
                dxx2 = (Y2[i+1, j] - 2*Y2[i, j] + Y2[i-1, j]) / (hx**2)
                dyy2 = (Y2[i, j+1] - 2*Y2[i, j] + Y2[i, j-1]) / (hy**2)
                dxy2 = (Y2[i+1, j+1] - Y2[i+1, j-1] - Y2[i-1, j+1] + Y2[i-1, j-1]) / (4*hx*hy)
                F_Y2[i, j] = a_x_val * dxx2 + a_y_val * dyy2 + a_xy_val * dxy2 - r * Y2[i, j]

        # Y0_hat = Y2 + 0.5*dt*(F(Y2) - F(V_old))  [second predictor]
        F_Vold = (Y0 - V_old) / dt  # recover F(V_old) from predictor
        Y0_hat = Y2 + 0.5 * dt * (F_Y2 - F_Vold)

        # Stages 4-5: implicit corrections on Y0_hat (same structure as 1-2)
        Lx_Y2 = np.zeros((nx, ny))
        Ly_Y2 = np.zeros((nx, ny))
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                hy = 0.5 * (dy[j-1] + dy[j])
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                Lx_Y2[i, j] = a_x_val * (Y2[i+1, j] - 2*Y2[i, j] + Y2[i-1, j]) / (hx**2)
                Ly_Y2[i, j] = a_y_val * (Y2[i, j+1] - 2*Y2[i, j] + Y2[i, j-1]) / (hy**2)

        Y3 = Y0_hat.copy()
        for j in range(1, ny - 1):
            a_coeff = np.zeros(nx)
            b_coeff = np.ones(nx)
            c_coeff = np.zeros(nx)
            d_vec = Y0_hat[:, j] - theta * dt * Lx_Y2[:, j]
            for i in range(1, nx - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                coeff = theta * dt * a_x_val / (hx ** 2)
                a_coeff[i] = -coeff
                b_coeff[i] = 1 + 2 * coeff
                c_coeff[i] = -coeff
            Y3[:, j] = _thomas(a_coeff, b_coeff, c_coeff, d_vec)

        V_new = Y3.copy()
        for i in range(1, nx - 1):
            a_coeff = np.zeros(ny)
            b_coeff = np.ones(ny)
            c_coeff = np.zeros(ny)
            d_vec = Y3[i, :] - theta * dt * Ly_Y2[i, :]
            for j in range(1, ny - 1):
                hy = 0.5 * (dy[j-1] + dy[j])
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                coeff = theta * dt * a_y_val / (hy ** 2)
                a_coeff[j] = -coeff
                b_coeff[j] = 1 + 2 * coeff
                c_coeff[j] = -coeff
            V_new[i, :] = _thomas(a_coeff, b_coeff, c_coeff, d_vec)

        V = V_new

    return PDEResult(V, x_grid, "hundsdorfer_verwer")


def psor_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    terminal: np.ndarray,
    exercise: np.ndarray,
    a_x, a_y,
    r: float,
    T: float,
    n_time: int = 100,
    omega: float = 1.3,
    max_sor_iter: int = 100,
    tol: float = 1e-8,
) -> PDEResult:
    """Projected SOR for 2D American options.

    At each time step, solves the linear complementarity problem:
    V >= exercise AND (LV - rV - ∂V/∂t) * (V - exercise) = 0.

    Args:
        exercise: early exercise values (n_x, n_y).
        omega: SOR relaxation parameter (1 < omega < 2 for over-relaxation).
    """
    nx, ny = len(x_grid), len(y_grid)
    dt = T / n_time
    V = terminal.copy()

    for step in range(n_time):
        V_old = V.copy()

        for sor_iter in range(max_sor_iter):
            max_change = 0.0
            for i in range(1, nx - 1):
                for j in range(1, ny - 1):
                    hx = x_grid[i+1] - x_grid[i-1]
                    hy = y_grid[j+1] - y_grid[j-1]
                    a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                    a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]

                    diag = 1 + dt * (2 * a_x_val / (hx/2)**2 + 2 * a_y_val / (hy/2)**2 + r)
                    rhs = V_old[i, j]
                    rhs += dt * a_x_val * (V[i+1, j] + V[i-1, j]) / (hx/2)**2
                    rhs += dt * a_y_val * (V[i, j+1] + V[i, j-1]) / (hy/2)**2

                    v_new = rhs / diag
                    # SOR relaxation first
                    v_new = V[i, j] + omega * (v_new - V[i, j])
                    # Then project: enforce V >= exercise (must come after SOR)
                    v_new = max(v_new, exercise[i, j])

                    change = abs(v_new - V[i, j])
                    max_change = max(max_change, change)
                    V[i, j] = v_new

            if max_change < tol:
                break

    return PDEResult(V, x_grid, "psor_2d")


def operator_splitting(
    V: np.ndarray,
    operators: list,
    dt: float,
    n_steps: int,
    method: str = "strang",
) -> np.ndarray:
    """Operator splitting for multi-factor PDE.

    Splits L = L1 + L2 + ... and applies each operator sequentially.

    Methods:
        'lie': L1(dt) L2(dt) — first-order (Lie-Trotter)
        'strang': L1(dt/2) L2(dt) L1(dt/2) — second-order (Strang)

    Args:
        V: initial values.
        operators: list of callables op(V, dt) → V_new.
    """
    for step in range(n_steps):
        if method == "lie":
            for op in operators:
                V = op(V, dt)
        elif method == "strang":
            # Symmetric Strang: L1(dt/2) L2(dt/2) ... Ln(dt) ... L2(dt/2) L1(dt/2)
            n_ops = len(operators)
            if n_ops == 1:
                V = operators[0](V, dt)
            else:
                # Forward half-steps
                for k in range(n_ops - 1):
                    V = operators[k](V, dt / 2)
                # Full step on last operator
                V = operators[-1](V, dt)
                # Backward half-steps
                for k in range(n_ops - 2, -1, -1):
                    V = operators[k](V, dt / 2)
        else:
            raise ValueError(f"unknown splitting method: {method!r}")

    return V


def _thomas(a, b, c, d):
    """Thomas algorithm for tridiagonal systems."""
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i-1]
        if abs(denom) < 1e-20:
            denom = 1e-20
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i-1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x
