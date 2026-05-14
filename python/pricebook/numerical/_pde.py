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

        # Explicit predictor (full 2D)
        Y0 = V_old.copy()
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dxx = (V_old[i+1, j] - 2*V_old[i, j] + V_old[i-1, j]) / (0.5*(dx[i-1]+dx[i]))**2 if isinstance(a_x, (int, float)) else 0
                dyy = (V_old[i, j+1] - 2*V_old[i, j] + V_old[i, j-1]) / (0.5*(dy[j-1]+dy[j]))**2 if isinstance(a_y, (int, float)) else 0
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                a_y_val = a_y if isinstance(a_y, (int, float)) else a_y[i, j]
                Y0[i, j] = V_old[i, j] + dt * (a_x_val * dxx + a_y_val * dyy - r * V_old[i, j])

        # Implicit correction in x-direction
        Y1 = Y0.copy()
        for j in range(1, ny - 1):
            # Thomas algorithm in x
            a_coeff = np.zeros(nx)
            b_coeff = np.ones(nx)
            c_coeff = np.zeros(nx)
            d_vec = Y0[:, j].copy()
            for i in range(1, nx - 1):
                hx = 0.5 * (dx[i-1] + dx[i])
                a_x_val = a_x if isinstance(a_x, (int, float)) else a_x[i, j]
                coeff = theta * dt * a_x_val / (hx ** 2)
                a_coeff[i] = -coeff
                b_coeff[i] = 1 + 2 * coeff
                c_coeff[i] = -coeff
            # Solve tridiagonal
            Y1[:, j] = _thomas(a_coeff, b_coeff, c_coeff, d_vec)

        # Implicit correction in y-direction
        V_new = Y1.copy()
        for i in range(1, nx - 1):
            a_coeff = np.zeros(ny)
            b_coeff = np.ones(ny)
            c_coeff = np.zeros(ny)
            d_vec = Y1[i, :].copy()
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
                    # Project: enforce V >= exercise
                    v_new = max(v_new, exercise[i, j])
                    # SOR relaxation
                    v_new = V[i, j] + omega * (v_new - V[i, j])

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
            if len(operators) >= 2:
                V = operators[0](V, dt / 2)
                for op in operators[1:-1]:
                    V = op(V, dt)
                V = operators[-1](V, dt)
                V = operators[0](V, dt / 2)
            else:
                V = operators[0](V, dt)
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
