"""PDE solvers — unified framework with method selection, grids, Greeks.

Supports 1D and 2D parabolic PDEs (Black-Scholes, Heston) with
multiple time-stepping schemes, grid types, boundary conditions,
and automatic Greeks extraction.

    from pricebook.numerical._pde import (
        PDESolver, PDEMethod, PDEGrid, PDEResult,
        solve_bs_pde, solve_heston_pde,
    )

    # 1D Black-Scholes
    result = solve_bs_pde(spot=100, strike=100, T=1, vol=0.25, rate=0.04)
    print(result.price, result.delta, result.gamma)

References:
    Duffy (2006). Finite Difference Methods in Financial Engineering.
    Hundsdorfer & Verwer (2003). Numerical Solution of Time-Dependent
    Advection-Diffusion-Reaction Equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class PDEMethod(Enum):
    """Time-stepping method for PDE."""
    EXPLICIT = "explicit"            # Forward Euler, O(Δt), CFL restricted
    IMPLICIT = "implicit"            # Backward Euler, O(Δt), unconditional
    CRANK_NICOLSON = "crank_nicolson"  # θ=0.5, O(Δt²), unconditional
    RANNACHER = "rannacher"          # 2 implicit steps then CN (smooths payoff kink)
    CRAIG_SNEYD = "craig_sneyd"      # ADI for 2D, handles mixed derivatives
    HUNDSDORFER_VERWER = "hundsdorfer_verwer"  # ADI 2D, more stable than CS
    METHOD_OF_LINES = "method_of_lines"  # FD in space → ODE in time


class BoundaryCondition(Enum):
    """Boundary condition type."""
    DIRICHLET = "dirichlet"      # fixed value
    NEUMANN = "neumann"          # fixed derivative (∂V/∂x = const)
    LINEAR = "linear"            # linear extrapolation (V ≈ a + bx)
    FREE = "free"                # free boundary (American exercise)


class GridType(Enum):
    """Spatial grid type."""
    UNIFORM = "uniform"          # equally spaced
    LOG = "log"                  # uniform in log-space (standard for equity)
    SINH = "sinh"                # concentrated near strike (Tavella-Randall)
    CHEBYSHEV = "chebyshev"      # Chebyshev-Gauss-Lobatto nodes


@dataclass
class PDEResult:
    """PDE solution with Greeks."""
    values: np.ndarray           # solution on grid (n_x,) or (n_x, n_y)
    grid: np.ndarray             # spatial grid (n_x,) or (n_x, n_y)
    price: float                 # interpolated price at spot
    delta: float                 # ∂V/∂S
    gamma: float                 # ∂²V/∂S²
    theta: float                 # -∂V/∂t (from one time step back)
    vega: float | None           # ∂V/∂σ (if computed)
    method: str
    n_space: int
    n_time: int
    grid_type: str

    def to_dict(self) -> dict:
        return {
            "price": self.price, "delta": self.delta, "gamma": self.gamma,
            "theta": self.theta, "method": self.method,
            "n_space": self.n_space, "n_time": self.n_time,
        }


# ═══════════════════════════════════════════════════════════════
# Grid builders
# ═══════════════════════════════════════════════════════════════


def build_grid(
    s_min: float,
    s_max: float,
    n_points: int,
    grid_type: GridType = GridType.UNIFORM,
    concentration_point: float | None = None,
) -> np.ndarray:
    """Build a spatial grid.

    Args:
        s_min, s_max: grid boundaries.
        n_points: number of grid points.
        grid_type: type of grid spacing.
        concentration_point: for SINH grid, concentrate around this value.
    """
    if grid_type == GridType.UNIFORM:
        return np.linspace(s_min, s_max, n_points)

    elif grid_type == GridType.LOG:
        return np.exp(np.linspace(np.log(max(s_min, 1e-10)), np.log(s_max), n_points))

    elif grid_type == GridType.SINH:
        # Tavella-Randall sinh grid: concentrated near concentration_point
        c = concentration_point or (s_min + s_max) / 2
        alpha = 0.5 * (s_max - s_min) / np.sinh(3)  # scale factor
        xi = np.linspace(-3, 3, n_points)
        return c + alpha * np.sinh(xi)

    elif grid_type == GridType.CHEBYSHEV:
        from pricebook.numerical._spectral import chebyshev_nodes
        return chebyshev_nodes(n_points - 1, s_min, s_max)

    return np.linspace(s_min, s_max, n_points)


# ═══════════════════════════════════════════════════════════════
# Greeks extraction
# ═══════════════════════════════════════════════════════════════


def extract_greeks(
    grid: np.ndarray,
    values: np.ndarray,
    spot: float,
    values_prev: np.ndarray | None = None,
    dt: float = 0.0,
) -> dict:
    """Extract Greeks from PDE grid solution via finite differences.

    Args:
        grid: (N,) spatial grid (S values).
        values: (N,) option values at current time.
        spot: current spot price (interpolation point).
        values_prev: (N,) values one time step earlier (for theta).
        dt: time step size.
    """
    # Find spot in grid
    idx = int(np.searchsorted(grid, spot))
    idx = max(1, min(idx, len(grid) - 2))

    # Delta: ∂V/∂S via central difference
    ds = grid[idx + 1] - grid[idx - 1]
    delta = (values[idx + 1] - values[idx - 1]) / ds if ds > 0 else 0.0

    # Gamma: ∂²V/∂S² via central difference
    ds_up = grid[idx + 1] - grid[idx]
    ds_dn = grid[idx] - grid[idx - 1]
    if ds_up > 0 and ds_dn > 0:
        gamma = 2 * (values[idx + 1] / (ds_up * (ds_up + ds_dn))
                      - values[idx] / (ds_up * ds_dn)
                      + values[idx - 1] / (ds_dn * (ds_up + ds_dn)))
    else:
        gamma = 0.0

    # Price: interpolated
    price = float(np.interp(spot, grid, values))

    # Theta: -(V_new - V_old) / dt
    theta = 0.0
    if values_prev is not None and dt > 0:
        price_prev = float(np.interp(spot, grid, values_prev))
        theta = -(price - price_prev) / dt

    return {"price": price, "delta": float(delta), "gamma": float(gamma), "theta": theta}


# ═══════════════════════════════════════════════════════════════
# 1D PDE Solver
# ═══════════════════════════════════════════════════════════════


class PDESolver1D:
    """1D parabolic PDE solver.

    Solves: ∂V/∂t + ½σ²S²∂²V/∂S² + (r-q)S∂V/∂S - rV = 0
    backward from T to 0.

    Args:
        method: time-stepping scheme.
        n_space: spatial grid points.
        n_time: time steps.
        grid_type: spatial grid type.
    """

    def __init__(
        self,
        method: PDEMethod = PDEMethod.CRANK_NICOLSON,
        n_space: int = 200,
        n_time: int = 200,
        grid_type: GridType = GridType.UNIFORM,
    ):
        self.method = method
        self.n_space = n_space
        self.n_time = n_time
        self.grid_type = grid_type

    def solve(
        self,
        spot: float,
        strike: float,
        T: float,
        vol: float,
        rate: float,
        div_yield: float = 0.0,
        is_call: bool = True,
        is_american: bool = False,
    ) -> PDEResult:
        """Solve the Black-Scholes PDE."""
        # Build grid
        s_min = spot * 0.01
        s_max = spot * 5.0
        S = build_grid(s_min, s_max, self.n_space, self.grid_type, concentration_point=strike)
        S = np.sort(S)  # ensure ascending
        N = len(S)
        dt = T / self.n_time

        # Terminal condition
        if is_call:
            V = np.maximum(S - strike, 0.0)
        else:
            V = np.maximum(strike - S, 0.0)

        payoff = V.copy()  # for American projection

        V_prev = V.copy()

        # Time-stepping backward
        theta = 0.5 if self.method == PDEMethod.CRANK_NICOLSON else (
            1.0 if self.method in (PDEMethod.IMPLICIT, PDEMethod.RANNACHER) else 0.0)

        for step in range(self.n_time):
            V_prev = V.copy()

            # Rannacher: use implicit for first 2 steps, then CN
            if self.method == PDEMethod.RANNACHER and step >= 2:
                theta = 0.5

            if self.method == PDEMethod.METHOD_OF_LINES:
                V = self._mol_step(S, V, vol, rate, div_yield, dt)
            else:
                V = self._theta_step(S, V, vol, rate, div_yield, dt, theta)

            # American: project onto payoff
            if is_american:
                V = np.maximum(V, payoff)

            # Boundary conditions
            if is_call:
                V[0] = 0.0
                V[-1] = S[-1] - strike * math.exp(-rate * (self.n_time - step) * dt)
            else:
                V[0] = strike * math.exp(-rate * (self.n_time - step) * dt) - S[0]
                V[-1] = 0.0

        # Extract Greeks
        greeks = extract_greeks(S, V, spot, V_prev, dt)

        return PDEResult(
            values=V, grid=S,
            price=greeks["price"],
            delta=greeks["delta"],
            gamma=greeks["gamma"],
            theta=greeks["theta"],
            vega=None,
            method=self.method.value,
            n_space=N,
            n_time=self.n_time,
            grid_type=self.grid_type.value,
        )

    def _theta_step(self, S, V, vol, r, q, dt, theta):
        """Single θ-scheme time step."""
        N = len(S)
        ds = np.diff(S)

        # Build tridiagonal coefficients at interior points
        a = np.zeros(N)  # sub-diagonal
        b = np.zeros(N)  # diagonal
        c = np.zeros(N)  # super-diagonal
        rhs = np.zeros(N)

        for i in range(1, N - 1):
            ds_m = S[i] - S[i - 1]
            ds_p = S[i + 1] - S[i]
            ds_avg = 0.5 * (ds_m + ds_p)

            sigma2 = vol**2 * S[i]**2
            drift = (r - q) * S[i]

            # Second derivative coefficient
            d2 = sigma2 / (ds_m * ds_p)
            # First derivative coefficient (central)
            d1 = drift / (ds_m + ds_p)

            a[i] = d2 / 2 - d1
            b[i] = -d2 - r
            c[i] = d2 / 2 + d1

        # Implicit part: (I - θ dt L) V_new = (I + (1-θ) dt L) V_old
        # Explicit RHS
        for i in range(1, N - 1):
            rhs[i] = V[i] + (1 - theta) * dt * (a[i] * V[i - 1] + b[i] * V[i] + c[i] * V[i + 1])

        # Implicit: solve tridiagonal
        if theta > 0:
            # Build tridiagonal system
            lower = np.zeros(N)
            diag = np.ones(N)
            upper = np.zeros(N)
            for i in range(1, N - 1):
                lower[i] = -theta * dt * a[i]
                diag[i] = 1 - theta * dt * b[i]
                upper[i] = -theta * dt * c[i]

            V_new = _solve_tridiagonal(lower, diag, upper, rhs)
        else:
            V_new = rhs.copy()

        V_new[0] = V[0]
        V_new[-1] = V[-1]
        return V_new

    def _mol_step(self, S, V, vol, r, q, dt):
        """Method of lines: spatial FD → RK4 in time."""
        N = len(S)

        def f_rhs(t, v):
            dv = np.zeros(N)
            for i in range(1, N - 1):
                ds_m = S[i] - S[i - 1]
                ds_p = S[i + 1] - S[i]
                sigma2 = vol**2 * S[i]**2
                drift = (r - q) * S[i]
                d2 = sigma2 / (ds_m * ds_p)
                d1 = drift / (ds_m + ds_p)
                dv[i] = (d2 / 2 - d1) * v[i - 1] + (-d2 - r) * v[i] + (d2 / 2 + d1) * v[i + 1]
            return dv

        # RK4 step (backward: negative dt)
        k1 = dt * f_rhs(0, V)
        k2 = dt * f_rhs(0, V + 0.5 * k1)
        k3 = dt * f_rhs(0, V + 0.5 * k2)
        k4 = dt * f_rhs(0, V + k3)
        return V + (k1 + 2 * k2 + 2 * k3 + k4) / 6


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════


def solve_bs_pde(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    rate: float,
    div_yield: float = 0.0,
    is_call: bool = True,
    is_american: bool = False,
    method: PDEMethod = PDEMethod.CRANK_NICOLSON,
    n_space: int = 200,
    n_time: int = 200,
    grid_type: GridType = GridType.UNIFORM,
) -> PDEResult:
    """Solve Black-Scholes PDE for European or American option."""
    solver = PDESolver1D(method, n_space, n_time, grid_type)
    return solver.solve(spot, strike, T, vol, rate, div_yield, is_call, is_american)


def solve_pde_with_vega(
    spot: float,
    strike: float,
    T: float,
    vol: float,
    rate: float,
    div_yield: float = 0.0,
    is_call: bool = True,
    **kwargs,
) -> PDEResult:
    """Solve BS PDE and compute vega via bump-and-reprice."""
    result = solve_bs_pde(spot, strike, T, vol, rate, div_yield, is_call, **kwargs)
    bump = 0.01  # 1% vol bump
    result_up = solve_bs_pde(spot, strike, T, vol + bump, rate, div_yield, is_call, **kwargs)
    result.vega = (result_up.price - result.price) / bump
    return result


# ═══════════════════════════════════════════════════════════════
# Tridiagonal solver
# ═══════════════════════════════════════════════════════════════


def _solve_tridiagonal(a, b, c, d):
    """Thomas algorithm for tridiagonal system Ax = d.

    a: sub-diagonal, b: diagonal, c: super-diagonal.
    """
    n = len(d)
    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] - a[i] * c_[i - 1]
        if abs(denom) < 1e-15:
            denom = 1e-15
        c_[i] = c[i] / denom
        d_[i] = (d[i] - a[i] * d_[i - 1]) / denom

    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x


# ═══════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════

# The old functions (hundsdorfer_verwer, psor_2d, operator_splitting)
# are preserved via imports from models/ where they now live.
# New code should use PDESolver1D or solve_bs_pde.
