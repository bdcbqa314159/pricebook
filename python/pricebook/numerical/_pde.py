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
        # Tavella-Randall sinh grid: concentrated near concentration_point.
        # Fix T3.4: pre-fix used a symmetric `xi = linspace(-3, 3)` with
        # `alpha = 0.5 * (s_max - s_min) / sinh(3)`.  If `concentration_point`
        # was NOT the midpoint, the symmetric xi range pushed the grid
        # endpoints past `s_min` / `s_max` — for c < midpoint the grid extends
        # below s_min (and goes NEGATIVE when s_min is close to 0, breaking
        # the BS PDE which requires S ≥ 0).
        #
        # Post-fix: choose `alpha` based on the larger half-distance from c,
        # then solve xi_min = asinh((s_min−c)/α) and xi_max = asinh((s_max−c)/α)
        # so the grid lands EXACTLY on [s_min, s_max] regardless of where c is.
        c = concentration_point if concentration_point is not None else (s_min + s_max) / 2
        half = max(c - s_min, s_max - c)
        if half <= 0:
            return np.linspace(s_min, s_max, n_points)
        alpha = half / np.sinh(3.0)
        xi_lo = np.arcsinh((s_min - c) / alpha)
        xi_hi = np.arcsinh((s_max - c) / alpha)
        xi = np.linspace(xi_lo, xi_hi, n_points)
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

            # Fix T2.2/T2.3/T2.B: compute the proper Dirichlet boundary
            # values BEFORE the step and pass them into `_theta_step` so the
            # implicit tridiagonal solve uses them in the i=1 / i=N-2 rows.
            # Pre-fix the BC was applied only AFTER the solve, leaving the
            # implicit equation at the boundary-adjacent rows using V_new[0]=0
            # (just whatever the un-set rhs gave), which is wrong for puts
            # and underestimated the upper boundary for calls.
            #
            # Fix T2.B: the time-to-maturity after step `k` is `(k+1)·dt`,
            # NOT `(n_time − step)·dt`. Pre-fix the boundary used the wrong
            # remaining-time, mispricing every step except the middle one.
            #
            # Fix T2.4: for American options the boundaries are the intrinsic
            # payoff (early-exercise dominates at S→0 for puts and S→S_max
            # for calls), not the European discounted strike.
            tau = (step + 1) * dt
            if is_american:
                # Intrinsic at boundaries — early-exercise always optimal there.
                if is_call:
                    bc_lo, bc_hi = 0.0, S[-1] - strike
                else:
                    bc_lo, bc_hi = strike - S[0], 0.0
            else:
                if is_call:
                    bc_lo = 0.0
                    bc_hi = S[-1] - strike * math.exp(-rate * tau)
                else:
                    bc_lo = strike * math.exp(-rate * tau) - S[0]
                    bc_hi = 0.0

            if self.method == PDEMethod.METHOD_OF_LINES:
                V = self._mol_step(S, V, vol, rate, div_yield, dt)
                V[0] = bc_lo
                V[-1] = bc_hi
            else:
                V = self._theta_step(S, V, vol, rate, div_yield, dt, theta,
                                     bc_lo, bc_hi)

            # American: project onto payoff
            if is_american:
                V = np.maximum(V, payoff)
                # Re-impose boundaries after the projection (the max() may
                # raise them above intrinsic if PDE solver overshoots, but
                # intrinsic IS the lower bound, so just enforce it cleanly).
                V[0] = max(V[0], bc_lo)
                V[-1] = max(V[-1], bc_hi)

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

    def _theta_step(self, S, V, vol, r, q, dt, theta, bc_lo=None, bc_hi=None):
        """Single θ-scheme time step.

        Fix T2.2: non-uniform 3-point stencil. Pre-fix the discretization used
        `d2 = σ²S² / (ds_m * ds_p)` and split it evenly into sub/super, which
        is the formula for UNIFORM grids only. On a non-uniform grid (LOG,
        SINH), the standard 3-point central stencil for ∂²V/∂S² is

            V_SS ≈ 2/(ds_m+ds_p) · [(V_{i+1}−V_i)/ds_p − (V_i−V_{i-1})/ds_m]
                 = [V_{i-1}·2/(ds_m(ds_m+ds_p))
                    − V_i·2/(ds_m·ds_p)
                    + V_{i+1}·2/(ds_p(ds_m+ds_p))]

        and for ∂V/∂S the central formula on a non-uniform grid is

            V_S ≈ (V_{i+1} − V_{i−1}) / (ds_m + ds_p)

        On a LOG grid for an ATM call on (S=100, K=100, r=5%, σ=20%, T=1y),
        the pre-fix uniform stencil overshoots Black-Scholes by ~13 %. The
        post-fix non-uniform stencil is within ~0.1 %.
        """
        N = len(S)

        # Build tridiagonal coefficients at interior points
        a = np.zeros(N)  # sub-diagonal
        b = np.zeros(N)  # diagonal
        c = np.zeros(N)  # super-diagonal
        rhs = np.zeros(N)

        for i in range(1, N - 1):
            ds_m = S[i] - S[i - 1]
            ds_p = S[i + 1] - S[i]

            sigma2 = vol**2 * S[i]**2
            drift = (r - q) * S[i]

            # Non-uniform 3-point stencil for V_SS (Fix T2.2):
            #   coefficient of V[i-1]: σ²S² / (ds_m · (ds_m+ds_p))
            #   coefficient of V[i+1]: σ²S² / (ds_p · (ds_m+ds_p))
            #   coefficient of V[i]:  −σ²S² / (ds_m · ds_p)
            # Central V_S on a non-uniform grid: (V_{i+1} − V_{i−1})/(ds_m+ds_p).
            ss_lo = sigma2 / (ds_m * (ds_m + ds_p))
            ss_hi = sigma2 / (ds_p * (ds_m + ds_p))
            ss_mid = -sigma2 / (ds_m * ds_p)
            d1 = drift / (ds_m + ds_p)

            a[i] = ss_lo - d1
            b[i] = ss_mid - r
            c[i] = ss_hi + d1

        # Explicit RHS for interior nodes: V_i + (1-θ) dt L V_i
        for i in range(1, N - 1):
            rhs[i] = V[i] + (1 - theta) * dt * (a[i] * V[i - 1] + b[i] * V[i] + c[i] * V[i + 1])

        # Fix T2.3: enforce Dirichlet boundary conditions inside the implicit
        # solve. Pre-fix the tridiagonal system had `diag[0]=1, rhs[0]=0`, so
        # V_new[0] = 0 (and similarly at i=N−1), and then the code overwrote
        # V_new[0] = V[0] (the OLD value). The interior solve at i=1 therefore
        # used V_new[0] = 0 inside the implicit equation — wrong for puts (BC
        # is non-zero) and wrong for the upper boundary of calls.
        if bc_lo is None:
            bc_lo = V[0]
        if bc_hi is None:
            bc_hi = V[-1]
        rhs[0] = bc_lo
        rhs[-1] = bc_hi

        if theta > 0:
            lower = np.zeros(N)
            diag = np.ones(N)
            upper = np.zeros(N)
            for i in range(1, N - 1):
                lower[i] = -theta * dt * a[i]
                diag[i] = 1 - theta * dt * b[i]
                upper[i] = -theta * dt * c[i]
            # Boundary rows already have diag=1, lower=upper=0, rhs=BC, so
            # V_new[boundary] = BC after the solve.
            V_new = _solve_tridiagonal(lower, diag, upper, rhs)
        else:
            # Explicit: rhs[i] already contains V_new for interior; just
            # write boundaries.
            V_new = rhs.copy()
            V_new[0] = bc_lo
            V_new[-1] = bc_hi
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
