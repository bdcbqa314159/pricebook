"""Unified PDE protocol: specification, engine, result.

Provides a common interface for all PDE solvers (1D, 2D, FEM, spectral)
analogous to MC's ProcessSpec → MCEngine → MCResult pipeline.

* :class:`PDESpec` — PDE coefficient specification.
* :class:`PDEEngine` — solver that implements the protocol.
* :class:`PDEPricingResult` — unified result with Greeks and convergence.
* :func:`pde_price` — one-function entry point.

References:
    Duffy, *Finite Difference Methods in Financial Engineering*, Wiley, 2006.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from pricebook.numerical._pde import (
    PDEMethod, GridType, BoundaryCondition, PDEResult,
    build_grid, extract_greeks,
)


# ═══════════════════════════════════════════════════════════════
# PDE Specification
# ═══════════════════════════════════════════════════════════════

@dataclass
class PDECoefficients:
    """PDE coefficients: ∂V/∂t + a(S,t)∂²V/∂S² + b(S,t)∂V/∂S + c(S,t)V = 0.

    For Black-Scholes: a = ½σ²S², b = (r−q)S, c = −r.
    For local vol: a = ½σ(S,t)²S², same b, c.

    Coefficients are callables of (S, t) for time-dependent problems.
    """
    diffusion: Callable[[float, float], float]  # a(S, t)
    convection: Callable[[float, float], float]  # b(S, t)
    reaction: Callable[[float, float], float]    # c(S, t)

    @staticmethod
    def black_scholes(vol: float, rate: float, div_yield: float = 0.0) -> PDECoefficients:
        """Standard BS coefficients."""
        return PDECoefficients(
            diffusion=lambda S, t: 0.5 * vol**2 * S**2,
            convection=lambda S, t: (rate - div_yield) * S,
            reaction=lambda S, t: -rate,
        )

    @staticmethod
    def local_vol(vol_surface, rate: float, div_yield: float = 0.0) -> PDECoefficients:
        """Local volatility coefficients: σ(S, t) from Dupire surface.

        Args:
            vol_surface: callable(S, t) → local vol, or object with .vol(T, K).
        """
        if callable(vol_surface):
            vol_fn = vol_surface
        else:
            vol_fn = lambda S, t: vol_surface.vol(t, S)

        return PDECoefficients(
            diffusion=lambda S, t: 0.5 * vol_fn(S, t)**2 * S**2,
            convection=lambda S, t: (rate - div_yield) * S,
            reaction=lambda S, t: -rate,
        )

    @staticmethod
    def time_dependent(vol_fn, rate_fn, div_fn=None) -> PDECoefficients:
        """Time-dependent r(t), σ(t), q(t)."""
        q_fn = div_fn or (lambda t: 0.0)
        return PDECoefficients(
            diffusion=lambda S, t: 0.5 * vol_fn(t)**2 * S**2,
            convection=lambda S, t: (rate_fn(t) - q_fn(t)) * S,
            reaction=lambda S, t: -rate_fn(t),
        )


@dataclass
class PDESpec:
    """Complete PDE problem specification.

    Defines coefficients, domain, boundary conditions, terminal payoff,
    and optional early exercise constraint.
    """
    coefficients: PDECoefficients
    s_min: float
    s_max: float
    T: float
    payoff: Callable[[np.ndarray], np.ndarray]  # terminal condition V(S, T)
    bc_lower: Callable[[float, float], float] | None = None  # V(s_min, t)
    bc_upper: Callable[[float, float], float] | None = None  # V(s_max, t)
    is_american: bool = False
    exercise_payoff: Callable[[np.ndarray], np.ndarray] | None = None

    @staticmethod
    def european_call(spot: float, strike: float, vol: float, rate: float,
                      T: float, div_yield: float = 0.0) -> PDESpec:
        """Standard European call specification."""
        s_min = max(spot * 0.01, 1e-4)
        s_max = spot * 5.0
        return PDESpec(
            coefficients=PDECoefficients.black_scholes(vol, rate, div_yield),
            s_min=s_min, s_max=s_max, T=T,
            payoff=lambda S: np.maximum(S - strike, 0),
            bc_lower=lambda S, t: 0.0,
            bc_upper=lambda S, t: S - strike * math.exp(-rate * t),
        )

    @staticmethod
    def european_put(spot: float, strike: float, vol: float, rate: float,
                     T: float, div_yield: float = 0.0) -> PDESpec:
        s_min = max(spot * 0.01, 1e-4)
        s_max = spot * 5.0
        return PDESpec(
            coefficients=PDECoefficients.black_scholes(vol, rate, div_yield),
            s_min=s_min, s_max=s_max, T=T,
            payoff=lambda S: np.maximum(strike - S, 0),
            bc_lower=lambda S, t: strike * math.exp(-rate * t) - S,
            bc_upper=lambda S, t: 0.0,
        )

    @staticmethod
    def american_put(spot: float, strike: float, vol: float, rate: float,
                     T: float, div_yield: float = 0.0) -> PDESpec:
        s_min = max(spot * 0.01, 1e-4)
        s_max = spot * 5.0
        payoff = lambda S: np.maximum(strike - S, 0)
        return PDESpec(
            coefficients=PDECoefficients.black_scholes(vol, rate, div_yield),
            s_min=s_min, s_max=s_max, T=T,
            payoff=payoff,
            bc_lower=lambda S, t: strike - S,
            bc_upper=lambda S, t: 0.0,
            is_american=True,
            exercise_payoff=payoff,
        )


# ═══════════════════════════════════════════════════════════════
# PDE Engine
# ═══════════════════════════════════════════════════════════════

@dataclass
class PDEConvergenceInfo:
    """PDE convergence diagnostics."""
    n_space: int
    n_time: int
    ds_min: float           # smallest grid spacing
    dt: float
    cfl: float              # CFL number (stability)
    elapsed_seconds: float
    method: str
    grid_type: str

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class PDEPricingResult:
    """Unified PDE pricing result."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    convergence: PDEConvergenceInfo
    grid: np.ndarray | None = None
    values: np.ndarray | None = None

    def to_dict(self) -> dict:
        return {
            "price": self.price, "delta": self.delta, "gamma": self.gamma,
            "theta": self.theta, "vega": self.vega,
            "convergence": self.convergence.to_dict(),
        }


class PDEEngine:
    """Unified PDE pricing engine.

    Solves a PDESpec using the configured method, grid, and scheme.

    Args:
        method: time-stepping scheme.
        grid_type: spatial grid type.
        n_space: spatial grid points.
        n_time: time steps.
        compute_vega: if True, bump vol for vega.
    """

    def __init__(
        self,
        method: str = "crank_nicolson",
        grid_type: str = "log",
        n_space: int = 200,
        n_time: int = 200,
        compute_vega: bool = True,
    ):
        self.method = method
        self.grid_type = grid_type
        self.n_space = n_space
        self.n_time = n_time
        self.compute_vega = compute_vega

    def solve(self, spec: PDESpec, spot: float) -> PDEPricingResult:
        """Solve a PDE specification and return unified result.

        Args:
            spec: PDE problem specification.
            spot: spot price for interpolation and Greeks.
        """
        t0 = time.time()

        # Build grid — default to log for better accuracy
        type_map = {"sinh": GridType.SINH, "log": GridType.LOG, "uniform": GridType.UNIFORM}
        gt = type_map.get(self.grid_type, GridType.LOG)
        grid = build_grid(spec.s_min, spec.s_max, self.n_space, gt, concentration_point=spot)
        dt = spec.T / self.n_time

        # Terminal condition
        V = spec.payoff(grid).astype(float)

        # Time stepping
        V_prev = V.copy()
        for step in range(self.n_time):
            tau = (step + 1) * dt  # time from maturity
            t = spec.T - tau       # calendar time

            V = self._theta_step(grid, V, dt, spec.coefficients, t, spec)

            # American exercise
            if spec.is_american and spec.exercise_payoff is not None:
                exercise = spec.exercise_payoff(grid)
                V = np.maximum(V, exercise)

            # Boundary conditions
            if spec.bc_lower is not None:
                V[0] = spec.bc_lower(grid[0], tau)
            if spec.bc_upper is not None:
                V[-1] = spec.bc_upper(grid[-1], tau)

            if step == self.n_time - 2:
                V_prev = V.copy()

        # Extract Greeks
        greeks = extract_greeks(grid, V, spot, V_prev, dt)

        # Vega via bump
        vega = 0.0
        if self.compute_vega:
            vega = self._compute_vega(spec, spot, greeks["price"])

        elapsed = time.time() - t0
        ds_min = float(np.min(np.diff(grid)))
        vol_approx = 0.2  # approximate for CFL
        cfl = vol_approx**2 * dt / (ds_min**2) if ds_min > 0 else 0

        conv = PDEConvergenceInfo(
            n_space=self.n_space, n_time=self.n_time,
            ds_min=ds_min, dt=dt, cfl=cfl,
            elapsed_seconds=elapsed,
            method=self.method, grid_type=self.grid_type,
        )

        return PDEPricingResult(
            price=greeks["price"],
            delta=greeks["delta"],
            gamma=greeks["gamma"],
            theta=greeks["theta"],
            vega=vega,
            convergence=conv,
            grid=grid,
            values=V,
        )

    def _theta_step(self, grid, V, dt, coeffs, t, spec):
        """θ-scheme time step with general coefficients."""
        N = len(grid)
        if self.method == "explicit":
            theta_val = 0.0
        elif self.method == "implicit":
            theta_val = 1.0
        else:  # crank_nicolson
            theta_val = 0.5

        # Build tridiagonal coefficients at each node
        a = np.zeros(N)  # sub-diagonal
        b = np.zeros(N)  # diagonal
        c = np.zeros(N)  # super-diagonal

        for i in range(1, N - 1):
            S = grid[i]
            ds_up = grid[i + 1] - grid[i]
            ds_dn = grid[i] - grid[i - 1]
            ds_avg = 0.5 * (ds_up + ds_dn)

            diff = coeffs.diffusion(S, t)
            conv = coeffs.convection(S, t)
            react = coeffs.reaction(S, t)

            # Second derivative: diff × (V[i+1]/ds_up - V[i](1/ds_up+1/ds_dn) + V[i-1]/ds_dn) / ds_avg
            # First derivative: conv × (V[i+1] - V[i-1]) / (ds_up + ds_dn)
            a[i] = diff / (ds_dn * ds_avg) - conv / (2 * ds_avg)  # coefficient of V[i-1]
            c[i] = diff / (ds_up * ds_avg) + conv / (2 * ds_avg)  # coefficient of V[i+1]
            b[i] = -(diff / (ds_up * ds_avg) + diff / (ds_dn * ds_avg)) + react  # coefficient of V[i]

        # Explicit part: V_rhs = V + (1 - θ) × dt × L × V
        rhs = V.copy()
        for i in range(1, N - 1):
            rhs[i] = V[i] + (1 - theta_val) * dt * (a[i] * V[i - 1] + b[i] * V[i] + c[i] * V[i + 1])

        if theta_val > 0:
            # Implicit part: (I - θ × dt × L) × V_new = rhs
            lower = -theta_val * dt * a[1:N - 1]
            diag = 1 - theta_val * dt * b[1:N - 1]
            upper = -theta_val * dt * c[1:N - 1]

            # Thomas algorithm
            V_new = V.copy()
            V_new[1:N - 1] = _thomas_solve(lower, diag, upper, rhs[1:N - 1])
        else:
            V_new = rhs

        return V_new

    def _compute_vega(self, spec, spot, base_price):
        """Vega via vol bump (requires BS or local vol spec)."""
        # Bump diffusion by 1% vol
        orig_diff = spec.coefficients.diffusion
        bump = 0.01

        def bumped_diff(S, t):
            base = orig_diff(S, t)
            # a = 0.5 σ² S² → bumped = 0.5 (σ+dσ)² S² ≈ a + σ dσ S²
            sigma_approx = math.sqrt(2 * base / max(S**2, 1e-10))
            return 0.5 * (sigma_approx + bump)**2 * S**2

        bumped_coeffs = PDECoefficients(bumped_diff, spec.coefficients.convection, spec.coefficients.reaction)
        bumped_spec = PDESpec(
            bumped_coeffs, spec.s_min, spec.s_max, spec.T,
            spec.payoff, spec.bc_lower, spec.bc_upper,
            spec.is_american, spec.exercise_payoff,
        )

        # Solve bumped (without vega to avoid recursion)
        engine = PDEEngine(self.method, self.grid_type, self.n_space, self.n_time, compute_vega=False)
        bumped_result = engine.solve(bumped_spec, spot)
        return bumped_result.price - base_price


def _thomas_solve(lower, diag, upper, rhs):
    """Thomas algorithm for tridiagonal system."""
    n = len(diag)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    c_prime[0] = upper[0] / diag[0] if diag[0] != 0 else 0
    d_prime[0] = rhs[0] / diag[0] if diag[0] != 0 else 0

    for i in range(1, n):
        m = lower[i - 1] / (diag[i] - lower[i - 1] * c_prime[i - 1]) if abs(diag[i] - lower[i - 1] * c_prime[i - 1]) > 1e-15 else 0
        c_prime[i] = upper[i] / (diag[i] - lower[i - 1] * c_prime[i - 1]) if i < n - 1 and abs(diag[i] - lower[i - 1] * c_prime[i - 1]) > 1e-15 else 0
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / (diag[i] - lower[i - 1] * c_prime[i - 1]) if abs(diag[i] - lower[i - 1] * c_prime[i - 1]) > 1e-15 else 0

    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ═══════════════════════════════════════════════════════════════
# Convenience entry point
# ═══════════════════════════════════════════════════════════════

def pde_price(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    T: float,
    is_call: bool = True,
    is_american: bool = False,
    div_yield: float = 0.0,
    method: str = "crank_nicolson",
    grid_type: str = "log",
    n_space: int = 200,
    n_time: int = 200,
    vol_surface=None,
) -> PDEPricingResult:
    """One-function PDE pricing entry point.

    Args:
        vol_surface: if given, use local vol PDE. Otherwise constant vol.
    """
    if vol_surface is not None:
        coeffs = PDECoefficients.local_vol(vol_surface, rate, div_yield)
    else:
        coeffs = PDECoefficients.black_scholes(vol, rate, div_yield)

    s_min = max(spot * 0.01, 1e-4)
    s_max = spot * 5.0
    payoff = (lambda S: np.maximum(S - strike, 0)) if is_call else (lambda S: np.maximum(strike - S, 0))

    if is_call:
        bc_lo = lambda S, t: 0.0
        bc_hi = lambda S, t: S - strike * math.exp(-rate * t)
    else:
        bc_lo = lambda S, t: strike * math.exp(-rate * t) - S
        bc_hi = lambda S, t: 0.0

    spec = PDESpec(
        coefficients=coeffs,
        s_min=s_min, s_max=s_max, T=T,
        payoff=payoff,
        bc_lower=bc_lo, bc_upper=bc_hi,
        is_american=is_american,
        exercise_payoff=payoff if is_american else None,
    )

    engine = PDEEngine(method, grid_type, n_space, n_time)
    return engine.solve(spec, spot)
