"""Boundary condition library for PDE solvers.

Unified BC specification that plugs into any solver.

* :class:`BCSpec` — boundary condition specification.
* :func:`apply_bc` — apply boundary conditions to solution vector.
* :func:`neumann_bc` — Neumann (flux) boundary.
* :func:`robin_bc` — Robin (mixed) boundary.

References:
    Duffy, *Finite Difference Methods in Financial Engineering*, Ch. 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class BCType(Enum):
    DIRICHLET = "dirichlet"       # V = g(S, t)
    NEUMANN = "neumann"           # ∂V/∂S = g(S, t)
    ROBIN = "robin"               # a V + b ∂V/∂S = g(S, t)
    LINEAR_EXTRAP = "linear"      # linear extrapolation from interior
    OUTFLOW = "outflow"           # zero second derivative


@dataclass
class BCSpec:
    """Boundary condition specification.

    For Dirichlet: V[boundary] = value_fn(S, t).
    For Neumann: (V[1] − V[0]) / ds = value_fn(S, t).
    For Robin: a × V + b × ∂V/∂S = value_fn(S, t).
    """
    bc_type: BCType
    value_fn: object = None     # callable(S, t) → float, or constant
    robin_a: float = 1.0        # Robin coefficient on V
    robin_b: float = 0.0        # Robin coefficient on ∂V/∂S

    @staticmethod
    def dirichlet(value) -> BCSpec:
        if callable(value):
            return BCSpec(BCType.DIRICHLET, value)
        return BCSpec(BCType.DIRICHLET, lambda S, t: value)

    @staticmethod
    def neumann(flux=0.0) -> BCSpec:
        if callable(flux):
            return BCSpec(BCType.NEUMANN, flux)
        return BCSpec(BCType.NEUMANN, lambda S, t: flux)

    @staticmethod
    def robin(a: float, b: float, value=0.0) -> BCSpec:
        if callable(value):
            return BCSpec(BCType.ROBIN, value, a, b)
        return BCSpec(BCType.ROBIN, lambda S, t: value, a, b)

    @staticmethod
    def linear_extrapolation() -> BCSpec:
        return BCSpec(BCType.LINEAR_EXTRAP)

    @staticmethod
    def outflow() -> BCSpec:
        return BCSpec(BCType.OUTFLOW)


def apply_bc(
    V: np.ndarray,
    grid: np.ndarray,
    t: float,
    bc_lower: BCSpec,
    bc_upper: BCSpec,
) -> np.ndarray:
    """Apply boundary conditions to solution vector.

    Modifies V in-place at boundaries.

    Args:
        V: solution vector (N,).
        grid: spatial grid (N,).
        t: current time.
        bc_lower: BC at grid[0].
        bc_upper: BC at grid[-1].
    """
    V = V.copy()

    # Lower boundary
    V[0] = _apply_single_bc(V, grid, 0, t, bc_lower, is_lower=True)

    # Upper boundary
    V[-1] = _apply_single_bc(V, grid, len(V) - 1, t, bc_upper, is_lower=False)

    return V


def _apply_single_bc(V, grid, idx, t, bc, is_lower):
    """Apply a single boundary condition."""
    S = grid[idx]

    if bc.bc_type == BCType.DIRICHLET:
        return bc.value_fn(S, t)

    elif bc.bc_type == BCType.NEUMANN:
        flux = bc.value_fn(S, t)
        if is_lower:
            ds = grid[1] - grid[0]
            return V[1] - flux * ds
        else:
            ds = grid[-1] - grid[-2]
            return V[-2] + flux * ds

    elif bc.bc_type == BCType.ROBIN:
        # a V + b ∂V/∂S = g  →  a*V[0] + b*(V[1]-V[0])/ds = g
        # V[0] = (g*ds - b*V[1]) / (a*ds - b)
        g = bc.value_fn(S, t)
        a, b = bc.robin_a, bc.robin_b
        if is_lower:
            ds = grid[1] - grid[0]
            denom = a * ds - b
            if abs(denom) > 1e-15:
                return (g * ds - b * V[1]) / denom
            return V[1]
        else:
            ds = grid[-1] - grid[-2]
            denom = a * ds + b
            if abs(denom) > 1e-15:
                return (g * ds + b * V[-2]) / denom
            return V[-2]

    elif bc.bc_type == BCType.LINEAR_EXTRAP:
        if is_lower:
            return 2 * V[1] - V[2]
        else:
            return 2 * V[-2] - V[-3]

    elif bc.bc_type == BCType.OUTFLOW:
        # Zero second derivative: V'' = 0 → V[0] = 2V[1] - V[2]
        if is_lower:
            return 2 * V[1] - V[2]
        else:
            return 2 * V[-2] - V[-3]

    return V[idx]


# ═══════════════════════════════════════════════════════════════
# Common financial BCs
# ═══════════════════════════════════════════════════════════════

def call_bcs(strike: float, rate: float) -> tuple[BCSpec, BCSpec]:
    """Standard call option BCs: V(0)=0, V(∞)=S−K×e^{-rτ}."""
    import math
    return (
        BCSpec.dirichlet(0.0),
        BCSpec.dirichlet(lambda S, t: S - strike * math.exp(-rate * t)),
    )


def put_bcs(strike: float, rate: float) -> tuple[BCSpec, BCSpec]:
    """Standard put option BCs: V(0)=K×e^{-rτ}, V(∞)=0."""
    import math
    return (
        BCSpec.dirichlet(lambda S, t: strike * math.exp(-rate * t)),
        BCSpec.dirichlet(0.0),
    )


def barrier_bcs(barrier: float, is_up: bool) -> tuple[BCSpec, BCSpec]:
    """Barrier option BCs: V=0 at barrier."""
    if is_up:
        return (BCSpec.linear_extrapolation(), BCSpec.dirichlet(0.0))
    else:
        return (BCSpec.dirichlet(0.0), BCSpec.linear_extrapolation())


def neumann_far_field() -> BCSpec:
    """Far-field Neumann: ∂V/∂S → 1 (call) or ∂V/∂S → 0."""
    return BCSpec.neumann(1.0)
