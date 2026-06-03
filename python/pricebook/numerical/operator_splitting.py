"""Operator splitting: Lie-Trotter and Strang for multi-physics PDEs.

* :func:`lie_trotter` — first-order A-B splitting.
* :func:`strang_splitting` — second-order A-B-A symmetric splitting.
* :func:`pide_strang` — Strang splitting for diffusion + jump PIDE.

References:
    Strang, *On the Construction and Comparison of Difference Schemes*,
    SIAM JNA, 1968.
    Ikonen & Toivanen, *Operator Splitting for American Options with
    Stochastic Volatility*, ANM, 2009.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SplittingResult:
    """Operator splitting result."""
    values: np.ndarray          # solution at T
    n_steps: int
    method: str                 # "lie_trotter" or "strang"
    splitting_error_order: int  # 1 for LT, 2 for Strang

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "method": self.method,
            "order": self.splitting_error_order,
        }


def lie_trotter(
    step_A,
    step_B,
    V0: np.ndarray,
    dt: float,
    n_steps: int,
) -> SplittingResult:
    """Lie-Trotter splitting: V^{n+1} = B(dt) A(dt) V^n.

    First-order in dt. Simple but O(dt) splitting error.

    Args:
        step_A: callable(V, dt) → V_new (operator A).
        step_B: callable(V, dt) → V_new (operator B).
        V0: initial condition.
        dt: time step.
        n_steps: number of steps.
    """
    V = V0.copy()
    for _ in range(n_steps):
        V = step_A(V, dt)
        V = step_B(V, dt)

    return SplittingResult(V, n_steps, "lie_trotter", 1)


def strang_splitting(
    step_A,
    step_B,
    V0: np.ndarray,
    dt: float,
    n_steps: int,
) -> SplittingResult:
    """Strang splitting: V^{n+1} = A(dt/2) B(dt) A(dt/2) V^n.

    Second-order in dt. Symmetric → no first-order error.

    Args:
        step_A: callable(V, dt) → V_new.
        step_B: callable(V, dt) → V_new.
        V0: initial condition.
        dt: time step.
        n_steps: number of steps.
    """
    V = V0.copy()
    half_dt = dt / 2

    for _ in range(n_steps):
        V = step_A(V, half_dt)
        V = step_B(V, dt)
        V = step_A(V, half_dt)

    return SplittingResult(V, n_steps, "strang", 2)


def pide_strang(
    diffusion_step,
    jump_step,
    V0: np.ndarray,
    dt: float,
    n_steps: int,
    boundary_fn=None,
) -> SplittingResult:
    """Strang splitting for PIDE: diffusion(dt/2) → jump(dt) → diffusion(dt/2).

    For jump-diffusion PDEs where:
    - diffusion_step solves ∂V/∂t = L_diff V (e.g. CN on BS operator)
    - jump_step applies ∫ [V(x+y) − V(x)] ν(dy) (explicit quadrature)

    The symmetric splitting reduces the splitting error from O(dt) to O(dt²).

    Args:
        diffusion_step: callable(V, dt) → V_new.
        jump_step: callable(V, dt) → V_new.
        boundary_fn: optional callable(V) → V with BCs applied.
    """
    V = V0.copy()
    half_dt = dt / 2

    for _ in range(n_steps):
        V = diffusion_step(V, half_dt)
        V = jump_step(V, dt)
        V = diffusion_step(V, half_dt)

        if boundary_fn is not None:
            V = boundary_fn(V)

    return SplittingResult(V, n_steps, "strang_pide", 2)


def splitting_error_estimate(
    step_A,
    step_B,
    V0: np.ndarray,
    dt: float,
    n_steps: int,
) -> float:
    """Estimate splitting error by comparing LT and Strang.

    error ≈ ||V_LT − V_Strang|| (should be O(dt)).
    """
    r_lt = lie_trotter(step_A, step_B, V0, dt, n_steps)
    r_st = strang_splitting(step_A, step_B, V0, dt, n_steps)
    return float(np.linalg.norm(r_lt.values - r_st.values))
