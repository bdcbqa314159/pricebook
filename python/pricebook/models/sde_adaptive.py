"""Adaptive SDE time stepping via embedded Euler-Milstein error pair.

Step-size control based on local error estimate. Automatically
uses smaller steps near jumps, barriers, or high-curvature regions.

* :func:`adaptive_euler` — Euler with step-size control.
* :func:`adaptive_milstein` — Milstein with embedded error estimate.

References:
    Gaines & Lyons, *Variable Step Size Control in SDE Simulation*, 1997.
    Lamba, *An Adaptive Time-Stepping Algorithm for SDEs*, 2003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptiveSDEResult:
    """Adaptive SDE simulation result."""
    terminal_values: np.ndarray     # (n_paths,)
    n_steps_avg: float              # average steps used
    n_steps_max: int
    n_steps_min: int
    dt_min_used: float
    dt_max_used: float

    def to_dict(self) -> dict:
        return {
            "n_steps_avg": self.n_steps_avg,
            "n_steps_min": self.n_steps_min,
            "n_steps_max": self.n_steps_max,
        }


def adaptive_euler(
    x0: float,
    mu_fn,
    sigma_fn,
    T: float,
    n_paths: int = 10_000,
    dt_init: float = 0.01,
    dt_min: float = 1e-5,
    dt_max: float = 0.1,
    tol: float = 1e-3,
    seed: int = 42,
) -> AdaptiveSDEResult:
    """Euler-Maruyama with adaptive step-size control.

    Error estimate: compare Euler step with half-step (two Euler steps
    of size dt/2). If error > tol, halve dt. If error < tol/4, double dt.

    Args:
        x0: initial value.
        mu_fn: drift callable(x, t) → float.
        sigma_fn: diffusion callable(x, t) → float.
        T: terminal time.
        dt_init: initial step size.
        dt_min: minimum step size.
        dt_max: maximum step size.
        tol: local error tolerance.
    """
    rng = np.random.default_rng(seed)

    terminal = np.zeros(n_paths)
    step_counts = np.zeros(n_paths, dtype=int)
    dt_mins = np.full(n_paths, dt_max)
    dt_maxs = np.zeros(n_paths)

    for path in range(n_paths):
        x = x0
        t = 0.0
        dt = dt_init
        steps = 0

        while t < T - 1e-12:
            dt = min(dt, T - t)
            dW = rng.standard_normal() * math.sqrt(dt)

            # Full step
            x_full = x + mu_fn(x, t) * dt + sigma_fn(x, t) * dW

            # Two half-steps on the SAME Brownian path via Brownian bridge.
            #
            # Fix T4-SDE1: pre-fix used `dW1 = dW * sqrt(0.5)` and
            # `dW2 = dW - dW1`.  This algebraically sums to dW but gives
            # Var(dW1) = 0.5·dt (correct) and Var(dW2) = (1 - √0.5)²·dt ≈
            # 0.086·dt (should be 0.5·dt).  The second half-step's diffusion
            # was grossly under-stated, making the half-step look artificially
            # close to the full-step and under-estimating the local error —
            # so the adaptive controller accepted steps that should have been
            # rejected.
            # Correct construction: given dW (the FULL Brownian increment),
            # the bridge midpoint dW1 = W_{t+dt/2} − W_t ~ N(dW/2, dt/4).
            dt_half = dt / 2
            Z = rng.standard_normal()
            dW1 = 0.5 * dW + math.sqrt(dt_half * 0.5) * Z   # = dW/2 + √(dt/4)·Z
            dW2 = dW - dW1

            x_half = x + mu_fn(x, t) * dt_half + sigma_fn(x, t) * dW1
            x_two = x_half + mu_fn(x_half, t + dt_half) * dt_half + sigma_fn(x_half, t + dt_half) * dW2

            # Error estimate
            error = abs(x_full - x_two)

            if error > tol and dt > dt_min:
                # Reject: halve step size
                dt = max(dt / 2, dt_min)
                continue

            # Accept step (use the more accurate two-step value)
            x = x_two
            t += dt
            steps += 1

            dt_mins[path] = min(dt_mins[path], dt)
            dt_maxs[path] = max(dt_maxs[path], dt)

            # Adjust step size
            if error < tol / 4 and dt < dt_max:
                dt = min(dt * 2, dt_max)

        terminal[path] = x
        step_counts[path] = steps

    return AdaptiveSDEResult(
        terminal_values=terminal,
        n_steps_avg=float(np.mean(step_counts)),
        n_steps_max=int(np.max(step_counts)),
        n_steps_min=int(np.min(step_counts)),
        dt_min_used=float(np.min(dt_mins)),
        dt_max_used=float(np.max(dt_maxs)),
    )


def adaptive_milstein(
    x0: float,
    mu_fn,
    sigma_fn,
    sigma_prime_fn,
    T: float,
    n_paths: int = 10_000,
    dt_init: float = 0.01,
    dt_min: float = 1e-5,
    dt_max: float = 0.1,
    tol: float = 1e-3,
    seed: int = 42,
) -> AdaptiveSDEResult:
    """Milstein with embedded Euler error estimate for step control.

    Error = |Milstein − Euler| ≈ 0.5 σ σ' (dW² − dt).
    If error > tol, reject and halve dt.

    Args:
        sigma_prime_fn: derivative of diffusion dσ/dx.
    """
    rng = np.random.default_rng(seed)

    terminal = np.zeros(n_paths)
    step_counts = np.zeros(n_paths, dtype=int)
    dt_mins = np.full(n_paths, dt_max)
    dt_maxs = np.zeros(n_paths)

    for path in range(n_paths):
        x = x0
        t = 0.0
        dt = dt_init
        steps = 0

        while t < T - 1e-12:
            dt = min(dt, T - t)
            dW = rng.standard_normal() * math.sqrt(dt)

            mu = mu_fn(x, t)
            sig = sigma_fn(x, t)
            sig_p = sigma_prime_fn(x, t)

            # Euler step
            x_euler = x + mu * dt + sig * dW

            # Milstein correction
            milstein_corr = 0.5 * sig * sig_p * (dW**2 - dt)
            x_milstein = x_euler + milstein_corr

            # Error estimate
            error = abs(milstein_corr)

            if error > tol and dt > dt_min:
                dt = max(dt / 2, dt_min)
                continue

            x = x_milstein
            t += dt
            steps += 1

            dt_mins[path] = min(dt_mins[path], dt)
            dt_maxs[path] = max(dt_maxs[path], dt)

            if error < tol / 4 and dt < dt_max:
                dt = min(dt * 2, dt_max)

        terminal[path] = x
        step_counts[path] = steps

    return AdaptiveSDEResult(
        terminal_values=terminal,
        n_steps_avg=float(np.mean(step_counts)),
        n_steps_max=int(np.max(step_counts)),
        n_steps_min=int(np.min(step_counts)),
        dt_min_used=float(np.min(dt_mins)),
        dt_max_used=float(np.max(dt_maxs)),
    )
