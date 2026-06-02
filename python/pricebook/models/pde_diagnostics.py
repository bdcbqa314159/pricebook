"""PDE convergence diagnostics and automatic scheme selection.

* :func:`convergence_study` — grid refinement convergence analysis.
* :func:`recommend_scheme` — automatic method/grid recommendation.
* :func:`stability_check` — CFL and stability verification.

References:
    Duffy, *Finite Difference Methods in Financial Engineering*, Ch. 5.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from pricebook.models.pde_protocol import PDESpec, PDEEngine, pde_price


@dataclass
class ConvergenceStudyResult:
    """Grid refinement convergence study."""
    grid_sizes: list[int]
    prices: list[float]
    errors: list[float]         # vs finest grid
    convergence_order: float    # estimated from log-log fit
    richardson_price: float     # Richardson-extrapolated
    elapsed: list[float]

    def to_dict(self) -> dict:
        return {
            "grid_sizes": self.grid_sizes,
            "prices": self.prices,
            "convergence_order": self.convergence_order,
            "richardson_price": self.richardson_price,
        }


def convergence_study(
    spot: float,
    strike: float,
    vol: float,
    rate: float,
    T: float,
    is_call: bool = True,
    method: str = "crank_nicolson",
    grid_sizes: list[int] | None = None,
) -> ConvergenceStudyResult:
    """Run PDE at multiple grid resolutions for convergence analysis.

    Args:
        grid_sizes: spatial grid sizes to test.
    """
    sizes = grid_sizes or [50, 100, 200, 400, 800]

    prices = []
    times_elapsed = []

    for n in sizes:
        t0 = time.time()
        r = pde_price(spot, strike, vol, rate, T, is_call, method=method,
                       n_space=n, n_time=n)
        elapsed = time.time() - t0
        prices.append(r.price)
        times_elapsed.append(elapsed)

    # Errors vs finest
    finest = prices[-1]
    errors = [abs(p - finest) for p in prices]

    # Convergence order: log(error) ~ -p × log(n) + const
    if len(sizes) >= 3 and all(e > 1e-15 for e in errors[:-1]):
        log_n = [math.log(n) for n in sizes[:-1]]
        log_e = [math.log(max(e, 1e-15)) for e in errors[:-1]]
        if len(log_n) >= 2:
            coeffs = np.polyfit(log_n, log_e, 1)
            order = -coeffs[0]
        else:
            order = 2.0
    else:
        order = 2.0

    # Richardson extrapolation (last two)
    if len(prices) >= 2:
        p = max(order, 1.0)
        ratio = sizes[-1] / sizes[-2]
        richardson = (ratio**p * prices[-1] - prices[-2]) / (ratio**p - 1)
    else:
        richardson = prices[-1]

    return ConvergenceStudyResult(
        grid_sizes=sizes,
        prices=prices,
        errors=errors,
        convergence_order=order,
        richardson_price=richardson,
        elapsed=times_elapsed,
    )


@dataclass
class SchemeRecommendation:
    """PDE scheme recommendation."""
    method: str
    grid_type: str
    n_space: int
    n_time: int
    reason: str

    def to_dict(self) -> dict:
        return vars(self)


def recommend_scheme(
    vol: float,
    T: float,
    is_american: bool = False,
    has_barrier: bool = False,
    has_local_vol: bool = False,
    target_accuracy: str = "medium",
) -> SchemeRecommendation:
    """Recommend PDE method, grid, and resolution.

    Args:
        target_accuracy: "low" (fast), "medium" (balanced), "high" (precise).
    """
    # Base resolution
    if target_accuracy == "low":
        n_base = 100
    elif target_accuracy == "high":
        n_base = 500
    else:
        n_base = 200

    # Method selection
    if is_american:
        method = "crank_nicolson"
        reason = "CN with projection for American exercise"
    elif has_barrier:
        method = "crank_nicolson"
        reason = "CN with barrier absorption"
    elif has_local_vol:
        method = "crank_nicolson"
        reason = "CN with time-dependent local vol coefficients"
    else:
        method = "crank_nicolson"
        reason = "CN (2nd-order accurate, unconditionally stable)"

    # Grid selection
    if has_barrier:
        grid_type = "sinh"
        reason += " + sinh grid concentrated near barrier"
    elif vol > 0.5:
        grid_type = "log"
        reason += " + log grid for high vol"
    else:
        grid_type = "sinh"
        reason += " + sinh grid concentrated near strike"

    # Time steps: match spatial resolution
    n_time = n_base

    return SchemeRecommendation(method, grid_type, n_base, n_time, reason)


def stability_check(
    vol: float,
    T: float,
    n_space: int,
    n_time: int,
    spot: float,
    method: str = "crank_nicolson",
) -> dict:
    """Check CFL stability condition.

    For explicit: CFL = σ² Δt / Δx² < 1 required.
    For CN/implicit: unconditionally stable but CFL > 1 may cause oscillations.

    Returns stability metrics and warnings.
    """
    # Approximate grid spacing
    s_range = spot * 4  # typical range
    dx = s_range / n_space
    dt = T / n_time

    # CFL in physical space
    cfl = vol**2 * spot**2 * dt / (dx**2) if dx > 0 else 0

    # CFL in log-space (more relevant)
    dx_log = 4 * vol * math.sqrt(T) / n_space
    cfl_log = 0.5 * vol**2 * dt / (dx_log**2) if dx_log > 0 else 0

    warnings = []
    if method == "explicit" and cfl_log > 0.5:
        warnings.append(f"CFL = {cfl_log:.2f} > 0.5: explicit scheme UNSTABLE")
    if cfl_log > 10 and method == "crank_nicolson":
        warnings.append(f"CFL = {cfl_log:.2f} high: CN may show oscillations near discontinuities")
    if n_time < 10:
        warnings.append("Very few time steps: temporal error may dominate")

    return {
        "cfl_physical": cfl,
        "cfl_logspace": cfl_log,
        "dx": dx,
        "dx_log": dx_log,
        "dt": dt,
        "stable": len(warnings) == 0,
        "warnings": warnings,
        "method": method,
    }
