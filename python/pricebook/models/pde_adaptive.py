"""Adaptive grid refinement for PDE solvers.

Error-indicator-based h-refinement: insert nodes where solution
gradient is large (near strikes, barriers).

* :func:`adaptive_pde` — PDE with h-refinement.
* :func:`refine_grid` — insert nodes based on error indicator.
* :func:`error_indicator` — gradient-based error estimate.

References:
    Figlewski & Gao, *The Adaptive Mesh Model*, JFE, 1999.
    Forsyth & Vetzal, *Quadratic Convergence for Valuing American Options
    Using a Penalty Method*, SISC, 2002.
"""

from __future__ import annotations

import math

import numpy as np

from pricebook.models.pde_protocol import (
    PDESpec, PDEEngine, PDEPricingResult,
)
from pricebook.numerical._pde import build_grid, GridType, extract_greeks


def error_indicator(grid: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Gradient-based error indicator per cell.

    Uses second derivative (curvature) as proxy for local truncation error.
    High curvature → more refinement needed.

    Returns: (N-2,) array of error indicators for interior cells.
    """
    N = len(grid)
    indicators = np.zeros(N - 2)

    for i in range(1, N - 1):
        ds_up = grid[i + 1] - grid[i]
        ds_dn = grid[i] - grid[i - 1]
        if ds_up > 0 and ds_dn > 0:
            # Second derivative (curvature)
            d2v = abs(
                values[i + 1] / (ds_up * (ds_up + ds_dn))
                - values[i] / (ds_up * ds_dn)
                + values[i - 1] / (ds_dn * (ds_up + ds_dn))
            )
            # Scale by cell size squared (truncation error ~ h² × V'')
            indicators[i - 1] = d2v * (ds_up + ds_dn)**2

    return indicators


def refine_grid(
    grid: np.ndarray,
    values: np.ndarray,
    max_new_points: int = 50,
    threshold_pct: float = 0.10,
) -> np.ndarray:
    """Insert nodes where error indicator is high.

    Args:
        grid: current grid.
        values: solution on current grid.
        max_new_points: maximum number of points to insert.
        threshold_pct: refine cells above this percentile of error.

    Returns:
        Refined grid (sorted, with new midpoints inserted).
    """
    indicators = error_indicator(grid, values)
    if len(indicators) == 0:
        return grid

    threshold = np.percentile(indicators, (1 - threshold_pct) * 100)

    new_points = []
    for i in range(len(indicators)):
        if indicators[i] >= threshold and len(new_points) < max_new_points:
            midpoint = 0.5 * (grid[i] + grid[i + 1])
            new_points.append(midpoint)

    if not new_points:
        return grid

    refined = np.sort(np.concatenate([grid, np.array(new_points)]))
    # Remove duplicates
    refined = np.unique(refined)
    return refined


def adaptive_pde(
    spec: PDESpec,
    spot: float,
    n_space_initial: int = 100,
    n_time: int = 200,
    max_refinements: int = 3,
    max_points: int = 500,
    target_error: float = 0.01,
) -> PDEPricingResult:
    """PDE with adaptive h-refinement.

    1. Solve on coarse grid.
    2. Compute error indicators.
    3. Refine grid where error is high.
    4. Re-solve on refined grid.
    5. Repeat until convergence or max refinements.

    Args:
        n_space_initial: initial grid size.
        max_refinements: maximum refinement iterations.
        max_points: maximum total grid points.
        target_error: stop if Richardson error < target.
    """
    grid = build_grid(spec.s_min, spec.s_max, n_space_initial, GridType.SINH,
                      concentration_point=spot)

    prev_price = None

    for ref_iter in range(max_refinements + 1):
        # Solve on current grid
        engine = PDEEngine("crank_nicolson", "uniform", len(grid), n_time, compute_vega=False)
        # Override grid in engine (hack: directly set in solve)
        result = _solve_on_grid(grid, spec, spot, n_time)

        if prev_price is not None:
            error_est = abs(result.price - prev_price)
            if error_est < target_error:
                break

        prev_price = result.price

        if ref_iter < max_refinements and len(grid) < max_points:
            V = result.values if result.values is not None else spec.payoff(grid)
            grid = refine_grid(grid, V, max_new_points=min(50, max_points - len(grid)))

    # Final solve with vega
    final = _solve_on_grid(grid, spec, spot, n_time, compute_vega=True)
    return final


def _solve_on_grid(grid, spec, spot, n_time, compute_vega=False):
    """Solve PDE on a specific (possibly non-uniform) grid."""
    from pricebook.models.pde_protocol import (
        PDEEngine, PDEPricingResult, PDEConvergenceInfo, _thomas_solve,
    )
    import time

    t0 = time.time()
    N = len(grid)
    dt = spec.T / n_time
    V = spec.payoff(grid).astype(float)
    V_prev = V.copy()

    for step in range(n_time):
        tau = (step + 1) * dt
        t = spec.T - tau

        # θ=0.5 (Crank-Nicolson) with non-uniform grid
        a = np.zeros(N)
        b = np.zeros(N)
        c = np.zeros(N)

        for i in range(1, N - 1):
            S = grid[i]
            ds_up = grid[i + 1] - grid[i]
            ds_dn = grid[i] - grid[i - 1]
            ds_avg = 0.5 * (ds_up + ds_dn)

            diff = spec.coefficients.diffusion(S, t)
            conv = spec.coefficients.convection(S, t)
            react = spec.coefficients.reaction(S, t)

            a[i] = diff / (ds_dn * ds_avg) - conv / (2 * ds_avg)
            c[i] = diff / (ds_up * ds_avg) + conv / (2 * ds_avg)
            b[i] = -(diff / (ds_up * ds_avg) + diff / (ds_dn * ds_avg)) + react

        rhs = V.copy()
        for i in range(1, N - 1):
            rhs[i] = V[i] + 0.5 * dt * (a[i] * V[i-1] + b[i] * V[i] + c[i] * V[i+1])

        lower = -0.5 * dt * a[1:N-1]
        diag = 1 - 0.5 * dt * b[1:N-1]
        upper = -0.5 * dt * c[1:N-1]
        V_new = V.copy()
        V_new[1:N-1] = _thomas_solve(lower, diag, upper, rhs[1:N-1])

        if spec.is_american and spec.exercise_payoff is not None:
            V_new = np.maximum(V_new, spec.exercise_payoff(grid))

        if spec.bc_lower is not None:
            V_new[0] = spec.bc_lower(grid[0], tau)
        if spec.bc_upper is not None:
            V_new[-1] = spec.bc_upper(grid[-1], tau)

        if step == n_time - 2:
            V_prev = V.copy()
        V = V_new

    greeks = extract_greeks(grid, V, spot, V_prev, dt)
    elapsed = time.time() - t0

    conv = PDEConvergenceInfo(
        n_space=N, n_time=n_time,
        ds_min=float(np.min(np.diff(grid))),
        dt=dt, cfl=0, elapsed_seconds=elapsed,
        method="adaptive_cn", grid_type="adaptive",
    )

    return PDEPricingResult(
        price=greeks["price"], delta=greeks["delta"],
        gamma=greeks["gamma"], theta=greeks["theta"],
        vega=0.0, convergence=conv, grid=grid, values=V,
    )
