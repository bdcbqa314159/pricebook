"""Exercise boundary extraction and analysis for American options.

The exercise boundary S*(t) is the critical spot price at time t below (for puts)
or above (for calls) which immediate exercise is optimal. This module extracts and
analyses the boundary from three numerical methods — PDE (Crank-Nicolson), CRR
binomial tree, and LSM Monte Carlo — and provides tools to compare them.

Key properties of the exercise boundary:
- S*(T) = K * min(1, r/q) for puts  (equals K when q=0)
- Slope near expiry follows the square-root formula (Kim 1990):
      dS*/dt ~ -σ√(2/π) * S*  for puts
- Boundary is monotone increasing in T−t for puts (deeper ITM as maturity grows)

References:
    Kim, I. J. (1990). The analytic valuation of American options.
        *Review of Financial Studies*, 3(4), 547–572.
    Carr, P., Jarrow, R., & Myneni, R. (1992). Alternative characterizations of
        American put options. *Mathematical Finance*, 2(2), 87–106.
    Detemple, J. (2006). *American-Style Derivatives: Valuation and Computation*.
        Chapman & Hall / CRC. Ch. 3–4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExerciseBoundaryResult:
    """Extracted exercise boundary from a single numerical method."""

    time_grid: np.ndarray
    """Time points t in [0, T] at which the boundary is reported (ascending)."""

    boundary_values: np.ndarray
    """Critical spot S*(t) at each point in time_grid."""

    critical_price_at_expiry: float
    """S*(T) — the boundary value at expiry (should equal K for puts, q=0)."""

    boundary_slope: np.ndarray
    """Numerical derivative dS*/dt along the time grid."""

    boundary_convexity: np.ndarray
    """Second derivative d²S*/dt² along the time grid."""

    method: str = ""
    """Method used: 'pde', 'tree', or 'lsm'."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numerical_derivatives(
    t: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference first and second derivatives."""
    slope = np.gradient(y, t)
    convexity = np.gradient(slope, t)
    return slope, convexity



# ---------------------------------------------------------------------------
# PDE exercise boundary
# ---------------------------------------------------------------------------

def pde_exercise_boundary(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "put",
    n_spot: int = 500,
    n_time: int = 1000,
) -> ExerciseBoundaryResult:
    """Extract the exercise boundary from the PDE (Crank-Nicolson) solution.

    Runs the standard American Crank-Nicolson PDE backward in time. At each
    time step the full grid V(S, t) is available. The critical spot S*(t) is
    the largest S (for puts) or smallest S (for calls) where
        V(S, t) == intrinsic(S)
    found by scanning the grid and interpolating between adjacent nodes.

    Args:
        spot: reference spot for grid centering.
        strike: option strike.
        rate: risk-free rate.
        vol: flat volatility.
        T: time to expiry in years.
        q: continuous dividend yield.
        option_type: "put" or "call".
        n_spot: number of spot grid nodes.
        n_time: number of time steps.

    Returns:
        ExerciseBoundaryResult with time_grid and boundary_values arrays of
        length n_time+1 (one entry per PDE time level).
    """
    from pricebook.models.finite_difference import (
        _build_grid, _apply_boundary, _cn_step,
    )
    from pricebook.models.black76 import OptionType as OT

    ot = OT.PUT if option_type == "put" else OT.CALL
    g = _build_grid(spot, strike, rate, vol, T, ot, q, n_spot, n_time, spot_range=4.0)
    payoff = g.V.copy()

    # boundary_raw[k] = S*(t_k), where t_k runs from T down to 0
    boundary_raw: list[float] = []

    def _find_boundary(V: np.ndarray, S: np.ndarray) -> float:
        """Scan grid to find the critical spot where V == intrinsic."""
        if option_type == "put":
            # For puts: exercise region is S < S*
            # V(S) > intrinsic(S) in the hold region (larger S)
            # V(S) == intrinsic(S) in the exercise region (smaller S)
            diff = V - np.maximum(strike - S, 0.0)
            # Find rightmost node where diff <= 0 (exercise optimal)
            exercise_nodes = np.where(diff <= 1e-8)[0]
            if len(exercise_nodes) == 0:
                return float(S[0])
            idx = exercise_nodes[-1]
            if idx >= len(S) - 1:
                return float(S[idx])
            # Linear interpolation between idx and idx+1
            d0, d1 = diff[idx], diff[idx + 1]
            if abs(d1 - d0) < 1e-14:
                return float(S[idx])
            w = -d0 / (d1 - d0)
            return float(S[idx] + w * (S[idx + 1] - S[idx]))
        else:
            # For calls: exercise region is S > S*
            diff = V - np.maximum(S - strike, 0.0)
            hold_nodes = np.where(diff <= 1e-8)[0]
            if len(hold_nodes) == 0:
                return float(S[-1])
            idx = hold_nodes[0]
            if idx == 0:
                return float(S[0])
            d0, d1 = diff[idx - 1], diff[idx]
            if abs(d1 - d0) < 1e-14:
                return float(S[idx])
            w = -d0 / (d1 - d0)
            return float(S[idx - 1] + w * (S[idx] - S[idx - 1]))

    # Time at expiry (step 0 of PDE, tau=T backward)
    boundary_raw.append(_find_boundary(g.V, g.S))

    for step in range(n_time):
        tau = T - (step + 1) * g.dt
        _apply_boundary(g, tau)
        _cn_step(g)
        g.V[1:g.n_spot] = np.maximum(g.V[1:g.n_spot], payoff[1:g.n_spot])
        boundary_raw.append(_find_boundary(g.V, g.S))

    # PDE runs from tau=T (t=0) to tau=0 (t=T); reverse so time_grid is ascending
    boundary_arr = np.array(boundary_raw[::-1])
    time_grid = np.linspace(0.0, T, n_time + 1)
    slope, convexity = _numerical_derivatives(time_grid, boundary_arr)

    return ExerciseBoundaryResult(
        time_grid=time_grid,
        boundary_values=boundary_arr,
        critical_price_at_expiry=float(boundary_arr[-1]),
        boundary_slope=slope,
        boundary_convexity=convexity,
        method="pde",
    )


# ---------------------------------------------------------------------------
# Binomial tree exercise boundary
# ---------------------------------------------------------------------------

def tree_exercise_boundary(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "put",
    n_steps: int = 2000,
) -> ExerciseBoundaryResult:
    """Extract the exercise boundary from a CRR binomial tree.

    Builds the full CRR tree and on the backward pass records at each step
    the critical node where early exercise first becomes optimal.  The spot
    level of that node is the boundary estimate S*(t).

    Args:
        spot: current spot price.
        strike: option strike.
        rate: risk-free rate.
        vol: flat volatility.
        T: time to expiry.
        q: dividend yield.
        option_type: "put" or "call".
        n_steps: number of tree steps (more = finer boundary resolution).

    Returns:
        ExerciseBoundaryResult with boundary at each tree time level.
    """
    dt = T / n_steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    df = math.exp(-rate * dt)
    pu = (math.exp((rate - q) * dt) - d) / (u - d)
    pd = 1.0 - pu

    # Terminal stock prices: S_j = spot * u^(n-2j) for j=0..n
    j_arr = np.arange(n_steps + 1)
    S_terminal = spot * (u ** (n_steps - 2 * j_arr))

    if option_type == "put":
        V = np.maximum(strike - S_terminal, 0.0)
    else:
        V = np.maximum(S_terminal - strike, 0.0)

    boundary_vals: list[float] = [float(strike) if option_type == "put" else float(strike)]
    # Boundary at expiry = strike (approximately, for q=0)

    for step in range(n_steps - 1, -1, -1):
        # Stock prices at this step
        j_s = np.arange(step + 1)
        S_step = spot * (u ** (step - 2 * j_s))

        if option_type == "put":
            intrinsic = np.maximum(strike - S_step, 0.0)
        else:
            intrinsic = np.maximum(S_step - strike, 0.0)

        # Continuation = discounted expected value
        cont = df * (pu * V[:step + 1] + pd * V[1:step + 2])
        V = np.maximum(intrinsic, cont)

        # Find boundary: critical spot where exercise >= continuation
        ex_flag = intrinsic >= cont

        if option_type == "put":
            # Boundary = largest S where exercise is optimal
            ex_nodes = np.where(ex_flag)[0]
            if len(ex_nodes) > 0:
                # Smallest j (largest S) where exercise is optimal
                b_idx = ex_nodes[0]
                boundary_vals.append(float(S_step[b_idx]))
            else:
                boundary_vals.append(0.0)
        else:
            # Boundary = smallest S where exercise is optimal
            ex_nodes = np.where(ex_flag)[0]
            if len(ex_nodes) > 0:
                b_idx = ex_nodes[-1]
                boundary_vals.append(float(S_step[b_idx]))
            else:
                boundary_vals.append(float(S_step[-1] * 10))

    # boundary_vals was accumulated from t=T backward; reverse to get ascending t
    boundary_arr = np.array(boundary_vals[::-1])
    time_grid = np.linspace(0.0, T, n_steps + 1)
    slope, convexity = _numerical_derivatives(time_grid, boundary_arr)

    return ExerciseBoundaryResult(
        time_grid=time_grid,
        boundary_values=boundary_arr,
        critical_price_at_expiry=float(boundary_arr[-1]),
        boundary_slope=slope,
        boundary_convexity=convexity,
        method="tree",
    )


# ---------------------------------------------------------------------------
# LSM exercise boundary
# ---------------------------------------------------------------------------

def lsm_exercise_boundary(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "put",
    n_paths: int = 200_000,
    n_steps: int = 500,
    seed: int = 42,
    n_basis: int = 4,
) -> ExerciseBoundaryResult:
    """Extract the exercise boundary from LSM regression.

    At each exercise step the LSM regression yields a fitted continuation value
    function C_hat(S).  The critical spot S*(t) is where C_hat(S*(t)) = intrinsic(S*(t)),
    found by 1-D root search on the fitted polynomial in the ITM region.

    Args:
        spot: current spot price.
        strike: option strike.
        rate: risk-free rate.
        vol: flat volatility.
        T: time to expiry.
        q: dividend yield.
        option_type: "put" or "call".
        n_paths: number of MC paths.
        n_steps: number of exercise dates (all steps = American).
        seed: random seed.
        n_basis: number of polynomial basis functions for regression.

    Returns:
        ExerciseBoundaryResult.  The boundary at t=T is set to strike (analytical
        limit); interior values come from the regression crossover.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu_dt = (rate - q - 0.5 * vol * vol) * dt
    sig_sdt = vol * math.sqrt(dt)
    df = math.exp(-rate * dt)

    # Simulate full path matrix  shape: (n_paths, n_steps+1)
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = spot
    Z = rng.standard_normal((n_paths, n_steps))
    for s in range(n_steps):
        paths[:, s + 1] = paths[:, s] * np.exp(mu_dt + sig_sdt * Z[:, s])

    if option_type == "put":
        def payoff(S): return np.maximum(strike - S, 0.0)
    else:
        def payoff(S): return np.maximum(S - strike, 0.0)

    # Initialise cashflow and timing (same structure as lsm.py)
    cashflow = payoff(paths[:, -1])
    cash_step = np.full(n_paths, n_steps, dtype=float)

    # boundary_steps[k] corresponds to time t = k * dt
    boundary_steps: dict[int, float] = {n_steps: float(strike)}

    for step in range(n_steps - 1, 0, -1):
        s = paths[:, step]
        intrinsic = payoff(s)
        itm = intrinsic > 0

        if itm.sum() < n_basis + 1:
            boundary_steps[step] = boundary_steps.get(step + 1, float(strike))
            continue

        s_itm = s[itm]
        s_norm = s_itm / spot
        basis = np.column_stack([s_norm ** k for k in range(n_basis)])

        steps_fwd = cash_step[itm] - step
        cont_at_step = cashflow[itm] * np.exp(-rate * steps_fwd * dt)

        try:
            coeffs = np.linalg.lstsq(basis, cont_at_step, rcond=None)[0]
        except np.linalg.LinAlgError:
            boundary_steps[step] = boundary_steps.get(step + 1, float(strike))
            continue

        # Find the crossover S* where polynomial continuation = intrinsic
        # Sample a fine grid within the ITM range and locate the sign change
        s_min, s_max = float(s_itm.min()), float(s_itm.max())
        n_scan = 500
        s_scan = np.linspace(s_min, s_max, n_scan)
        s_scan_norm = s_scan / spot
        basis_scan = np.column_stack([s_scan_norm ** k for k in range(n_basis)])
        cont_scan = basis_scan @ coeffs

        if option_type == "put":
            intrinsic_scan = np.maximum(strike - s_scan, 0.0)
            # exercise if intrinsic > continuation → intrinsic - cont > 0
            diff = intrinsic_scan - cont_scan
            # Boundary = largest S where diff > 0 (before it flips negative)
            pos = np.where(diff > 0)[0]
            if len(pos) > 0 and pos[-1] < n_scan - 1:
                i0 = pos[-1]
                # Linear interpolation of zero crossing
                d0, d1 = diff[i0], diff[i0 + 1]
                if abs(d1 - d0) > 1e-14:
                    w = -d0 / (d1 - d0)
                    crit = s_scan[i0] + w * (s_scan[i0 + 1] - s_scan[i0])
                else:
                    crit = s_scan[i0]
                boundary_steps[step] = float(crit)
            elif len(pos) > 0:
                boundary_steps[step] = float(s_scan[pos[-1]])
            else:
                boundary_steps[step] = float(s_min)
        else:
            intrinsic_scan = np.maximum(s_scan - strike, 0.0)
            diff = intrinsic_scan - cont_scan
            neg = np.where(diff > 0)[0]
            if len(neg) > 0 and neg[0] > 0:
                i0 = neg[0] - 1
                d0, d1 = diff[i0], diff[i0 + 1]
                if abs(d1 - d0) > 1e-14:
                    w = -d0 / (d1 - d0)
                    crit = s_scan[i0] + w * (s_scan[i0 + 1] - s_scan[i0])
                else:
                    crit = s_scan[i0 + 1]
                boundary_steps[step] = float(crit)
            elif len(neg) > 0:
                boundary_steps[step] = float(s_scan[neg[0]])
            else:
                boundary_steps[step] = float(s_max)

        # Update cashflow (standard LSM)
        cont_hat = basis @ coeffs
        exercise = intrinsic[itm] > cont_hat
        ex_idx = np.where(itm)[0][exercise]
        cashflow[ex_idx] = intrinsic[itm][exercise]
        cash_step[ex_idx] = step

    # Assemble boundary array (ascending time)
    time_grid = np.linspace(0.0, T, n_steps + 1)
    boundary_arr = np.array([
        boundary_steps.get(k, float(strike)) for k in range(n_steps + 1)
    ])

    slope, convexity = _numerical_derivatives(time_grid, boundary_arr)

    return ExerciseBoundaryResult(
        time_grid=time_grid,
        boundary_values=boundary_arr,
        critical_price_at_expiry=float(boundary_arr[-1]),
        boundary_slope=slope,
        boundary_convexity=convexity,
        method="lsm",
    )


# ---------------------------------------------------------------------------
# Boundary analytics
# ---------------------------------------------------------------------------

def boundary_analytics(boundary_result: ExerciseBoundaryResult) -> dict:
    """Compute diagnostic properties of an exercise boundary.

    Properties returned:
    - slope_near_expiry: average dS*/dt in the last 5 % of time to expiry.
    - expected_slope_theory: theoretical slope −σ√(2/π)·S* (Kim 1990 approximation
      for puts near expiry).
    - slope_ratio: slope_near_expiry / expected_slope_theory.
    - mean_convexity: mean of d²S*/dt² over the full horizon.
    - moneyness_range: (min S*/K, max S*/K) showing the range of boundary moneyness.
    - time_to_exercise_50pct: time remaining when boundary is at 50 % of
      the distance between its min and max values (proxy for mid-life boundary).

    Args:
        boundary_result: output of pde_exercise_boundary, tree_exercise_boundary,
                         or lsm_exercise_boundary.

    Returns:
        dict with boundary diagnostics.
    """
    t = boundary_result.time_grid
    b = boundary_result.boundary_values
    slope = boundary_result.boundary_slope
    conv = boundary_result.boundary_convexity

    T = float(t[-1])
    # Near-expiry: last 5 % of calendar time
    near_expiry_mask = t >= 0.95 * T
    slope_near = float(np.mean(slope[near_expiry_mask])) if near_expiry_mask.sum() > 1 else float(slope[-1])

    # Theoretical slope for puts: dS*/dt ≈ −σ√(2/π)·S* near expiry
    # (This comes from the leading-order expansion of the integral equation)
    S_star_T = float(b[near_expiry_mask].mean()) if near_expiry_mask.sum() > 0 else float(b[-1])
    # NOTE: we don't have vol stored in ExerciseBoundaryResult.
    # Return slope_ratio as nan; caller can compute with known vol.
    expected_slope = float("nan")  # would need vol: -vol*sqrt(2/pi)*S_star_T
    slope_ratio = float("nan")

    # Moneyness of boundary
    # boundary_result does not carry strike; return absolute stats
    b_min = float(b.min())
    b_max = float(b.max())
    mean_convexity = float(np.mean(np.abs(conv)))

    # Time when boundary is at midpoint of its range
    b_mid = 0.5 * (b_min + b_max)
    crossings = np.where(np.diff(np.sign(b - b_mid)))[0]
    if len(crossings) > 0:
        t_mid = float(t[crossings[0]])
    else:
        t_mid = float("nan")

    return {
        "slope_near_expiry": slope_near,
        "expected_slope_theory": expected_slope,
        "slope_ratio": slope_ratio,
        "mean_convexity": mean_convexity,
        "boundary_min": b_min,
        "boundary_max": b_max,
        "critical_price_at_expiry": boundary_result.critical_price_at_expiry,
        "time_at_boundary_midpoint": t_mid,
        "method": boundary_result.method,
    }


# ---------------------------------------------------------------------------
# Boundary comparison
# ---------------------------------------------------------------------------

def compare_boundaries(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    q: float = 0.0,
    option_type: str = "put",
    n_spot: int = 300,
    n_time_pde: int = 600,
    n_steps_tree: int = 800,
    n_paths_lsm: int = 100_000,
    n_steps_lsm: int = 200,
    seed: int = 42,
) -> dict:
    """Run all three boundary extraction methods and compare the results.

    The three boundaries are interpolated onto a common time grid with
    n_time_pde+1 points for comparison.  Maximum and mean absolute deviations
    are computed for each pair of methods.

    Args:
        spot: current spot price.
        strike: option strike.
        rate: risk-free rate.
        vol: flat volatility.
        T: time to expiry.
        q: dividend yield.
        option_type: "put" or "call".
        n_spot: PDE spot grid size.
        n_time_pde: PDE time steps.
        n_steps_tree: binomial tree steps.
        n_paths_lsm: LSM paths.
        n_steps_lsm: LSM exercise steps.
        seed: random seed.

    Returns:
        dict with keys:
            pde    — ExerciseBoundaryResult from PDE.
            tree   — ExerciseBoundaryResult from tree (interpolated to PDE grid).
            lsm    — ExerciseBoundaryResult from LSM (interpolated to PDE grid).
            common_time_grid — the reference time grid.
            pde_values       — boundary on common grid.
            tree_values      — boundary on common grid.
            lsm_values       — boundary on common grid.
            pde_tree_max_dev  — max|S*_pde − S*_tree|.
            pde_lsm_max_dev   — max|S*_pde − S*_lsm|.
            tree_lsm_max_dev  — max|S*_tree − S*_lsm|.
            pde_tree_mean_dev — mean|S*_pde − S*_tree|.
            pde_lsm_mean_dev  — mean|S*_pde − S*_lsm|.
            tree_lsm_mean_dev — mean|S*_tree − S*_lsm|.
    """
    pde_result = pde_exercise_boundary(
        spot, strike, rate, vol, T, q, option_type,
        n_spot=n_spot, n_time=n_time_pde,
    )
    tree_result = tree_exercise_boundary(
        spot, strike, rate, vol, T, q, option_type,
        n_steps=n_steps_tree,
    )
    lsm_result = lsm_exercise_boundary(
        spot, strike, rate, vol, T, q, option_type,
        n_paths=n_paths_lsm, n_steps=n_steps_lsm, seed=seed,
    )

    common_t = pde_result.time_grid  # use PDE grid as reference

    # Interpolate tree and LSM onto common grid
    tree_interp = np.interp(common_t, tree_result.time_grid, tree_result.boundary_values)
    lsm_interp = np.interp(common_t, lsm_result.time_grid, lsm_result.boundary_values)
    pde_vals = pde_result.boundary_values

    def _max_dev(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b)))

    def _mean_dev(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.mean(np.abs(a - b)))

    return {
        "pde": pde_result,
        "tree": tree_result,
        "lsm": lsm_result,
        "common_time_grid": common_t,
        "pde_values": pde_vals,
        "tree_values": tree_interp,
        "lsm_values": lsm_interp,
        "pde_tree_max_dev": _max_dev(pde_vals, tree_interp),
        "pde_lsm_max_dev": _max_dev(pde_vals, lsm_interp),
        "tree_lsm_max_dev": _max_dev(tree_interp, lsm_interp),
        "pde_tree_mean_dev": _mean_dev(pde_vals, tree_interp),
        "pde_lsm_mean_dev": _mean_dev(pde_vals, lsm_interp),
        "tree_lsm_mean_dev": _mean_dev(tree_interp, lsm_interp),
    }
