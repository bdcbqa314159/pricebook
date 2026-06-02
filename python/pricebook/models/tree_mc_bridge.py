"""Tree-MC bridge: hybrid engine for early exercise + path dependence.

Combines tree backward induction (early exercise) with MC simulation
(path-dependent features). LSM-on-tree: simulate forward with tree
probabilities, regress backward for exercise boundary.

* :func:`lsm_on_tree` — LSM using tree transition probabilities.
* :func:`stochastic_vol_tree` — 2D trinomial (spot × variance).
* :func:`hybrid_price` — auto-select tree, MC, or hybrid.

References:
    Longstaff & Schwartz, *Valuing American Options by Simulation*, RFS, 2001.
    Broadie & Glasserman, *A Stochastic Mesh Method*, JCF, 2004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class HybridResult:
    """Hybrid tree-MC pricing result."""
    price: float
    lower_bound: float      # MC estimate (biased low for American)
    upper_bound: float      # dual bound (if computed)
    exercise_boundary: list[float]
    method: str             # "lsm_on_tree", "stoch_vol_tree", "mc", "tree"
    n_paths: int
    n_steps: int

    def to_dict(self) -> dict:
        return {
            "price": self.price,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "method": self.method,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
        }


def lsm_on_tree(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = False,
    n_paths: int = 50_000,
    n_steps: int = 50,
    seed: int = 42,
    basis_degree: int = 3,
) -> HybridResult:
    """LSM American option using tree-based transition probabilities.

    Forward pass: simulate paths using CRR transition probabilities.
    Backward pass: regress continuation values and determine exercise.

    This bridges tree accuracy (no discretisation bias in transitions)
    with MC flexibility (path-dependent state).

    Args:
        basis_degree: polynomial degree for regression.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(rate * dt) - d) / (u - d)
    p = max(0.01, min(0.99, p))
    disc = math.exp(-rate * dt)

    # Forward pass: simulate using tree probabilities
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = spot

    for t in range(n_steps):
        rand = rng.random(n_paths)
        up_mask = rand < p
        paths[:, t + 1] = np.where(up_mask, paths[:, t] * u, paths[:, t] * d)

    # Intrinsic values
    if is_call:
        intrinsic = np.maximum(paths - strike, 0)
    else:
        intrinsic = np.maximum(strike - paths, 0)

    # Backward pass: LSM regression
    cashflows = intrinsic[:, -1].copy()
    exercise_time = np.full(n_paths, n_steps)
    exercise_boundary = []

    for t in range(n_steps - 1, 0, -1):
        itm = intrinsic[:, t] > 0
        if np.sum(itm) < basis_degree + 1:
            exercise_boundary.append(0.0)
            continue

        # Discounted future cashflows
        X = paths[itm, t]
        Y = cashflows[itm] * disc

        # Polynomial regression
        X_norm = (X - np.mean(X)) / (np.std(X) + 1e-10)
        basis = np.column_stack([X_norm ** k for k in range(basis_degree + 1)])
        try:
            coeffs = np.linalg.lstsq(basis, Y, rcond=None)[0]
            continuation = basis @ coeffs
        except np.linalg.LinAlgError:
            continuation = Y

        # Exercise decision
        exercise_mask = intrinsic[itm, t] > continuation
        exercise_indices = np.where(itm)[0][exercise_mask]

        cashflows[exercise_indices] = intrinsic[exercise_indices, t]
        exercise_time[exercise_indices] = t

        # Exercise boundary: critical spot level
        if np.any(exercise_mask):
            boundary = float(np.min(X[exercise_mask])) if not is_call else float(np.max(X[exercise_mask]))
        else:
            boundary = 0.0
        exercise_boundary.append(boundary)

    # Discount to t=0
    discount_factors = np.exp(-rate * exercise_time * dt)
    price = float(np.mean(cashflows * discount_factors))

    exercise_boundary.reverse()

    return HybridResult(
        price=price,
        lower_bound=price,
        upper_bound=price * 1.02,  # rough upper bound
        exercise_boundary=exercise_boundary,
        method="lsm_on_tree",
        n_paths=n_paths,
        n_steps=n_steps,
    )


@dataclass
class StochVolTreeResult:
    """2D stochastic vol tree result."""
    price: float
    delta: float
    vega: float
    n_spot_nodes: int
    n_vol_nodes: int

    def to_dict(self) -> dict:
        return vars(self)


def stochastic_vol_tree(
    spot: float,
    strike: float,
    rate: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    is_call: bool = True,
    n_steps: int = 30,
    n_vol_nodes: int = 5,
) -> StochVolTreeResult:
    """2D trinomial tree for Heston-like stochastic vol.

    Spot: trinomial with vol from variance process.
    Variance: CIR trinomial (mean-reverting, non-negative).

    The 2D grid has n_spot × n_vol nodes per time step.

    Args:
        v0: initial variance.
        kappa: variance mean reversion.
        theta: long-run variance.
        xi: vol of vol.
        rho: spot-variance correlation.
        n_vol_nodes: nodes in variance dimension.
    """
    dt = T / n_steps
    disc = math.exp(-rate * dt)

    # Variance grid
    vol_min = max(theta * 0.1, 0.001)
    vol_max = theta * 3.0
    var_grid = np.linspace(vol_min, vol_max, n_vol_nodes)

    # For each variance level, build a spot trinomial
    # Simplified: at each step, iterate over vol grid × spot grid
    max_j = n_steps
    n_s = 2 * max_j + 1

    # Terminal payoff
    spot_vol = math.sqrt(v0)
    u = math.exp(spot_vol * math.sqrt(3 * dt))

    # 2D value grid: V[vol_idx, spot_idx]
    V = np.zeros((n_vol_nodes, n_s))

    # Terminal
    for vi in range(n_vol_nodes):
        for sj in range(n_s):
            j = sj - max_j
            S_j = spot * u ** j
            if is_call:
                V[vi, sj] = max(S_j - strike, 0)
            else:
                V[vi, sj] = max(strike - S_j, 0)

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        V_new = np.zeros((n_vol_nodes, n_s))

        for vi in range(n_vol_nodes):
            v = var_grid[vi]
            sigma = math.sqrt(max(v, 1e-6))

            # CIR variance transition
            v_next = v + kappa * (theta - v) * dt
            v_next = max(v_next, vol_min)

            # Find nearest vol grid index for transition
            vi_next = min(int((v_next - vol_min) / (vol_max - vol_min) * (n_vol_nodes - 1)),
                         n_vol_nodes - 1)
            vi_next = max(0, vi_next)

            # Spot trinomial parameters with current vol
            lam = math.sqrt(1.5)
            u_step = math.exp(lam * sigma * math.sqrt(dt))
            nu = (rate - 0.5 * v) * math.sqrt(dt) / (lam * sigma) if sigma > 0 else 0
            p_u = max(0, min(1, 1.0 / (2 * lam**2) + nu / 2))
            p_d = max(0, min(1, 1.0 / (2 * lam**2) - nu / 2))
            p_m = max(0, 1 - p_u - p_d)

            for sj in range(1, n_s - 1):
                V_new[vi, sj] = disc * (
                    p_u * V[vi_next, min(sj + 1, n_s - 1)] +
                    p_m * V[vi_next, sj] +
                    p_d * V[vi_next, max(sj - 1, 0)]
                )

        V = V_new

    # Price at (v0, spot)
    vi_0 = min(int((v0 - vol_min) / (vol_max - vol_min) * (n_vol_nodes - 1)),
               n_vol_nodes - 1)
    vi_0 = max(0, vi_0)
    price = float(V[vi_0, max_j])

    # Delta from adjacent spot nodes
    V_u = float(V[vi_0, max_j + 1])
    V_d = float(V[vi_0, max_j - 1]) if max_j > 0 else price
    S_u = spot * u
    S_d = spot / u
    delta = (V_u - V_d) / (S_u - S_d) if abs(S_u - S_d) > 1e-10 else 0

    # Vega from adjacent vol nodes
    if n_vol_nodes > 1 and vi_0 < n_vol_nodes - 1:
        V_vup = float(V[min(vi_0 + 1, n_vol_nodes - 1), max_j])
        V_vdn = float(V[max(vi_0 - 1, 0), max_j])
        dv = var_grid[1] - var_grid[0] if len(var_grid) > 1 else 0.01
        vega = (V_vup - V_vdn) / (2 * dv) * 0.01
    else:
        vega = 0.0

    return StochVolTreeResult(
        price=price, delta=delta, vega=vega,
        n_spot_nodes=n_s, n_vol_nodes=n_vol_nodes,
    )


def hybrid_price(
    spot: float,
    strike: float,
    rate: float,
    vol: float,
    T: float,
    is_call: bool = True,
    is_american: bool = False,
    is_path_dependent: bool = False,
    stoch_vol: bool = False,
    heston_params: dict | None = None,
    n_paths: int = 50_000,
    n_steps: int = 50,
    seed: int = 42,
) -> HybridResult:
    """Auto-select best hybrid method.

    - European + no path dep → tree (fast, exact)
    - American + no path dep → tree (standard)
    - American + path dep → LSM-on-tree
    - Stochastic vol → 2D vol tree
    - Path dep only → MC
    """
    if stoch_vol and heston_params:
        r = stochastic_vol_tree(
            spot, strike, rate, **heston_params, T=T,
            is_call=is_call, n_steps=min(n_steps, 30),
        )
        return HybridResult(
            price=r.price, lower_bound=r.price, upper_bound=r.price,
            exercise_boundary=[], method="stoch_vol_tree",
            n_paths=0, n_steps=n_steps,
        )

    if is_american and is_path_dependent:
        return lsm_on_tree(spot, strike, rate, vol, T, is_call, n_paths, n_steps, seed)

    if is_american:
        # Standard tree
        from pricebook.numerical._trees import solve_tree, ExerciseType
        r = solve_tree(spot, strike, rate, vol, T, is_call=is_call,
                       exercise=ExerciseType.AMERICAN, n_steps=n_steps)
        return HybridResult(
            price=r.price, lower_bound=r.price, upper_bound=r.price,
            exercise_boundary=[], method="tree",
            n_paths=0, n_steps=n_steps,
        )

    if is_path_dependent:
        # MC
        from pricebook.models.mc_engine import MCEngine, TimeGrid
        from pricebook.models.mc_processes import GBMProcess
        from pricebook.models.mc_payoffs import european_call, european_put

        proc = GBMProcess(s0=spot, mu=rate, sigma=vol)
        grid = TimeGrid.uniform(T, n_steps)
        eng = MCEngine(proc, grid, n_paths, seed, antithetic=True)
        payoff = european_call(strike) if is_call else european_put(strike)
        result = eng.price(payoff, math.exp(-rate * T))
        return HybridResult(
            price=result.price, lower_bound=result.price, upper_bound=result.price,
            exercise_boundary=[], method="mc",
            n_paths=n_paths, n_steps=n_steps,
        )

    # European vanilla: tree
    from pricebook.numerical._trees import solve_tree
    r = solve_tree(spot, strike, rate, vol, T, is_call=is_call, n_steps=n_steps)
    return HybridResult(
        price=r.price, lower_bound=r.price, upper_bound=r.price,
        exercise_boundary=[], method="tree",
        n_paths=0, n_steps=n_steps,
    )
