"""Trinomial tree for TRS pricing with switching discount rate.

Implements Lou (2018) Section 4: Boyle trinomial tree with repo-rate drift
and Section 5: XVA decomposition via switching effective discount rate.

* :func:`trs_trinomial_tree` — single-period TRS via trinomial tree.
* :func:`trs_tree_xva` — XVA decomposition: CVA, DVA, CFA, DFA.

References:
    Lou, W. (2018). Pricing Total Return Swap. SSRN 3217420, Sections 4-5.
    Boyle, P.P. (1988). A Lattice Framework for Option Pricing with Two State Variables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricebook.xva import effective_discount_rate, xva_spread_decomposition


@dataclass
class TRSTreeResult:
    """TRS tree pricing result."""
    value: float
    value_star: float       # OIS-discounted (counterparty-risk-free)
    n_steps: int
    spot: float

    @property
    def price(self) -> float:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {"price": self.value, "value_star": self.value_star}


@dataclass
class TRSXVAResult:
    """TRS XVA decomposition result."""
    value_star: float       # V* (OIS-discounted)
    value: float            # V (with switching discount)
    total_xva: float        # U = V* - V
    cva: float
    dva: float
    cfa: float
    dfa: float

    @property
    def price(self) -> float:
        return self.value

    def to_dict(self) -> dict[str, float]:
        return {
            "price": self.value, "value_star": self.value_star,
            "total_xva": self.total_xva, "cva": self.cva, "dva": self.dva,
            "cfa": self.cfa, "dfa": self.dfa,
        }


def _trinomial_probabilities(
    rs_minus_q: float,
    sigma: float,
    dt: float,
) -> tuple[float, float, float, float, float]:
    """Trinomial tree parameters (Lou Eq 12).

    Returns (u, d, pu, pm, pd).
    """
    u = math.exp(sigma * math.sqrt(2 * dt))
    d = 1.0 / u

    # Probabilities (Eq 12)
    # Note: exponents use σ√(Δt/2), NOT σ√Δt/2
    sqrt_half_dt = math.sqrt(dt / 2)
    e_half = math.exp(rs_minus_q * dt / 2)
    e_up = math.exp(sigma * sqrt_half_dt)
    e_dn = math.exp(-sigma * sqrt_half_dt)

    denom = e_up - e_dn
    if abs(denom) < 1e-15:
        return u, d, 1.0 / 3, 1.0 / 3, 1.0 / 3

    pu = ((e_half - e_dn) / denom) ** 2
    pd = ((e_up - e_half) / denom) ** 2
    pm = 1.0 - pu - pd

    # Clamp probabilities
    # Clamp sequentially to preserve pu + pd + pm = 1
    pu = max(0.0, min(1.0, pu))
    pd = max(0.0, min(1.0 - pu, pd))
    pm = 1.0 - pu - pd

    return u, d, pu, pm, pd


def trs_trinomial_tree(
    S_0: float,
    r_f: float,
    T: float,
    r: float,
    rs_minus_r: float,
    sigma: float,
    div_yield: float = 0.0,
    n_steps: int = 100,
    mu: float = 1.0,
    r_b: float | None = None,
    r_c: float | None = None,
    M_0: float | None = None,
    margin_style: str = "full_csa",
) -> TRSTreeResult:
    """TRS pricing via trinomial tree with switching discount rate.

    Args:
        mu: collateralisation ratio. 1 = full CSA, 0 = uncollateralised.
        r_b: bank's unsecured rate (for switching discount).
        r_c: customer's unsecured rate.
        margin_style: "full_csa" or "repo_style".
    """
    if M_0 is None:
        M_0 = S_0
    if r_b is None:
        r_b = r
    if r_c is None:
        r_c = r
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not 0 <= mu <= 1:
        raise ValueError(f"mu (collateralisation ratio) must be in [0,1], got {mu}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    rs = r + rs_minus_r
    dt = T / n_steps

    u, d, pu, pm, pd = _trinomial_probabilities(rs - div_yield, sigma, dt)

    # Build spot tree (recombining: 2*n_steps + 1 nodes at step n)
    n_nodes = 2 * n_steps + 1
    S = np.zeros(n_nodes)
    for j in range(n_nodes):
        power = n_steps - j  # ranges from n_steps (top) to -n_steps (bottom)
        S[j] = S_0 * u ** max(power, 0) * d ** max(-power, 0)
        # Simpler: S[j] = S_0 * exp(sigma * sqrt(2dt) * (n_steps - j))
        S[j] = S_0 * math.exp(sigma * math.sqrt(2 * dt) * (n_steps - j))

    # Terminal payoff: H(T) = M0 rf T - (ST - S0)
    V = np.zeros(n_nodes)
    for j in range(n_nodes):
        V[j] = M_0 * r_f * T - (S[j] - S_0)

    # Also compute V* (OIS-discounted, no switching)
    V_star = V.copy()

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        n_j = 2 * step + 1  # nodes at this step
        S_step = np.zeros(n_j)
        V_new = np.zeros(n_j)
        V_star_new = np.zeros(n_j)

        for j in range(n_j):
            S_step[j] = S_0 * math.exp(sigma * math.sqrt(2 * dt) * (step - j))

            # Expected continuation
            j_next = j  # maps to j, j+1, j+2 in the next step's (n_j+2) array
            cont = pu * V[j_next] + pm * V[j_next + 1] + pd * V[j_next + 2]
            cont_star = pu * V_star[j_next] + pm * V_star[j_next + 1] + pd * V_star[j_next + 2]

            # Switching discount rate (Lou Eq 5)
            # W = V - L; sign of W determines rw
            if margin_style == "full_csa":
                W = 0.0  # full CSA: L = V, so W = 0
            elif margin_style == "repo_style":
                # repo-style: L = S0 - S + rF M0 (elapsed time)
                elapsed = (step + 1) * dt
                L = S_0 - S_step[j] + r_f * M_0 * elapsed
                # V at this node ≈ discounted continuation
                V_approx = cont * math.exp(-r * dt)
                W = V_approx - L
            else:
                raise ValueError(f"Unknown margin_style: {margin_style!r}")

            re = effective_discount_rate(mu, r, r_b, r_c, W)
            df = math.exp(-re * dt)
            df_star = math.exp(-r * dt)

            V_new[j] = cont * df
            V_star_new[j] = cont_star * df_star

        V = V_new
        V_star = V_star_new

    return TRSTreeResult(float(V[0]), float(V_star[0]), n_steps, S_0)


def trs_trinomial_tree_multi(
    S_0: float,
    r_f: float,
    T: float,
    r: float,
    rs_minus_r: float,
    sigma: float,
    n_periods: int = 1,
    div_yield: float = 0.0,
    n_steps_per_period: int = 50,
    M_0: float | None = None,
) -> TRSTreeResult:
    """Multi-period TRS via recursive trinomial tree (Lou 2018 Section 4).

    Backward recursion through reset dates:
    1. Compute V_K(S) = value of last period as a function of starting spot.
    2. For period K-1: terminal payoff includes V_K(S_terminal) as continuation.
    3. Continue backward to t_0.

    Each V_k(S) is stored on a log-spot grid and interpolated at sub-tree nodes.
    """
    if M_0 is None:
        M_0 = S_0
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    rs = r + rs_minus_r
    period_T = T / n_periods
    dt = period_T / n_steps_per_period

    u, d, pu, pm, pd = _trinomial_probabilities(rs - div_yield, sigma, dt)
    log_step = sigma * math.sqrt(2 * dt)
    df_dt = math.exp(-r * dt)

    # Continuation value function: V_next(S) for future periods.
    # Stored as (log_S_grid, values) for interpolation.
    continuation_log_grid = None
    continuation_values = None

    for period in range(n_periods - 1, -1, -1):
        # For each starting spot S_start, compute the value of this period
        # plus continuation from future periods.

        # Build a representative grid of starting spots.
        # At period > 0, the starting spots come from the previous period's tree.
        # At period 0, only S_0.
        if period == 0:
            start_spots = [S_0]
        else:
            # Use the tree grid at the end of the previous period
            n_grid = 2 * n_steps_per_period + 1
            start_spots = [S_0 * math.exp(log_step * (n_steps_per_period - j))
                           for j in range(n_grid)]

        period_values = np.zeros(len(start_spots))

        for idx, S_start in enumerate(start_spots):
            # Build sub-tree from S_start for one period
            n_nodes = 2 * n_steps_per_period + 1
            V = np.zeros(n_nodes)

            for j in range(n_nodes):
                S_j = S_start * math.exp(log_step * (n_steps_per_period - j))

                # Period payoff: M rf Δt - (S_end - S_start)
                payoff = M_0 * r_f * period_T - (S_j - S_start)

                # Add continuation from future periods
                if continuation_log_grid is not None:
                    log_S_j = math.log(S_j)
                    # Linear interpolation on log-spot grid
                    cont_val = np.interp(log_S_j, continuation_log_grid,
                                          continuation_values)
                    payoff += cont_val

                V[j] = payoff

            # Roll back this sub-tree to get period value at S_start
            for step in range(n_steps_per_period - 1, -1, -1):
                n_j = 2 * step + 1
                V_new = np.zeros(n_j)
                for j in range(n_j):
                    cont = pu * V[j] + pm * V[j + 1] + pd * V[j + 2]
                    V_new[j] = cont * df_dt
                V = V_new

            period_values[idx] = float(V[0])

        if period == 0:
            return TRSTreeResult(float(period_values[0]), float(period_values[0]),
                                  n_steps_per_period * n_periods, S_0)

        # Store this period's values as continuation for the next (earlier) period
        continuation_log_grid = np.array([math.log(s) for s in start_spots])
        continuation_values = period_values

        # Sort for interpolation
        sort_idx = np.argsort(continuation_log_grid)
        continuation_log_grid = continuation_log_grid[sort_idx]
        continuation_values = continuation_values[sort_idx]

    return TRSTreeResult(0.0, 0.0, 0, S_0)


def trs_tree_xva(
    S_0: float,
    r_f: float,
    T: float,
    r: float,
    rs_minus_r: float,
    sigma: float,
    r_b: float,
    r_c: float,
    s_b: float,
    s_c: float,
    mu_b: float = 0.0,
    mu_c: float = 0.0,
    div_yield: float = 0.0,
    n_steps: int = 100,
    mu: float = 0.0,
    M_0: float | None = None,
    margin_style: str = "full_csa",
) -> TRSXVAResult:
    """Path-level XVA decomposition: CVA, DVA, CFA, DFA (Lou Eq 14-18).

    Runs a single combined tree pass. At each node:
    - Computes V* (OIS-discounted) and V (switching discount)
    - Determines exposure W = V* - L (unsecured amount)
    - Accumulates CVA/DVA/CFA/DFA based on exposure sign and spreads

    CVA = Σ sc × max(W, 0) × df_rw × Δt  (Eq 15)
    DVA = Σ sb × max(-W, 0) × df_rw × Δt  (Eq 16)
    CFA = Σ μc × max(W, 0) × df_rw × Δt  (Eq 17)
    DFA = Σ μb × max(-W, 0) × df_rw × Δt  (Eq 18)
    """
    if M_0 is None:
        M_0 = S_0
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    rs = r + rs_minus_r
    dt = T / n_steps

    u, d, pu, pm, pd = _trinomial_probabilities(rs - div_yield, sigma, dt)

    # Terminal payoff
    n_nodes = 2 * n_steps + 1
    V_star = np.zeros(n_nodes)  # OIS-discounted
    V = np.zeros(n_nodes)       # switching discount
    for j in range(n_nodes):
        S_j = S_0 * math.exp(sigma * math.sqrt(2 * dt) * (n_steps - j))
        payoff = M_0 * r_f * T - (S_j - S_0)
        V_star[j] = payoff
        V[j] = payoff

    # XVA accumulators (backward-accumulated, probability-weighted)
    # At each step, we compute the expected XVA contribution from this time slice
    cva_total = 0.0
    dva_total = 0.0
    cfa_total = 0.0
    dfa_total = 0.0

    # Backward induction
    for step in range(n_steps - 1, -1, -1):
        n_j = 2 * step + 1
        V_star_new = np.zeros(n_j)
        V_new = np.zeros(n_j)

        for j in range(n_j):
            S_j = S_0 * math.exp(sigma * math.sqrt(2 * dt) * (step - j))

            j_next = j
            cont_star = pu * V_star[j_next] + pm * V_star[j_next + 1] + pd * V_star[j_next + 2]
            cont = pu * V[j_next] + pm * V[j_next + 1] + pd * V[j_next + 2]

            # Compute exposure at next step (average across children)
            # W = V* - L; for full CSA: L = V, W = V* - V
            if margin_style == "full_csa":
                W = 0.0
            elif margin_style == "repo_style":
                elapsed = (step + 1) * dt
                L = S_0 - S_j + r_f * M_0 * elapsed
                V_approx = cont * math.exp(-r * dt)
                W = V_approx - L
            else:
                raise ValueError(f"Unknown margin_style: {margin_style!r}")

            # Switching discount
            re = effective_discount_rate(mu, r, r_b, r_c, W)
            df = math.exp(-re * dt)
            df_star = math.exp(-r * dt)

            V_star_new[j] = cont_star * df_star
            V_new[j] = cont * df

        # Accumulate XVA from this time slice
        # The exposure at step+1 determines the XVA contribution
        # Weight by probability of reaching each node (uniform for tree expectation)
        n_next = 2 * (step + 1) + 1
        for j in range(min(n_next, len(V_star))):
            S_j = S_0 * math.exp(sigma * math.sqrt(2 * dt) * ((step + 1) - j))

            # Unsecured exposure at this node
            if margin_style == "full_csa":
                W_node = V_star[j] - V[j]  # V* - V is the XVA itself
            elif margin_style == "repo_style":
                elapsed = (step + 1) * dt
                L_node = S_0 - S_j + r_f * M_0 * elapsed
                W_node = V_star[j] - L_node
            else:
                W_node = 0.0

            # The node probability is uniform across the tree nodes
            # (the probabilities are already embedded in the backward induction)
            # For XVA accumulation, we use the exposure sign
            node_weight = dt / n_next  # approximate probability weight

            if W_node > 0:
                cva_total += s_c * W_node * node_weight
                cfa_total += mu_c * W_node * node_weight
            else:
                dva_total += s_b * abs(W_node) * node_weight
                dfa_total += mu_b * abs(W_node) * node_weight

        V_star = V_star_new
        V = V_new

    V_star_0 = float(V_star[0])
    V_0 = float(V[0])
    U = V_star_0 - V_0

    return TRSXVAResult(
        value_star=V_star_0, value=V_0, total_xva=U,
        cva=cva_total, dva=dva_total, cfa=cfa_total, dfa=dfa_total,
    )
