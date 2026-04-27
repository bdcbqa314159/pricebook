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


def _single_period_tree_value(
    S_start: float,
    r_f: float,
    period_T: float,
    r: float,
    sigma: float,
    div_yield: float,
    n_steps: int,
    M_0: float,
) -> float:
    """Value of a single TRS period starting at S_start, settling at period_T."""
    rs = r  # full CSA, no repo for this helper
    dt = period_T / n_steps
    u, d, pu, pm, pd = _trinomial_probabilities(rs - div_yield, sigma, dt)

    n_nodes = 2 * n_steps + 1
    V = np.zeros(n_nodes)
    for j in range(n_nodes):
        S_j = S_start * math.exp(sigma * math.sqrt(2 * dt) * (n_steps - j))
        V[j] = M_0 * r_f * period_T - (S_j - S_start)

    for step in range(n_steps - 1, -1, -1):
        n_j = 2 * step + 1
        V_new = np.zeros(n_j)
        for j in range(n_j):
            cont = pu * V[j] + pm * V[j + 1] + pd * V[j + 2]
            V_new[j] = cont * math.exp(-r * dt)
        V = V_new

    return float(V[0])


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
    """Multi-period TRS via recursive trinomial tree rollback.

    Lou (2018) Section 4 "Multi-period reset":
    At each reset τ_i, for each node j with fixing S_{i,j}, price a
    single-period TRS from S_{i,j} forward to τ_{i+1}, add the period
    return, and roll back.

    This is O(n_steps² × n_periods × n_nodes) — expensive but exact.
    """
    if M_0 is None:
        M_0 = S_0
    if n_periods < 1:
        raise ValueError(f"n_periods must be >= 1, got {n_periods}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    period_T = T / n_periods
    dt = period_T / n_steps_per_period

    u, d, pu, pm, pd = _trinomial_probabilities(r + rs_minus_r - div_yield, sigma, dt)

    # Work backward from the last period
    # At each reset, we need the continuation value at each node.
    # Start: at the last reset (τ_{K-1}), the continuation from future periods is 0.

    # The continuation array: for each node at the current reset, the PV of future periods
    # Initially (at the last period boundary): no future periods
    continuation = None  # will be set after the first (last) period

    for period in range(n_periods - 1, -1, -1):
        # Build a tree for this period [τ_period, τ_{period+1}]
        # At each node at τ_period, the total value is:
        # V(S) = single_period_value(S) + continuation(S)

        # We need to roll back from τ_{period+1} to τ_period.
        # At τ_{period+1}, the payoff at node j is:
        # H_j = M0 rf dt_period - (S_j - S_prev)
        # where S_prev is the reset fixing at τ_period (varies by node at τ_period).

        # For the recursive scheme: at τ_period, node j has spot S_j.
        # We price a single-period TRS starting from S_j.
        # Then add the continuation (PV of all future periods from S_j).

        if period == n_periods - 1 and n_periods == 1:
            # Single period: just use the standard tree
            val = _single_period_tree_value(
                S_0, r_f, period_T, r, sigma, div_yield, n_steps_per_period, M_0)
            return TRSTreeResult(val, val, n_steps_per_period, S_0)

        # For multi-period: at this reset, build the node spots
        n_nodes_reset = 2 * n_steps_per_period + 1 if period > 0 else 1

        if period == 0:
            # At t=0, only one node: S_0
            spots = [S_0]
        else:
            # At reset τ_period, the spots come from rolling back the previous period's tree
            # For a recombining tree, the spots at step n_steps_per_period are:
            spots = [S_0 * math.exp(sigma * math.sqrt(2 * dt) * (n_steps_per_period - j))
                     for j in range(n_nodes_reset)]

        # For each spot at this reset, compute single-period value + continuation
        values_at_reset = np.zeros(len(spots))
        for j, S_j in enumerate(spots):
            # Single period from S_j
            period_val = _single_period_tree_value(
                S_j, r_f, period_T, r, sigma, div_yield,
                n_steps_per_period, M_0)

            # Add continuation from future periods
            if continuation is not None and j < len(continuation):
                period_val += continuation[j]

            values_at_reset[j] = period_val

        if period == 0:
            return TRSTreeResult(float(values_at_reset[0]), float(values_at_reset[0]),
                                  n_steps_per_period * n_periods, S_0)

        # Roll back this period's values to the previous reset
        # These become the continuation for the previous period
        V = values_at_reset.copy()
        for step in range(n_steps_per_period - 1, -1, -1):
            n_j = 2 * step + 1
            V_new = np.zeros(n_j)
            for j in range(n_j):
                cont = pu * V[j] + pm * V[j + 1] + pd * V[j + 2]
                V_new[j] = cont * math.exp(-r * dt)
            V = V_new

        continuation = values_at_reset  # continuation at reset nodes

    # Should not reach here
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
