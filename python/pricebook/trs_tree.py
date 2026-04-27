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
    sqrt_dt = math.sqrt(dt)
    u = math.exp(sigma * math.sqrt(2 * dt))
    d = 1.0 / u

    # Probabilities (Eq 12)
    e_half = math.exp(rs_minus_q * dt / 2)
    e_up = math.exp(sigma * sqrt_dt / 2)
    e_dn = math.exp(-sigma * sqrt_dt / 2)

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
    """XVA decomposition: CVA, DVA, CFA, DFA (Lou Eq 14-18).

    U = V* - V = CVA - DVA + CFA - DFA

    where rb - r = sb + μb (CDS premium + bond/CDS basis).

    Args:
        s_b, s_c: CDS premiums of bank and customer.
        mu_b, mu_c: bond/CDS basis of bank and customer.
    """
    # Full CSA (V = V*)
    full_csa = trs_trinomial_tree(
        S_0, r_f, T, r, rs_minus_r, sigma, div_yield, n_steps,
        mu=1.0, r_b=r_b, r_c=r_c, M_0=M_0, margin_style=margin_style)

    # With switching (V)
    switching = trs_trinomial_tree(
        S_0, r_f, T, r, rs_minus_r, sigma, div_yield, n_steps,
        mu=mu, r_b=r_b, r_c=r_c, M_0=M_0, margin_style=margin_style)

    V_star = full_csa.value
    V = switching.value
    U = V_star - V

    # Approximate XVA decomposition
    # In the full decomposition, each component requires path-level indicators.
    # For the tree, we approximate by splitting U proportionally:
    total_spread_b = s_b + mu_b
    total_spread_c = s_c + mu_c

    decomp = xva_spread_decomposition(U, s_b, s_c, mu_b, mu_c)
    cva = decomp["cva"]
    dva = decomp["dva"]
    cfa = decomp["cfa"]
    dfa = decomp["dfa"]

    return TRSXVAResult(
        value_star=V_star, value=V, total_xva=U,
        cva=cva, dva=dva, cfa=cfa, dfa=dfa,
    )
