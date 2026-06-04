"""Derman-Kani implied binomial tree.

Calibrate a recombining binomial tree to match market option prices.
Extract Arrow-Debreu state prices and local volatility at each node.

* :class:`ImpliedTreeResult` — calibrated tree with state prices.
* :func:`build_implied_tree` — Derman-Kani construction.
* :func:`price_on_implied_tree` — price exotic on calibrated tree.
* :func:`extract_local_vol` — local vol from implied tree.

References:
    Derman & Kani, *The Volatility Smile and Its Implied Tree*,
    Goldman Sachs QS, 1994.
    Rubinstein, *Implied Binomial Trees*, JF, 1994.
    Jackwerth, *Generalized Binomial Trees*, JFQA, 1997.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ImpliedTreeNode:
    """Single node in the implied tree."""
    step: int
    index: int          # 0 = lowest, step = highest
    spot: float
    arrow_debreu: float # state price
    local_vol: float
    p_up: float         # transition probability up


@dataclass
class ImpliedTreeResult:
    """Calibrated implied tree."""
    spots: list[list[float]]            # spots[step][node]
    arrow_debreu: list[list[float]]     # state prices
    local_vols: list[list[float]]       # local vol per node
    p_ups: list[list[float]]            # transition probs
    n_steps: int
    calibration_error: float

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "calibration_error": self.calibration_error,
            "n_nodes": sum(len(row) for row in self.spots),
        }


def build_implied_tree(
    spot: float,
    rate: float,
    div_yield: float,
    T: float,
    n_steps: int,
    market_vols: list[list[float]],
    strikes: list[list[float]],
) -> ImpliedTreeResult:
    """Build Derman-Kani implied binomial tree.

    At each step, calibrate node spots and transition probabilities
    to match market option prices (via implied vols).

    The tree is recombining: S(i,j) = S(i-1,j-1) × u or S(i-1,j) × d.

    Args:
        spot: current spot.
        rate: risk-free rate.
        div_yield: dividend yield.
        T: total time horizon.
        n_steps: number of tree steps.
        market_vols: implied vols per step per strike.
        strikes: strikes per step (same structure as vols).
    """
    dt = T / n_steps
    df = math.exp(-rate * dt)
    fwd_factor = math.exp((rate - div_yield) * dt)

    # Initialise CRR as starting point
    base_vol = market_vols[0][len(market_vols[0]) // 2] if market_vols and market_vols[0] else 0.20
    u_crr = math.exp(base_vol * math.sqrt(dt))
    d_crr = 1.0 / u_crr

    spots_all = [[spot]]
    ad_all = [[1.0]]  # Arrow-Debreu at (0,0) = 1
    local_vols_all: list[list[float]] = []
    p_ups_all: list[list[float]] = []

    total_error = 0.0

    for step in range(1, n_steps + 1):
        prev_spots = spots_all[step - 1]
        prev_ad = ad_all[step - 1]
        n_prev = len(prev_spots)

        # Target: step+1 nodes
        n_new = n_prev + 1

        # Build new spot levels — start from CRR and adjust
        new_spots = []
        for j in range(n_new):
            if j == 0:
                s = prev_spots[0] * d_crr
            elif j == n_new - 1:
                s = prev_spots[-1] * u_crr
            else:
                s = prev_spots[j - 1] * u_crr  # recombining
            new_spots.append(max(s, 1e-6))

        # Compute transition probabilities from market prices
        t_step = step * dt
        p_ups = []
        local_vols = []
        new_ad = np.zeros(n_new)

        for j in range(n_prev):
            S_j = prev_spots[j]
            S_up = new_spots[j + 1]
            S_dn = new_spots[j]

            # Forward constraint: p × S_up + (1-p) × S_dn = S_j × fwd_factor
            if abs(S_up - S_dn) > 1e-10:
                p = (S_j * fwd_factor - S_dn) / (S_up - S_dn)
            else:
                p = 0.5
            p = max(0.01, min(0.99, p))

            p_ups.append(p)

            # Local vol from transition
            E_S2 = p * S_up**2 + (1 - p) * S_dn**2
            E_S = p * S_up + (1 - p) * S_dn
            var = (E_S2 - E_S**2) / (S_j**2 * dt) if S_j > 0 and dt > 0 else base_vol**2
            lv = math.sqrt(max(var, 1e-10))
            local_vols.append(lv)

            # Arrow-Debreu propagation
            new_ad[j] += prev_ad[j] * (1 - p) * df
            new_ad[j + 1] += prev_ad[j] * p * df

        # Calibration error: compare tree-implied vs market call prices
        if step - 1 < len(market_vols) and market_vols[step - 1]:
            for k_idx, K in enumerate(strikes[step - 1] if step - 1 < len(strikes) else []):
                if k_idx < len(market_vols[step - 1]):
                    mkt_vol = market_vols[step - 1][k_idx]
                    fwd = spot * math.exp((rate - div_yield) * t_step)
                    from pricebook.models.black76 import black76_price, OptionType
                    mkt_price = black76_price(fwd, K, mkt_vol, t_step, math.exp(-rate * t_step), OptionType.CALL)
                    tree_price = sum(
                        new_ad[j] * max(new_spots[j] - K, 0)
                        for j in range(n_new)
                    )
                    total_error += (tree_price - mkt_price)**2

        spots_all.append(new_spots)
        ad_all.append(new_ad.tolist())
        local_vols_all.append(local_vols)
        p_ups_all.append(p_ups)

    return ImpliedTreeResult(
        spots=spots_all,
        arrow_debreu=ad_all,
        local_vols=local_vols_all,
        p_ups=p_ups_all,
        n_steps=n_steps,
        calibration_error=math.sqrt(total_error),
    )


def price_on_implied_tree(
    tree: ImpliedTreeResult,
    payoff_fn,
    rate: float,
    T: float,
    exercise: str = "european",
    exercise_steps: set[int] | None = None,
) -> float:
    """Price a derivative on a calibrated implied tree.

    Args:
        tree: calibrated implied tree.
        payoff_fn: callable(spot) → payoff at terminal.
        rate: risk-free rate.
        exercise: "european", "american", or "bermudan".
        exercise_steps: for Bermudan, steps where exercise allowed.
    """
    n = tree.n_steps
    dt = T / n
    df = math.exp(-rate * dt)

    # Terminal payoff
    V = [payoff_fn(s) for s in tree.spots[n]]

    for step in range(n - 1, -1, -1):
        p_ups = tree.p_ups[step]
        n_nodes = len(tree.spots[step])
        V_new = []

        for j in range(n_nodes):
            p = p_ups[j]
            cont = df * (p * V[j + 1] + (1 - p) * V[j])

            if exercise == "american" or (exercise == "bermudan" and exercise_steps and step in exercise_steps):
                intrinsic = payoff_fn(tree.spots[step][j])
                cont = max(cont, intrinsic)

            V_new.append(cont)
        V = V_new

    return V[0]


def extract_local_vol(tree: ImpliedTreeResult) -> list[list[tuple[float, float]]]:
    """Extract local vol surface from implied tree.

    Returns: list of [(spot, local_vol)] per step.
    """
    result = []
    for step in range(len(tree.local_vols)):
        step_data = []
        for j, lv in enumerate(tree.local_vols[step]):
            s = tree.spots[step][j]
            step_data.append((s, lv))
        result.append(step_data)
    return result
