"""Bargaining theory: Nash bargaining, Rubinstein alternating offers.

* :func:`nash_bargaining` — Nash bargaining solution.
* :func:`rubinstein_alternating` — Rubinstein equilibrium.
* :func:`kalai_smorodinsky` — Kalai-Smorodinsky solution.
* :func:`debt_restructuring_bargain` — creditor-debtor bargaining.

References:
    Nash, *The Bargaining Problem*, Econometrica, 1950.
    Rubinstein, *Perfect Equilibrium in a Bargaining Model*, Ecta, 1982.
    Kalai & Smorodinsky, *Other Solutions to Nash's Bargaining Problem*, Ecta, 1975.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class BargainingResult:
    """Bargaining solution result."""
    payoff_1: float
    payoff_2: float
    surplus_split: float        # fraction of surplus to player 1
    total_surplus: float
    disagreement_1: float
    disagreement_2: float
    method: str

    def to_dict(self) -> dict:
        return vars(self)


def nash_bargaining(
    feasible_set: np.ndarray,
    disagreement: tuple[float, float] = (0.0, 0.0),
    bargaining_power: tuple[float, float] = (0.5, 0.5),
) -> BargainingResult:
    """Nash bargaining solution.

    Maximises: (u₁ − d₁)^α × (u₂ − d₂)^(1−α)

    over the feasible set, where d = disagreement point.

    For linear feasible set u₁ + u₂ = S:
    u₁* = d₁ + α(S − d₁ − d₂)

    Args:
        feasible_set: (K, 2) array of feasible (u₁, u₂) pairs.
        disagreement: (d₁, d₂) payoffs if no agreement.
        bargaining_power: (α, 1−α) relative bargaining power.
    """
    d1, d2 = disagreement
    alpha = bargaining_power[0] / sum(bargaining_power)

    # Filter feasible set to individually rational points
    ir = feasible_set[(feasible_set[:, 0] >= d1) & (feasible_set[:, 1] >= d2)]
    if len(ir) == 0:
        return BargainingResult(d1, d2, 0.5, 0, d1, d2, "nash_bargaining")

    # Maximise Nash product
    best_idx = -1
    best_product = -1
    for i in range(len(ir)):
        u1, u2 = ir[i]
        product = (u1 - d1) ** alpha * (u2 - d2) ** (1 - alpha)
        if product > best_product:
            best_product = product
            best_idx = i

    u1_star, u2_star = ir[best_idx]
    total_surplus = (u1_star - d1) + (u2_star - d2)
    split = (u1_star - d1) / total_surplus if total_surplus > 0 else 0.5

    return BargainingResult(u1_star, u2_star, split, total_surplus, d1, d2, "nash_bargaining")


def rubinstein_alternating(
    total_surplus: float,
    discount_1: float = 0.95,
    discount_2: float = 0.95,
    n_rounds: int = 100,
) -> BargainingResult:
    """Rubinstein alternating-offer bargaining.

    Unique SPE: player 1 offers x₁ = (1 − δ₂)/(1 − δ₁δ₂) × S,
    player 2 accepts immediately.

    More patient player (higher δ) gets more surplus.

    Args:
        total_surplus: total pie to divide.
        discount_1: player 1's discount factor (patience).
        discount_2: player 2's discount factor.
        n_rounds: max rounds (for finite version).
    """
    d1, d2 = discount_1, discount_2

    # Infinite-horizon SPE
    if d1 * d2 < 1:
        share_1 = (1 - d2) / (1 - d1 * d2)
    else:
        share_1 = 0.5  # equal split if both perfectly patient

    u1 = share_1 * total_surplus
    u2 = (1 - share_1) * total_surplus

    return BargainingResult(u1, u2, share_1, total_surplus, 0, 0, "rubinstein")


def kalai_smorodinsky(
    feasible_set: np.ndarray,
    disagreement: tuple[float, float] = (0.0, 0.0),
) -> BargainingResult:
    """Kalai-Smorodinsky solution.

    The solution lies on the line from d to the ideal point
    (max feasible u₁, max feasible u₂), at the Pareto frontier.

    Unlike Nash, this satisfies monotonicity: if the feasible set
    expands in player i's favour, player i gets at least as much.

    Args:
        feasible_set: (K, 2) feasible payoff pairs.
        disagreement: disagreement point.
    """
    d1, d2 = disagreement

    # Ideal point: max of each player independently
    ir = feasible_set[(feasible_set[:, 0] >= d1) & (feasible_set[:, 1] >= d2)]
    if len(ir) == 0:
        return BargainingResult(d1, d2, 0.5, 0, d1, d2, "kalai_smorodinsky")

    ideal_1 = float(np.max(ir[:, 0]))
    ideal_2 = float(np.max(ir[:, 1]))

    # Slope from d to ideal
    if ideal_1 - d1 > 0 and ideal_2 - d2 > 0:
        slope = (ideal_2 - d2) / (ideal_1 - d1)
    else:
        # Degenerate: split equally
        best = ir[np.argmax(ir[:, 0] + ir[:, 1])]
        return BargainingResult(float(best[0]), float(best[1]), 0.5, 0, d1, d2, "kalai_smorodinsky")

    # Find intersection of line d→ideal with Pareto frontier
    # Line: u₂ = d₂ + slope × (u₁ − d₁)
    best_idx = -1
    best_dist = -1
    for i in range(len(ir)):
        u1, u2 = ir[i]
        # Distance along the line
        dist = u1 - d1
        expected_u2 = d2 + slope * dist
        if abs(u2 - expected_u2) < abs(u2 - expected_u2) * 0.1 + 0.01:
            if dist > best_dist:
                best_dist = dist
                best_idx = i

    if best_idx >= 0:
        u1_star, u2_star = ir[best_idx]
    else:
        # Closest point to the line
        distances = np.abs(ir[:, 1] - (d2 + slope * (ir[:, 0] - d1)))
        best_idx = np.argmin(distances)
        u1_star, u2_star = ir[best_idx]

    total = (u1_star - d1) + (u2_star - d2)
    split = (u1_star - d1) / total if total > 0 else 0.5

    return BargainingResult(float(u1_star), float(u2_star), split, total, d1, d2, "kalai_smorodinsky")


def debt_restructuring_bargain(
    firm_value: float,
    debt_face: float,
    creditor_recovery_rate: float = 0.40,
    debtor_discount: float = 0.90,
    creditor_discount: float = 0.95,
) -> BargainingResult:
    """Creditor-debtor restructuring bargaining.

    Disagreement: creditor gets recovery × face, debtor gets nothing.
    Surplus: firm_value − recovery × face (saved from bankruptcy costs).

    More patient creditor (higher discount) extracts more surplus.

    Args:
        firm_value: going-concern value of the firm.
        debt_face: face value of debt.
        creditor_recovery_rate: recovery in liquidation.
        debtor_discount: debtor's patience (lower = more desperate).
        creditor_discount: creditor's patience.
    """
    liquidation_value = creditor_recovery_rate * debt_face
    surplus = max(firm_value - liquidation_value, 0)

    # Rubinstein split of the surplus
    r = rubinstein_alternating(surplus, debtor_discount, creditor_discount)

    debtor_payoff = r.payoff_1  # debtor is player 1 (proposes first)
    creditor_payoff = liquidation_value + r.payoff_2

    return BargainingResult(
        payoff_1=debtor_payoff,          # debtor's equity value
        payoff_2=creditor_payoff,        # creditor's recovery
        surplus_split=r.surplus_split,
        total_surplus=surplus,
        disagreement_1=0,                # debtor gets 0 in liquidation
        disagreement_2=liquidation_value,
        method="debt_restructuring",
    )
