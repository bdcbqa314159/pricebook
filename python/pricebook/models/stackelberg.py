"""Stackelberg leader-follower games: Cournot, Bertrand, credit markets.

* :func:`stackelberg_cournot` — quantity competition with leader advantage.
* :func:`stackelberg_bertrand` — price competition with leader advantage.
* :func:`credit_market_stackelberg` — lead bank sets spread, others follow.
* :func:`general_stackelberg` — generic two-player Stackelberg.

References:
    von Stackelberg, *Marktform und Gleichgewicht*, 1934.
    Tirole, *The Theory of Industrial Organization*, Ch. 8.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, minimize


@dataclass
class StackelbergResult:
    """Stackelberg equilibrium result."""
    leader_action: float
    follower_action: float
    leader_payoff: float
    follower_payoff: float
    total_surplus: float
    leader_advantage: float     # leader payoff − Cournot/Nash payoff

    def to_dict(self) -> dict:
        return vars(self)


def stackelberg_cournot(
    a: float = 100.0,
    b: float = 1.0,
    c_leader: float = 10.0,
    c_follower: float = 10.0,
) -> StackelbergResult:
    """Stackelberg-Cournot: quantity competition with leader advantage.

    Inverse demand: P = a − b(q_L + q_F).
    Costs: C_i = c_i × q_i.

    Follower best response: q_F(q_L) = (a − c_F − b×q_L) / (2b).
    Leader maximises knowing follower's BR.

    Args:
        a: demand intercept.
        b: demand slope.
        c_leader: leader marginal cost.
        c_follower: follower marginal cost.
    """
    # Follower best response
    def follower_br(q_L):
        q_F = (a - c_follower - b * q_L) / (2 * b)
        return max(q_F, 0)

    # Leader maximises: π_L = (a − b(q_L + q_F(q_L)) − c_L) × q_L
    def neg_leader_profit(q_L):
        q_F = follower_br(q_L)
        price = a - b * (q_L + q_F)
        return -(price - c_leader) * q_L

    result = minimize_scalar(neg_leader_profit, bounds=(0, a / b), method='bounded')
    q_L = result.x
    q_F = follower_br(q_L)
    price = a - b * (q_L + q_F)

    pi_L = (price - c_leader) * q_L
    pi_F = (price - c_follower) * q_F

    # Cournot (simultaneous) for comparison — asymmetric costs
    q1_cournot = (a - 2 * c_leader + c_follower) / (3 * b)
    q2_cournot = (a + c_leader - 2 * c_follower) / (3 * b)
    p_cournot = a - b * (q1_cournot + q2_cournot)
    pi_cournot = (p_cournot - c_leader) * q1_cournot

    return StackelbergResult(
        leader_action=q_L,
        follower_action=q_F,
        leader_payoff=pi_L,
        follower_payoff=pi_F,
        total_surplus=pi_L + pi_F,
        leader_advantage=pi_L - pi_cournot,
    )


def stackelberg_bertrand(
    a: float = 100.0,
    b: float = 2.0,
    d: float = 1.0,
    c_leader: float = 10.0,
    c_follower: float = 10.0,
) -> StackelbergResult:
    """Stackelberg-Bertrand: price competition with leader advantage.

    Demand: q_i = a − b×p_i + d×p_j (substitutes if d > 0).
    Leader sets price first, follower responds.

    Args:
        a: demand intercept.
        b: own-price sensitivity.
        d: cross-price sensitivity (substitutability).
        c_leader: leader marginal cost.
        c_follower: follower marginal cost.
    """
    # Follower BR: max (p_F - c_F) × (a - b×p_F + d×p_L)
    # FOC: a - 2b×p_F + d×p_L + b×c_F = 0
    # p_F(p_L) = (a + d×p_L + b×c_F) / (2b)
    def follower_br(p_L):
        return (a + d * p_L + b * c_follower) / (2 * b)

    # Leader maximises knowing BR
    def neg_leader_profit(p_L):
        p_F = follower_br(p_L)
        q_L = a - b * p_L + d * p_F
        return -(p_L - c_leader) * max(q_L, 0)

    result = minimize_scalar(neg_leader_profit, bounds=(c_leader, a / b * 2), method='bounded')
    p_L = result.x
    p_F = follower_br(p_L)
    q_L = max(a - b * p_L + d * p_F, 0)
    q_F = max(a - b * p_F + d * p_L, 0)

    pi_L = (p_L - c_leader) * q_L
    pi_F = (p_F - c_follower) * q_F

    return StackelbergResult(
        leader_action=p_L,
        follower_action=p_F,
        leader_payoff=pi_L,
        follower_payoff=pi_F,
        total_surplus=pi_L + pi_F,
        leader_advantage=0,  # no Cournot comparison for Bertrand
    )


def credit_market_stackelberg(
    risk_free_rate: float = 0.04,
    default_prob: float = 0.02,
    lgd: float = 0.40,
    funding_cost_leader: float = 0.001,
    funding_cost_follower: float = 0.003,
    demand_elasticity: float = 5.0,
    total_demand: float = 1_000_000_000,
) -> StackelbergResult:
    """Credit market Stackelberg: lead bank sets spread, others follow.

    The lead bank (lower funding cost) sets the lending spread.
    Follower banks respond with their best spread given the leader's.

    Borrower chooses the cheapest offer. If both offer, demand splits.

    Args:
        risk_free_rate: base rate.
        default_prob: annual PD.
        lgd: loss given default.
        funding_cost_leader: leader's funding spread.
        funding_cost_follower: follower's funding spread.
        demand_elasticity: how sensitive demand is to spread.
        total_demand: total loan demand in the market.
    """
    # Minimum breakeven spread
    be_leader = default_prob * lgd + funding_cost_leader
    be_follower = default_prob * lgd + funding_cost_follower

    # Follower BR: set spread just above breakeven but below leader + margin
    def follower_br(s_L):
        # Follower undercuts slightly if profitable, matches if not
        s_F = max(be_follower, s_L - 0.0005)
        return s_F

    # Leader maximises: profit = (spread - be) × demand_share × total_demand
    def neg_leader_profit(s_L):
        s_F = follower_br(s_L)
        # Market share: leader gets more if cheaper
        if s_L < s_F:
            share = max(0.0, min(1.0, 0.5 + demand_elasticity * (s_F - s_L)))
        else:
            share = max(0.0, min(1.0, 0.5 - demand_elasticity * (s_L - s_F)))
        profit = (s_L - be_leader) * share * total_demand
        return -profit

    result = minimize_scalar(neg_leader_profit,
                              bounds=(be_leader, be_leader + 0.05),
                              method='bounded')
    s_L = result.x
    s_F = follower_br(s_L)

    if s_L < s_F:
        share_L = min(0.8, 0.5 + demand_elasticity * (s_F - s_L))
    else:
        share_L = max(0.2, 0.5 - demand_elasticity * (s_L - s_F))

    pi_L = (s_L - be_leader) * share_L * total_demand
    pi_F = (s_F - be_follower) * (1 - share_L) * total_demand

    return StackelbergResult(
        leader_action=s_L,
        follower_action=s_F,
        leader_payoff=pi_L,
        follower_payoff=pi_F,
        total_surplus=pi_L + pi_F,
        leader_advantage=pi_L - pi_F,
    )


def general_stackelberg(
    leader_payoff_fn,
    follower_br_fn,
    leader_bounds: tuple[float, float] = (0, 100),
) -> StackelbergResult:
    """Generic Stackelberg: leader maximises given follower best response.

    Args:
        leader_payoff_fn: callable(leader_action, follower_action) → payoff.
        follower_br_fn: callable(leader_action) → follower_best_response.
        leader_bounds: search bounds for leader action.
    """
    def neg_obj(a_L):
        a_F = follower_br_fn(a_L)
        return -leader_payoff_fn(a_L, a_F)

    result = minimize_scalar(neg_obj, bounds=leader_bounds, method='bounded')
    a_L = result.x
    a_F = follower_br_fn(a_L)

    return StackelbergResult(
        leader_action=a_L,
        follower_action=a_F,
        leader_payoff=leader_payoff_fn(a_L, a_F),
        follower_payoff=0,  # generic — user computes
        total_surplus=0,
        leader_advantage=0,
    )
