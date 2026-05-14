"""Tree improvements: Greeks from trees, 2D binomial.

    from pricebook.numerical import tree_greeks, binomial_2d
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class TreeGreeks:
    """Greeks computed from a binomial/trinomial tree."""
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


def tree_greeks(
    tree_pricer,
    spot: float,
    rate: float,
    vol: float,
    T: float,
    strike: float,
    n_steps: int = 200,
    is_call: bool = True,
    is_american: bool = False,
    div_yield: float = 0.0,
    vol_bump: float = 0.01,
) -> TreeGreeks:
    """Compute Greeks from any tree pricer via the tree structure.

    Delta and gamma from nodes at t=1, theta from t=2 → t=0 comparison.
    Vega via bump-and-reprice.

    Args:
        tree_pricer: callable(spot, strike, rate, vol, T, n_steps, is_call, is_american, div_yield) → price.
    """
    # Base price
    price = tree_pricer(spot, strike, rate, vol, T, n_steps, is_call, is_american, div_yield)

    # Delta & gamma: bump spot
    ds = spot * 0.01
    p_up = tree_pricer(spot + ds, strike, rate, vol, T, n_steps, is_call, is_american, div_yield)
    p_dn = tree_pricer(spot - ds, strike, rate, vol, T, n_steps, is_call, is_american, div_yield)
    delta = (p_up - p_dn) / (2 * ds)
    gamma = (p_up - 2 * price + p_dn) / (ds ** 2)

    # Theta: shift T by 1 day
    dt = 1.0 / 365
    if T > dt:
        p_theta = tree_pricer(spot, strike, rate, vol, T - dt, n_steps, is_call, is_american, div_yield)
        theta = (p_theta - price) / dt
    else:
        theta = 0.0

    # Vega: bump vol
    p_vega = tree_pricer(spot, strike, rate, vol + vol_bump, T, n_steps, is_call, is_american, div_yield)
    vega = (p_vega - price) / vol_bump

    return TreeGreeks(price=price, delta=delta, gamma=gamma, theta=theta, vega=vega)


@dataclass
class Binomial2DResult:
    """Two-asset binomial tree result."""
    price: float
    n_steps: int

    def to_dict(self) -> dict:
        return vars(self)


def binomial_2d(
    S1: float,
    S2: float,
    strike: float,
    rate: float,
    vol1: float,
    vol2: float,
    rho: float,
    T: float,
    n_steps: int = 50,
    payoff_type: str = "spread_call",
    div_yield1: float = 0.0,
    div_yield2: float = 0.0,
) -> Binomial2DResult:
    """Two-asset binomial tree (Rubinstein 1994).

    Prices options on two correlated assets via a recombining 2D tree.
    Each step has 4 branches: (up-up, up-down, down-up, down-down).

    Args:
        payoff_type: 'spread_call' (max(S1-S2-K,0)), 'spread_put',
                     'best_of_call' (max(max(S1,S2)-K,0)),
                     'worst_of_call' (max(min(S1,S2)-K,0)).
    """
    dt = T / n_steps
    u1 = math.exp(vol1 * math.sqrt(dt))
    d1 = 1.0 / u1
    u2 = math.exp(vol2 * math.sqrt(dt))
    d2 = 1.0 / u2

    # Risk-neutral probabilities for 2D tree
    mu1 = (rate - div_yield1 - 0.5 * vol1 ** 2) * dt
    mu2 = (rate - div_yield2 - 0.5 * vol2 ** 2) * dt

    p_uu = 0.25 * (1 + rho + (mu1 / (vol1 * math.sqrt(dt))) + (mu2 / (vol2 * math.sqrt(dt))))
    p_ud = 0.25 * (1 - rho + (mu1 / (vol1 * math.sqrt(dt))) - (mu2 / (vol2 * math.sqrt(dt))))
    p_du = 0.25 * (1 - rho - (mu1 / (vol1 * math.sqrt(dt))) + (mu2 / (vol2 * math.sqrt(dt))))
    p_dd = 1.0 - p_uu - p_ud - p_du

    # Clamp probabilities
    p_uu = max(p_uu, 0.0)
    p_ud = max(p_ud, 0.0)
    p_du = max(p_du, 0.0)
    p_dd = max(p_dd, 0.0)
    p_total = p_uu + p_ud + p_du + p_dd
    if p_total > 0:
        p_uu /= p_total
        p_ud /= p_total
        p_du /= p_total
        p_dd /= p_total

    df = math.exp(-rate * dt)

    # Terminal values
    n = n_steps
    values = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(n + 1):
            s1 = S1 * u1 ** (n - i) * d1 ** i
            s2 = S2 * u2 ** (n - j) * d2 ** j

            if payoff_type == "spread_call":
                values[i, j] = max(s1 - s2 - strike, 0)
            elif payoff_type == "spread_put":
                values[i, j] = max(strike - (s1 - s2), 0)
            elif payoff_type == "best_of_call":
                values[i, j] = max(max(s1, s2) - strike, 0)
            elif payoff_type == "worst_of_call":
                values[i, j] = max(min(s1, s2) - strike, 0)

    # Backward induction
    for step in range(n - 1, -1, -1):
        new_values = np.zeros((step + 1, step + 1))
        for i in range(step + 1):
            for j in range(step + 1):
                new_values[i, j] = df * (
                    p_uu * values[i, j] +
                    p_ud * values[i, j + 1] +
                    p_du * values[i + 1, j] +
                    p_dd * values[i + 1, j + 1]
                )
        values = new_values

    return Binomial2DResult(price=float(values[0, 0]), n_steps=n_steps)
