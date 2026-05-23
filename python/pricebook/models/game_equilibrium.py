"""Nash equilibrium and market microstructure games.

    from pricebook.models.game_equilibrium import (
        nash_2player, market_maker_equilibrium, optimal_execution_game,
    )

References:
    Nash (1950). Equilibrium Points in N-Person Games. PNAS.
    Avellaneda & Stoikov (2008). High-Frequency Trading in a Limit Order Book.
    Almgren & Chriss (2001). Optimal Execution of Portfolio Transactions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NashResult:
    """Result of Nash equilibrium computation."""
    strategy_a: np.ndarray       # mixed strategy for player A
    strategy_b: np.ndarray       # mixed strategy for player B
    value_a: float               # expected payoff for A
    value_b: float               # expected payoff for B
    is_pure: bool                # True if equilibrium is in pure strategies
    n_support_a: int             # number of strategies with positive probability
    n_support_b: int

    def to_dict(self) -> dict:
        return {
            "strategy_a": self.strategy_a.tolist(),
            "strategy_b": self.strategy_b.tolist(),
            "value_a": self.value_a,
            "value_b": self.value_b,
            "is_pure": self.is_pure,
        }


def nash_2player(
    payoff_a: np.ndarray,
    payoff_b: np.ndarray,
) -> NashResult:
    """Find Nash equilibrium of a 2-player bimatrix game.

    Uses support enumeration: for each pair of supports (S_A, S_B),
    solve the indifference conditions.

    Args:
        payoff_a: (m, n) payoff matrix for player A.
        payoff_b: (m, n) payoff matrix for player B.

    Returns:
        NashResult with mixed strategies and payoffs.
    """
    A = np.asarray(payoff_a, dtype=float)
    B = np.asarray(payoff_b, dtype=float)
    m, n = A.shape

    best_result = None
    best_value = -np.inf

    # Try all support pairs
    from itertools import combinations

    for sa_size in range(1, m + 1):
        for sb_size in range(1, n + 1):
            for sa in combinations(range(m), sa_size):
                for sb in combinations(range(n), sb_size):
                    result = _solve_support(A, B, list(sa), list(sb))
                    if result is not None:
                        val = result.value_a + result.value_b
                        if val > best_value:
                            best_value = val
                            best_result = result

    if best_result is None:
        # Fallback: uniform mixed strategy
        p = np.ones(m) / m
        q = np.ones(n) / n
        return NashResult(p, q, float(p @ A @ q), float(p @ B @ q), False, m, n)

    return best_result


def _solve_support(A, B, sa, sb):
    """Solve indifference conditions for given supports."""
    m, n = A.shape
    sa_size = len(sa)
    sb_size = len(sb)

    if sa_size != sb_size and sa_size > 1 and sb_size > 1:
        return None

    try:
        # Player B's strategy makes A indifferent over sa
        # A[i, :] @ q = A[j, :] @ q for all i, j in sa
        # Plus: sum(q[sb]) = 1
        if sb_size == 1:
            q = np.zeros(n)
            q[sb[0]] = 1.0
        else:
            sub_A = A[np.ix_(sa, sb)]
            # Indifference: (sub_A[0] - sub_A[i]) @ q_sub = 0 for i > 0
            # Plus sum = 1
            n_eq = sa_size - 1 + 1
            M = np.zeros((n_eq, sb_size))
            rhs = np.zeros(n_eq)
            for i in range(sa_size - 1):
                M[i] = sub_A[0] - sub_A[i + 1]
            M[-1] = 1.0
            rhs[-1] = 1.0
            q_sub = np.linalg.lstsq(M, rhs, rcond=None)[0]
            if np.any(q_sub < -1e-10):
                return None
            q_sub = np.maximum(q_sub, 0)
            q_sub /= q_sub.sum()
            q = np.zeros(n)
            for i, idx in enumerate(sb):
                q[idx] = q_sub[i]

        # Player A's strategy
        if sa_size == 1:
            p = np.zeros(m)
            p[sa[0]] = 1.0
        else:
            sub_B = B[np.ix_(sa, sb)]
            n_eq = sb_size - 1 + 1
            M = np.zeros((n_eq, sa_size))
            rhs = np.zeros(n_eq)
            for j in range(sb_size - 1):
                M[j] = sub_B[:, 0] - sub_B[:, j + 1]
            M[-1] = 1.0
            rhs[-1] = 1.0
            p_sub = np.linalg.lstsq(M, rhs, rcond=None)[0]
            if np.any(p_sub < -1e-10):
                return None
            p_sub = np.maximum(p_sub, 0)
            p_sub /= p_sub.sum()
            p = np.zeros(m)
            for i, idx in enumerate(sa):
                p[idx] = p_sub[i]

        val_a = float(p @ A @ q)
        val_b = float(p @ B @ q)
        is_pure = (np.count_nonzero(p > 1e-10) == 1 and np.count_nonzero(q > 1e-10) == 1)

        return NashResult(p, q, val_a, val_b, is_pure,
                          int(np.count_nonzero(p > 1e-10)),
                          int(np.count_nonzero(q > 1e-10)))
    except (np.linalg.LinAlgError, ValueError):
        return None


@dataclass
class MarketMakerResult:
    """Result of market maker equilibrium."""
    optimal_spread: float        # optimal bid-ask spread
    optimal_bid: float
    optimal_ask: float
    expected_pnl: float
    inventory_risk: float

    def to_dict(self) -> dict:
        return vars(self)


def market_maker_equilibrium(
    mid_price: float,
    inventory: float,
    volatility: float,
    arrival_rate: float = 1.0,
    risk_aversion: float = 0.01,
    T: float = 1.0,
) -> MarketMakerResult:
    """Optimal market-making spread (Avellaneda-Stoikov 2008).

    Optimal spread = γσ²T + (2/γ)ln(1 + γ/κ)

    where γ = risk aversion, σ = vol, T = horizon, κ = arrival intensity.
    Inventory adjustment: reservation price = mid - γσ²q(T-t).

    Args:
        mid_price: current mid price.
        inventory: current inventory (positive = long).
        volatility: price volatility.
        arrival_rate: order arrival intensity (κ).
        risk_aversion: risk aversion parameter (γ).
        T: time horizon.
    """
    import math

    gamma = risk_aversion
    sigma = volatility

    # Reservation price (inventory-adjusted mid)
    reservation = mid_price - gamma * sigma**2 * inventory * T

    # Optimal spread
    spread = gamma * sigma**2 * T + (2 / gamma) * math.log(1 + gamma / arrival_rate)

    half_spread = spread / 2
    bid = reservation - half_spread
    ask = reservation + half_spread

    # Expected PnL: spread capture - inventory risk
    expected_pnl = spread * arrival_rate * T - 0.5 * gamma * sigma**2 * inventory**2 * T

    return MarketMakerResult(
        optimal_spread=spread,
        optimal_bid=bid,
        optimal_ask=ask,
        expected_pnl=expected_pnl,
        inventory_risk=gamma * sigma**2 * inventory**2 * T,
    )


@dataclass
class ExecutionResult:
    """Result of optimal execution computation."""
    trade_schedule: list[float]  # notional per period
    expected_cost: float         # total expected execution cost
    risk: float                  # execution risk (timing variance)
    urgency: float               # fraction executed in first period

    def to_dict(self) -> dict:
        return vars(self)


def optimal_execution_game(
    total_shares: float,
    n_periods: int,
    volatility: float,
    market_impact: float,
    risk_aversion: float = 0.01,
) -> ExecutionResult:
    """Almgren-Chriss optimal execution with linear market impact.

    Minimise: E[cost] + λ × Var[cost]
    where cost = Σ impact(xᵢ) + Σ vol_risk(remaining)

    Optimal: exponentially front-loaded schedule.

    Args:
        total_shares: total position to liquidate.
        n_periods: number of trading periods.
        volatility: per-period price volatility.
        market_impact: linear impact coefficient (η).
        risk_aversion: trade-off between cost and risk (λ).
    """
    import math

    # Almgren-Chriss: x_k = X × sinh(κ(T-k)) / sinh(κT)
    # where κ = acosh(1 + λσ²/(2η))
    lam = risk_aversion
    sigma = volatility
    eta = market_impact

    kappa_arg = 1 + lam * sigma**2 / (2 * eta)
    kappa = math.acosh(max(kappa_arg, 1.0))

    T = n_periods
    schedule = []
    for k in range(T):
        if kappa * T > 0:
            x_k = total_shares * math.sinh(kappa * (T - k)) / math.sinh(kappa * T)
        else:
            x_k = total_shares / T
        schedule.append(float(x_k))

    # Normalise to exact total
    s = sum(schedule)
    if s > 0:
        schedule = [x * total_shares / s for x in schedule]

    # Costs
    expected_cost = sum(eta * x**2 for x in schedule)
    remaining = [total_shares - sum(schedule[:k]) for k in range(T)]
    risk = sum(sigma**2 * r**2 for r in remaining)
    urgency = schedule[0] / total_shares if total_shares > 0 else 0

    return ExecutionResult(
        trade_schedule=schedule,
        expected_cost=expected_cost,
        risk=risk,
        urgency=urgency,
    )
