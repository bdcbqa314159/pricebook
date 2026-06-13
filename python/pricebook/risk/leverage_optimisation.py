"""Leverage optimization for hedge fund repo portfolios.

LP: maximize carry subject to haircut + capital + concentration constraints.

    from pricebook.risk.leverage_optimisation import (
        optimise_leverage, leverage_frontier, LeverageOptResult,
    )

References:
    Ang, Gorovyy & van Inwegen (2011). Hedge Fund Leverage. JFE.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class LeverageOptResult:
    """Result of leverage optimization."""
    optimal_weights: list[float]     # fraction of capital per trade
    optimal_carry: float             # maximised total carry
    leverage_ratio: float            # total notional / capital
    binding_constraints: list[str]   # which constraints are active
    n_trades: int

    def to_dict(self) -> dict:
        return {
            "optimal_weights": self.optimal_weights,
            "optimal_carry": self.optimal_carry,
            "leverage_ratio": self.leverage_ratio,
            "binding_constraints": self.binding_constraints,
            "n_trades": self.n_trades,
        }


def optimise_leverage(
    trade_carries: list[float],
    trade_haircuts: list[float],
    trade_rwa_weights: list[float],
    capital: float,
    max_leverage: float = 10.0,
    max_single_trade_pct: float = 1.0,
    capital_ratio_floor: float = 0.08,
) -> LeverageOptResult:
    """Maximize carry subject to constraints via LP.

    Decision variable: w_i = notional allocated to trade i.

    max  Σ carry_i × w_i
    s.t. Σ haircut_i × w_i ≤ capital           (haircut constraint)
         Σ w_i ≤ capital × max_leverage         (leverage cap)
         w_i ≤ max_single_trade × Σw            (concentration)
         Σ rwa_i × w_i × capital_ratio ≤ capital (capital adequacy)
         w_i ≥ 0

    Args:
        trade_carries: annual carry per unit notional for each trade.
        trade_haircuts: haircut fraction per trade.
        trade_rwa_weights: RWA weight per trade (e.g. 0.20 for sovereign).
        capital: total equity capital available.
        max_leverage: maximum gross leverage ratio.
        max_single_trade_pct: max fraction of portfolio in one trade.
        capital_ratio_floor: minimum capital / RWA ratio.
    """
    n = len(trade_carries)
    if n == 0:
        return LeverageOptResult([], 0.0, 0.0, [], 0)

    # Objective: maximize Σ carry_i × w_i → minimize -carry
    c = [-carry for carry in trade_carries]

    # Constraints (Ax ≤ b)
    A_ub = []
    b_ub = []
    constraint_names = []

    # 1. Haircut constraint: Σ haircut_i × w_i ≤ capital
    A_ub.append(trade_haircuts)
    b_ub.append(capital)
    constraint_names.append("haircut")

    # 2. Leverage cap: Σ w_i ≤ capital × max_leverage
    A_ub.append([1.0] * n)
    b_ub.append(capital * max_leverage)
    constraint_names.append("leverage")

    # 3. Capital adequacy: Σ rwa_i × w_i × capital_ratio_floor ≤ capital
    A_ub.append([rwa * capital_ratio_floor for rwa in trade_rwa_weights])
    b_ub.append(capital)
    constraint_names.append("capital")

    # 4. Concentration: w_i ≤ max_single_trade_pct × Σw (relative).
    # Fix T4-RISK24: pre-fix used the ABSOLUTE form
    # ``w_i ≤ max_single × capital × max_leverage`` — looser than
    # the docstring-promised relative form whenever actual leverage
    # is less than max_leverage.  Example: at capital=$100M,
    # max_lev=10, max_single=0.30, actual_lev=5x ($500M notional),
    # pre-fix allowed one trade up to $300M (60% of portfolio);
    # the relative form caps it at $150M (30% of portfolio).
    # The relative constraint is linear: w_i - max_pct·Σw ≤ 0
    # ⇒ (1 - max_pct)·w_i - max_pct·Σ_{j≠i} w_j ≤ 0.
    for i in range(n):
        row = [-max_single_trade_pct] * n
        row[i] = 1.0 - max_single_trade_pct
        A_ub.append(row)
        b_ub.append(0.0)
    constraint_names.extend([f"concentration_{i}" for i in range(n)])

    bounds = [(0, None) for _ in range(n)]

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        weights = list(result.x)
        total_carry = -result.fun
        total_notional = sum(weights)
        lev = total_notional / capital if capital > 0 else 0.0

        # Identify binding constraints
        binding = []
        slack = np.array(b_ub) - np.array(A_ub) @ np.array(weights)
        for i, (name, s) in enumerate(zip(constraint_names, slack)):
            if abs(s) < 1e-6:
                binding.append(name)
    else:
        weights = [0.0] * n
        total_carry = 0.0
        lev = 0.0
        binding = ["infeasible"]

    return LeverageOptResult(
        optimal_weights=weights,
        optimal_carry=total_carry,
        leverage_ratio=lev,
        binding_constraints=binding,
        n_trades=n,
    )


def leverage_frontier(
    trade_carries: list[float],
    trade_haircuts: list[float],
    trade_rwa_weights: list[float],
    capital: float,
    leverage_range: list[float] | None = None,
) -> list[dict]:
    """Efficient frontier of carry vs leverage.

    Sweeps max_leverage from 1× to 20× and computes optimal carry at each.
    """
    if leverage_range is None:
        leverage_range = [1, 2, 3, 5, 7, 10, 15, 20]

    frontier = []
    for lev in leverage_range:
        result = optimise_leverage(
            trade_carries, trade_haircuts, trade_rwa_weights,
            capital, max_leverage=lev,
        )
        frontier.append({
            "max_leverage": lev,
            "optimal_carry": result.optimal_carry,
            "actual_leverage": result.leverage_ratio,
            "binding": result.binding_constraints,
        })
    return frontier
