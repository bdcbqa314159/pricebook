"""Balance sheet allocation — capital-efficient trade selection.

    from pricebook.regulatory.balance_sheet_allocation import (
        rank_by_roc, optimise_allocation, TradeROC,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class TradeROC:
    """Return on capital for a single repo trade."""
    trade_id: str
    carry: float                 # annual carry
    xva_cost: float              # annual XVA cost
    net_income: float            # carry - xva
    rwa: float                   # risk-weighted assets
    capital_required: float      # rwa × 8%
    roc: float                   # net_income / capital (annualised)

    def to_dict(self) -> dict:
        return vars(self)


def rank_by_roc(trades: list[dict]) -> list[TradeROC]:
    """Rank trades by return on capital.

    Args:
        trades: list of {trade_id, carry, xva_cost, rwa}.

    Returns sorted by ROC (highest first).
    """
    results = []
    for t in trades:
        carry = t["carry"]
        xva = t.get("xva_cost", 0.0)
        net = carry - xva
        rwa = t["rwa"]
        cap = rwa * 0.08
        roc = net / cap if cap > 0 else 0.0
        results.append(TradeROC(
            trade_id=t["trade_id"],
            carry=carry, xva_cost=xva, net_income=net,
            rwa=rwa, capital_required=cap, roc=roc,
        ))
    return sorted(results, key=lambda r: -r.roc)


@dataclass
class AllocationResult:
    """Result of balance sheet allocation optimization."""
    allocations: dict[str, float]  # {trade_id: notional}
    total_roc: float
    total_rwa: float
    total_capital_used: float
    capital_utilisation_pct: float
    n_trades_selected: int

    def to_dict(self) -> dict:
        return vars(self)


def optimise_allocation(
    trades: list[dict],
    total_capital: float,
    max_rwa: float | None = None,
    max_single_trade_pct: float = 0.25,
) -> AllocationResult:
    """Optimize trade allocation to maximize total ROC.

    LP: maximize Σ roc_i × w_i
    s.t. Σ rwa_i × w_i × 8% ≤ total_capital
         w_i ≤ max_single × total_notional
         w_i ≥ 0

    Args:
        trades: list of {trade_id, carry, xva_cost, rwa, max_notional}.
        total_capital: capital budget.
        max_rwa: RWA limit (default = capital / 8%).
        max_single_trade_pct: concentration limit.
    """
    n = len(trades)
    if n == 0:
        return AllocationResult({}, 0.0, 0.0, 0.0, 0.0, 0)

    if max_rwa is None:
        max_rwa = total_capital / 0.08

    # Objective: maximize Σ net_i × w_i → minimize -net_i × w_i
    c = [-(t["carry"] - t.get("xva_cost", 0)) for t in trades]

    A_ub = []
    b_ub = []

    # Capital constraint: Σ rwa_i × w_i × 8% ≤ total_capital
    A_ub.append([t["rwa"] * 0.08 for t in trades])
    b_ub.append(total_capital)

    # RWA constraint
    A_ub.append([t["rwa"] for t in trades])
    b_ub.append(max_rwa)

    # Fix T4-REG: ``max_single_trade_pct`` was declared in the signature
    # but never referenced — allocations could concentrate up to the
    # ``max_notional`` per-trade ceiling, so a single trade could absorb
    # the entire capital budget despite the documented concentration
    # limit.  Now enforced as a per-trade capital cap:
    #     rwa_i × w_i × 0.08 ≤ max_single_trade_pct × total_capital
    # collapsed into the upper bound on w_i.
    bounds = []
    for t in trades:
        ub_notional = t.get("max_notional", total_capital * 10)
        cap_per_trade = max_single_trade_pct * total_capital
        rwa_i = t["rwa"]
        if rwa_i > 0:
            ub_concentration = cap_per_trade / (rwa_i * 0.08)
            bounds.append((0, min(ub_notional, ub_concentration)))
        else:
            bounds.append((0, ub_notional))

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if result.success:
        allocs = {trades[i]["trade_id"]: float(result.x[i]) for i in range(n) if result.x[i] > 1e-6}
        total_net = -result.fun
        total_rwa_used = sum(trades[i]["rwa"] * result.x[i] for i in range(n))
        total_cap_used = total_rwa_used * 0.08
    else:
        allocs = {}
        total_net = 0.0
        total_rwa_used = 0.0
        total_cap_used = 0.0

    return AllocationResult(
        allocations=allocs,
        total_roc=total_net / max(total_cap_used, 1e-10),
        total_rwa=total_rwa_used,
        total_capital_used=total_cap_used,
        capital_utilisation_pct=total_cap_used / total_capital * 100 if total_capital > 0 else 0,
        n_trades_selected=len(allocs),
    )
