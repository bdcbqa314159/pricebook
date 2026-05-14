"""Capital allocation & RORC: Euler allocation, hurdle rates, optimisation.

Allocate portfolio capital to desks, compute return on regulatory capital,
and monitor capital limits.

    from pricebook.regulatory.capital_allocation import (
        allocate_and_report, euler_allocation, capital_limit_monitor,
    )

References:
    Tasche (2008). Capital Allocation to Business Units and Sub-Portfolios.
    McNeil, Frey & Embrechts (2015). Quantitative Risk Management, Ch. 8.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ═══════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════

@dataclass
class DeskCapitalInput:
    """Input for capital allocation: one desk's standalone capital and P&L."""
    desk_id: str
    standalone_capital: float
    pnl_annual: float
    rwa: float = 0.0
    revenue: float = 0.0
    costs: float = 0.0

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class DeskAllocation:
    """Allocated capital and performance for one desk."""
    desk_id: str
    standalone_capital: float
    allocated_capital: float
    allocation_pct: float
    pnl: float
    rorc: float
    exceeds_hurdle: bool
    excess_return: float
    economic_profit: float

    def to_dict(self) -> dict:
        return vars(self)


@dataclass
class CapitalAllocationResult:
    """Full capital allocation report."""
    total_portfolio_capital: float
    sum_standalone: float
    diversification_benefit: float
    diversification_ratio: float
    method: str
    hurdle_rate: float
    desk_allocations: list[DeskAllocation]
    portfolio_rorc: float
    n_desks_above_hurdle: int
    n_desks_below_hurdle: int
    worst_rorc_desk: str
    best_rorc_desk: str

    def to_dict(self) -> dict:
        return {
            "total_portfolio_capital": self.total_portfolio_capital,
            "sum_standalone": self.sum_standalone,
            "diversification_benefit": self.diversification_benefit,
            "diversification_ratio": self.diversification_ratio,
            "method": self.method,
            "hurdle_rate": self.hurdle_rate,
            "portfolio_rorc": self.portfolio_rorc,
            "n_desks_above_hurdle": self.n_desks_above_hurdle,
            "n_desks_below_hurdle": self.n_desks_below_hurdle,
            "worst_rorc_desk": self.worst_rorc_desk,
            "best_rorc_desk": self.best_rorc_desk,
            "desk_allocations": [d.to_dict() for d in self.desk_allocations],
        }


# ═══════════════════════════════════════════════════════════════
# Allocation Methods
# ═══════════════════════════════════════════════════════════════

def euler_allocation(
    desk_inputs: list[DeskCapitalInput],
    portfolio_capital: float | None = None,
    correlation_matrix: np.ndarray | None = None,
) -> list[float]:
    """Euler (risk contribution) allocation.

    If correlation_matrix provided:
        RC_i = (Sigma @ w)_i × w_i / portfolio_vol
    Otherwise: proportional to standalone capital.

    Properties: sum(allocated) = portfolio_capital.

    Args:
        desk_inputs: per-desk standalone capital.
        portfolio_capital: total capital to allocate. If None, computed
            from correlation matrix or sum of standalone × 0.85 (15% diversification).
        correlation_matrix: (n×n) desk-level correlation.

    Returns:
        List of allocated capital amounts (same order as desk_inputs).
    """
    n = len(desk_inputs)
    if n == 0:
        return []

    standalones = np.array([d.standalone_capital for d in desk_inputs])
    total_standalone = float(np.sum(standalones))

    if total_standalone <= 0:
        return [0.0] * n

    if correlation_matrix is not None:
        corr = np.asarray(correlation_matrix)
        # Covariance: Sigma_ij = s_i × s_j × rho_ij (treat standalone as std dev proxy)
        cov = np.outer(standalones, standalones) * corr
        # Portfolio variance: w' Sigma w where w_i = s_i / sum(s)
        w = standalones / total_standalone
        portfolio_var = float(w @ cov @ w)
        portfolio_vol = float(np.sqrt(max(portfolio_var, 0.0)))

        if portfolio_capital is None:
            # Diversified capital = sqrt(sum_ij s_i * s_j * rho_ij)
            portfolio_capital = float(np.sqrt(max(float(standalones @ corr @ standalones), 0.0)))
        marginal = cov @ w
        risk_contributions = marginal * w
        rc_total = float(np.sum(risk_contributions))

        if rc_total > 0:
            allocated = [float(rc / rc_total * portfolio_capital) for rc in risk_contributions]
        else:
            allocated = [float(s / total_standalone * portfolio_capital) for s in standalones]
    else:
        if portfolio_capital is None:
            portfolio_capital = total_standalone * 0.85  # 15% diversification assumption

        # Pro-rata by standalone
        allocated = [float(s / total_standalone * portfolio_capital) for s in standalones]

    return allocated


def pro_rata_allocation(
    desk_inputs: list[DeskCapitalInput],
    portfolio_capital: float,
) -> list[float]:
    """Simple proportional allocation by standalone capital."""
    total = sum(d.standalone_capital for d in desk_inputs)
    if total <= 0:
        return [0.0] * len(desk_inputs)
    return [d.standalone_capital / total * portfolio_capital for d in desk_inputs]


def calculate_rorc(pnl: float, allocated_capital: float) -> float:
    """Return on Regulatory Capital = PnL / Capital."""
    if allocated_capital <= 0:
        return 0.0
    return pnl / allocated_capital


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

def allocate_and_report(
    desk_inputs: list[DeskCapitalInput],
    portfolio_capital: float | None = None,
    method: str = "euler",
    hurdle_rate: float = 0.10,
    correlation_matrix: np.ndarray | None = None,
) -> CapitalAllocationResult:
    """Allocate capital, compute RORC, check hurdle rates.

    Args:
        desk_inputs: per-desk standalone capital and P&L.
        portfolio_capital: total capital budget. If None, derived.
        method: "euler" or "pro_rata".
        hurdle_rate: minimum required RORC.
        correlation_matrix: optional desk correlation matrix for Euler.
    """
    if not desk_inputs:
        raise ValueError("desk_inputs must not be empty")

    if method == "euler":
        allocated = euler_allocation(desk_inputs, portfolio_capital, correlation_matrix)
    else:
        if portfolio_capital is None:
            portfolio_capital = sum(d.standalone_capital for d in desk_inputs) * 0.85
        allocated = pro_rata_allocation(desk_inputs, portfolio_capital)

    total_portfolio = sum(allocated)
    total_standalone = sum(d.standalone_capital for d in desk_inputs)
    total_pnl = sum(d.pnl_annual for d in desk_inputs)

    desk_allocs = []
    for i, d in enumerate(desk_inputs):
        alloc = allocated[i]
        alloc_pct = alloc / total_portfolio if total_portfolio > 0 else 0.0
        rorc = calculate_rorc(d.pnl_annual, alloc)
        exceeds = rorc >= hurdle_rate
        excess = d.pnl_annual - hurdle_rate * alloc
        ep = excess  # economic profit

        desk_allocs.append(DeskAllocation(
            desk_id=d.desk_id,
            standalone_capital=d.standalone_capital,
            allocated_capital=alloc,
            allocation_pct=alloc_pct,
            pnl=d.pnl_annual,
            rorc=rorc,
            exceeds_hurdle=exceeds,
            excess_return=excess,
            economic_profit=ep,
        ))

    div_benefit = total_standalone - total_portfolio
    div_ratio = total_portfolio / total_standalone if total_standalone > 0 else 1.0
    portfolio_rorc = total_pnl / total_portfolio if total_portfolio > 0 else 0.0

    above = sum(1 for da in desk_allocs if da.exceeds_hurdle)
    below = len(desk_allocs) - above
    worst = min(desk_allocs, key=lambda da: da.rorc)
    best = max(desk_allocs, key=lambda da: da.rorc)

    return CapitalAllocationResult(
        total_portfolio_capital=total_portfolio,
        sum_standalone=total_standalone,
        diversification_benefit=div_benefit,
        diversification_ratio=div_ratio,
        method=method,
        hurdle_rate=hurdle_rate,
        desk_allocations=desk_allocs,
        portfolio_rorc=portfolio_rorc,
        n_desks_above_hurdle=above,
        n_desks_below_hurdle=below,
        worst_rorc_desk=worst.desk_id,
        best_rorc_desk=best.desk_id,
    )


# ═══════════════════════════════════════════════════════════════
# Capital Limit Monitor
# ═══════════════════════════════════════════════════════════════

def capital_limit_monitor(
    desk_allocations: list[DeskAllocation],
    limits: dict[str, float],
) -> list[dict]:
    """Check each desk's allocated capital against limits.

    Args:
        desk_allocations: from allocate_and_report().
        limits: {desk_id: max_capital}.

    Returns:
        List of breach dicts: {desk_id, allocated, limit, breach_pct, action}.
    """
    breaches = []
    for da in desk_allocations:
        limit = limits.get(da.desk_id)
        if limit is not None and da.allocated_capital > limit:
            breaches.append({
                "desk_id": da.desk_id,
                "allocated": da.allocated_capital,
                "limit": limit,
                "breach_pct": da.allocated_capital / limit,
                "action": f"Reduce {da.desk_id} capital by {da.allocated_capital - limit:,.0f}",
            })
    return breaches
