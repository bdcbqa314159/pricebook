"""Repo XVA: funding cost, capital cost, margin cost, all-in pricing.

Wires the generic XVA framework (xva.py) to repo-specific economics.
For repos, exposure is constant (cash amount, not path-dependent),
so XVA computations simplify to deterministic formulas.

    from pricebook.repo_xva import (
        repo_fva, repo_kva, repo_mva,
        repo_total_cost, RepoAllInCost,
    )

References:
    Lou, W. (2016). Gap Risk in Secured Financing. SSRN.
    Green, A. (2015). XVA: Credit, Funding, and Capital Valuation Adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.repo_desk import RepoTrade


def repo_fva(
    trade: RepoTrade,
    funding_spread: float,
) -> float:
    """FVA for a repo: cost of funding at spread over OIS.

    For repos, exposure = cash_amount (constant over term).
    FVA = cash × funding_spread × term / 360.

    Positive FVA = cost to the desk (funding > OIS).

    Args:
        funding_spread: dealer's funding rate minus OIS (e.g. 0.002 = 20bp).
    """
    dt = trade.term_days / 360.0
    return trade.cash_amount * funding_spread * dt


def repo_kva(
    trade: RepoTrade,
    capital_charge: float,
    hurdle_rate: float = 0.10,
) -> float:
    """KVA for a repo: cost of holding regulatory capital.

    KVA = capital × hurdle_rate × term.

    The capital charge should come from sft_ead() × risk_weight × 8%.

    Args:
        capital_charge: regulatory capital required ($ amount).
        hurdle_rate: return required on capital (e.g. 0.10 = 10%).
    """
    dt = trade.term_days / 365.0
    return capital_charge * hurdle_rate * dt


def repo_mva(
    trade: RepoTrade,
    initial_margin: float | None = None,
    funding_spread: float = 0.002,
) -> float:
    """MVA for a repo: cost of funding initial margin.

    MVA = IM × funding_spread × term.

    If IM not provided, uses haircut × market_value as proxy.

    Args:
        initial_margin: IM posted ($ amount). None = use haircut.
        funding_spread: cost of funding the margin.
    """
    if initial_margin is None:
        initial_margin = trade.market_value * (trade.haircut + trade.fx_haircut)
    dt = trade.term_days / 360.0
    return initial_margin * funding_spread * dt


def repo_gap_cost(
    trade: RepoTrade,
    funding_rate: float,
    collateral_coverage: float = 1.0,
) -> float:
    """Gap risk: cost when collateral coverage < 100%.

    The unfunded portion must be financed at the unsecured rate.
    gap_cost = (1 - coverage) × cash × (funding - repo) × term.

    Reuses: xva.repo_gap_risk() formula.
    """
    from pricebook.xva import repo_gap_risk
    return repo_gap_risk(
        trade.cash_amount, trade.repo_rate,
        funding_rate, collateral_coverage,
        trade.term_days / 360.0,
    )


# ---------------------------------------------------------------------------
# All-in cost
# ---------------------------------------------------------------------------

@dataclass
class RepoAllInCost:
    """Complete cost decomposition beyond the headline repo rate."""
    interest: float
    agent_fees: float
    fva: float
    kva: float
    mva: float
    gap_cost: float
    total_cost: float
    all_in_rate: float   # annualised, ACT/360
    headline_rate: float

    def to_dict(self) -> dict:
        return {
            "interest": self.interest, "agent_fees": self.agent_fees,
            "fva": self.fva, "kva": self.kva, "mva": self.mva,
            "gap_cost": self.gap_cost, "total_cost": self.total_cost,
            "all_in_rate_pct": self.all_in_rate * 100,
            "headline_rate_pct": self.headline_rate * 100,
            "hidden_cost": self.total_cost - self.interest,
            "hidden_cost_bps": (self.all_in_rate - self.headline_rate) * 10_000,
        }


def repo_total_cost(
    trade: RepoTrade,
    funding_spread: float = 0.002,
    capital_charge: float = 0.0,
    hurdle_rate: float = 0.10,
    initial_margin: float | None = None,
    agent_fees: float = 0.0,
    collateral_coverage: float = 1.0,
    funding_rate: float = 0.05,
) -> RepoAllInCost:
    """Total cost of a repo: interest + FVA + KVA + MVA + gap + agent.

    This is the 'true cost' that systems hide behind the headline rate.
    The difference (all_in - headline) is what the desk actually pays
    beyond what's on the ticket.

    Args:
        funding_spread: desk funding rate minus OIS.
        capital_charge: regulatory capital ($ from sft_ead × rw × 8%).
        hurdle_rate: return on capital.
        initial_margin: IM posted. None = use haircut.
        agent_fees: tri-party agent fees.
        collateral_coverage: fraction covered (< 1 = gap risk).
        funding_rate: unsecured funding rate (for gap risk).
    """
    interest = trade.interest
    _fva = repo_fva(trade, funding_spread)
    _kva = repo_kva(trade, capital_charge, hurdle_rate)
    _mva = repo_mva(trade, initial_margin, funding_spread)
    _gap = repo_gap_cost(trade, funding_rate, collateral_coverage)

    total = interest + agent_fees + _fva + _kva + _mva + _gap

    # Annualise
    dt = trade.term_days / 360.0
    all_in = total / trade.cash_amount / dt if trade.cash_amount > 0 and dt > 0 else 0.0

    return RepoAllInCost(
        interest=interest, agent_fees=agent_fees,
        fva=_fva, kva=_kva, mva=_mva, gap_cost=_gap,
        total_cost=total, all_in_rate=all_in,
        headline_rate=trade.repo_rate,
    )
