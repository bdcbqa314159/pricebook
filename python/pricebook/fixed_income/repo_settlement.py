"""Settlement fail cascades and buy-in.

    from pricebook.fixed_income.repo_settlement import (
        propagate_fails, buy_in_process, fail_cost_analysis,
    )

References:
    CSDR (2022). Central Securities Depositories Regulation, Art 7.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FailCascadeResult:
    """Result of fail cascade analysis."""
    initial_fail_id: str
    affected_trades: list[str]
    total_cascade_notional: float
    penalty_cost: float
    n_affected: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def propagate_fails(
    failed_trade_id: str,
    matched_book: list[dict],
    penalty_rate_bp: float = 50.0,
) -> FailCascadeResult:
    """Propagate a settlement fail through a matched book.

    If a repo fails to settle, the reverse repo funded by it also fails.

    Args:
        failed_trade_id: the initially failed trade.
        matched_book: list of {trade_id, linked_trade_id, notional, bond_id}.
        penalty_rate_bp: penalty for each failed day.
    """
    affected = [failed_trade_id]
    seen = {failed_trade_id}
    queue = [failed_trade_id]

    while queue:
        current = queue.pop(0)
        for trade in matched_book:
            if trade.get("linked_trade_id") == current and trade["trade_id"] not in seen:
                affected.append(trade["trade_id"])
                seen.add(trade["trade_id"])
                queue.append(trade["trade_id"])

    total_notional = sum(
        t["notional"] for t in matched_book if t["trade_id"] in seen
    )
    penalty = total_notional * penalty_rate_bp / 10_000 / 360  # 1 day penalty

    return FailCascadeResult(
        initial_fail_id=failed_trade_id,
        affected_trades=affected,
        total_cascade_notional=total_notional,
        penalty_cost=penalty,
        n_affected=len(affected),
    )


@dataclass
class BuyInResult:
    """Result of mandatory buy-in process."""
    bond_id: str
    buy_in_price: float
    original_price: float
    cost_difference: float
    timeline_days: int           # CSDR: 4-7 business days
    penalty_cost: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def buy_in_process(
    bond_id: str,
    failed_notional: float,
    original_price: float,
    current_market_price: float,
    days_failed: int,
    penalty_rate_bp: float = 50.0,
    csdr_extension_days: int = 4,
) -> BuyInResult:
    """CSDR mandatory buy-in process.

    After settlement fail persists beyond extension period,
    the receiving party initiates a buy-in at current market price.

    Args:
        bond_id: failed bond.
        failed_notional: face amount that failed.
        original_price: agreed price in the repo.
        current_market_price: current market price for buy-in.
        days_failed: business days since original settlement date.
        penalty_rate_bp: daily penalty rate.
        csdr_extension_days: CSDR extension period (4 for liquid, 7 for illiquid).
    """
    cost_diff = (current_market_price - original_price) / 100 * failed_notional
    penalty = failed_notional * penalty_rate_bp / 10_000 * days_failed / 360

    return BuyInResult(
        bond_id=bond_id,
        buy_in_price=current_market_price,
        original_price=original_price,
        cost_difference=cost_diff,
        timeline_days=max(csdr_extension_days - days_failed, 0),
        penalty_cost=penalty,
    )


def fail_cost_analysis(
    failed_notional: float,
    days_failed: int,
    penalty_rate_bp: float = 50.0,
    opportunity_cost_bp: float = 5.0,
    reputation_cost: float = 0.0,
) -> dict:
    """Total cost of a settlement fail.

    Components: penalty + opportunity cost + reputation impact.
    """
    penalty = failed_notional * penalty_rate_bp / 10_000 * days_failed / 360
    opportunity = failed_notional * opportunity_cost_bp / 10_000 * days_failed / 360

    return {
        "penalty_cost": penalty,
        "opportunity_cost": opportunity,
        "reputation_cost": reputation_cost,
        "total_cost": penalty + opportunity + reputation_cost,
        "days_failed": days_failed,
        "notional": failed_notional,
    }
