"""Securities lending integration.

    from pricebook.fixed_income.securities_lending import (
        SecLendingTrade, lending_vs_repo_arbitrage, locate_availability,
    )
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SecLendingTrade:
    """A securities lending agreement."""
    bond_id: str
    lender: str
    borrower: str
    quantity: float
    lending_fee_bp: float        # annualised fee in bp
    collateral_type: str         # "cash", "non_cash"
    collateral_haircut: float    # haircut on posted collateral
    term_days: int               # 0 = open/on-demand
    rebate_rate: float = 0.0     # for cash collateral: rebate to borrower
    currency: str = "USD"

    @property
    def net_fee_bp(self) -> float:
        """Net fee = lending_fee - rebate (for cash collateral)."""
        return self.lending_fee_bp - self.rebate_rate * 10_000

    def income(self, days: int | None = None) -> float:
        """Fee income over a period."""
        d = days or self.term_days or 1
        return self.quantity * self.lending_fee_bp / 10_000 * d / 360

    def to_dict(self) -> dict:
        return {
            "bond_id": self.bond_id,
            "lending_fee_bp": self.lending_fee_bp,
            "net_fee_bp": self.net_fee_bp,
            "quantity": self.quantity,
            "term_days": self.term_days,
            "collateral_type": self.collateral_type,
        }


def lending_vs_repo_arbitrage(
    lending_fee_bp: float,
    repo_special_spread_bp: float,
    transaction_cost_bp: float = 2.0,
) -> dict:
    """Compare securities lending fee vs repo special rate.

    If lending_fee > special_spread → lend, don't repo.
    If special_spread > lending_fee → repo (go special), don't lend.

    Args:
        lending_fee_bp: fee earned from lending.
        repo_special_spread_bp: GC-special spread earned from repo.
        transaction_cost_bp: round-trip transaction costs.
    """
    lending_net = lending_fee_bp - transaction_cost_bp
    repo_net = repo_special_spread_bp - transaction_cost_bp

    if lending_net > repo_net:
        recommendation = "LEND"
        advantage_bp = lending_net - repo_net
    elif repo_net > lending_net:
        recommendation = "REPO_SPECIAL"
        advantage_bp = repo_net - lending_net
    else:
        recommendation = "NEUTRAL"
        advantage_bp = 0.0

    return {
        "lending_fee_bp": lending_fee_bp,
        "repo_special_bp": repo_special_spread_bp,
        "lending_net_bp": lending_net,
        "repo_net_bp": repo_net,
        "recommendation": recommendation,
        "advantage_bp": advantage_bp,
    }


def locate_availability(
    inventory: dict[str, float],
    requested_bonds: list[str],
) -> list[dict]:
    """Check availability of bonds for lending/shorting.

    Args:
        inventory: {bond_id: available_quantity}.
        requested_bonds: list of bond IDs to check.

    Returns list of {bond_id, available, quantity, status}.
    """
    results = []
    for bond in requested_bonds:
        qty = inventory.get(bond, 0.0)
        results.append({
            "bond_id": bond,
            "available": qty > 0,
            "quantity": qty,
            "status": "available" if qty > 0 else "hard_to_borrow" if bond in inventory else "not_held",
        })
    return results
