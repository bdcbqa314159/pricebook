"""Loan settlement mechanics and trade economics.

LSTA conventions for secondary loan trading: delayed compensation,
break funding, failed settlement penalties, trade P&L.

    from pricebook.loan_settlement import (
        LoanSettlement, delayed_compensation, break_funding,
        LoanTradeEcon, loan_bond_basis,
    )

References:
    LSTA (2022). The Handbook of Loan Syndications and Trading, Ch. 18-19.
    LSTA (2021). Guide to Settlement and Clearing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any


# Standard settlement days
SETTLEMENT_DAYS = {
    "assignment": 7,
    "participation": 10,
    "distressed": 20,
}


@dataclass
class LoanSettlement:
    """Loan secondary trade settlement.

    Args:
        trade_date: when the trade was agreed.
        settle_date: expected settlement date.
        price: clean price (% of par, e.g. 98.5).
        par_amount: face amount traded.
        accrued: accrued interest at trade date.
        trade_type: "assignment" or "participation".
    """
    trade_date: date
    settle_date: date
    price: float
    par_amount: float
    accrued: float = 0.0
    trade_type: str = "assignment"

    @property
    def settlement_days(self) -> int:
        return (self.settle_date - self.trade_date).days

    @property
    def total_consideration(self) -> float:
        """Total cash: price × par + accrued."""
        return self.par_amount * self.price / 100.0 + self.accrued

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date.isoformat(),
            "settle_date": self.settle_date.isoformat(),
            "price": self.price, "par_amount": self.par_amount,
            "accrued": self.accrued, "trade_type": self.trade_type,
        }


def delayed_compensation(
    rate: float,
    days: int,
    notional: float,
) -> float:
    """Delayed compensation: buyer pays carry to seller for settlement delay.

    comp = rate × days/360 × notional

    LSTA convention: simple interest, ACT/360.
    """
    return rate * days / 360.0 * notional


def break_funding(
    old_rate: float,
    new_rate: float,
    remaining_days: int,
    notional: float,
) -> float:
    """Break funding cost when loan rate resets before settlement.

    If settlement spans a rate reset, buyer compensates seller for
    the rate difference over the remaining interest period.

    cost = (old_rate - new_rate) × remaining_days/360 × notional
    """
    return (old_rate - new_rate) * remaining_days / 360.0 * notional


def failed_settlement_penalty(
    days_overdue: int,
    notional: float,
    base_rate: float = 0.05,
    penalty_spread: float = 0.02,
) -> float:
    """Penalty for failed settlement.

    LSTA standard: base_rate + 200bp per day overdue.
    penalty = (base_rate + penalty_spread) × days/360 × notional
    """
    return (base_rate + penalty_spread) * days_overdue / 360.0 * notional


# ---------------------------------------------------------------------------
# Trade economics
# ---------------------------------------------------------------------------

@dataclass
class LoanTradeEcon:
    """P&L decomposition for a loan trade.

    Args:
        buy_price: purchase price (% of par).
        sell_price: sale price or current mark (% of par).
        par_amount: face amount.
        hold_days: days held.
        coupon_income: total coupon received during hold period.
        funding_cost: total funding cost during hold period.
    """
    buy_price: float
    sell_price: float
    par_amount: float
    hold_days: int
    coupon_income: float = 0.0
    funding_cost: float = 0.0

    @property
    def price_pnl(self) -> float:
        """P&L from price change."""
        return (self.sell_price - self.buy_price) / 100.0 * self.par_amount

    @property
    def carry(self) -> float:
        """Net carry: coupon income - funding cost."""
        return self.coupon_income - self.funding_cost

    @property
    def total_return(self) -> float:
        """Total return: price P&L + carry."""
        return self.price_pnl + self.carry

    @property
    def annualised_return(self) -> float:
        """Annualised total return."""
        if self.hold_days <= 0 or self.par_amount <= 0:
            return 0.0
        invested = self.buy_price / 100.0 * self.par_amount
        if invested <= 0:
            return 0.0
        return self.total_return / invested * 365.0 / self.hold_days

    def breakeven_hold(self, daily_carry: float | None = None) -> int:
        """Days to hold to break even on bid-ask (assuming constant carry)."""
        if daily_carry is None:
            daily_carry = self.carry / max(self.hold_days, 1)
        if daily_carry <= 0:
            return 999999
        loss = abs(self.price_pnl) if self.price_pnl < 0 else 0
        return max(1, int(math.ceil(loss / daily_carry)))

    def to_dict(self) -> dict:
        return {
            "price_pnl": self.price_pnl, "carry": self.carry,
            "total_return": self.total_return,
            "annualised_return": self.annualised_return,
            "hold_days": self.hold_days,
        }


# ---------------------------------------------------------------------------
# Loan-bond basis
# ---------------------------------------------------------------------------

def loan_bond_basis(loan_dm: float, bond_asw: float) -> float:
    """Loan-bond basis in basis points.

    basis = loan_dm - bond_asw (in bp)

    Typically positive: loans trade wider than bonds from same issuer
    due to illiquidity premium, settlement friction, structural differences.

    Historical range: 50-150bp for IG, wider for HY.
    """
    return (loan_dm - bond_asw) * 10_000


def basis_z_score(
    current_basis: float,
    historical_mean: float,
    historical_std: float,
) -> float:
    """Z-score of current basis vs historical distribution.

    Z > 1: basis is wide (loan is cheap vs bond)
    Z < -1: basis is tight (loan is rich vs bond)
    """
    if historical_std <= 0:
        return 0.0
    return (current_basis - historical_mean) / historical_std
