"""Settlement framework: physical vs cash settlement across products.

Defines settlement types and conventions, computes settlement amounts
for different products, and models the settlement risk exposure window.

    from pricebook.settlement import (
        SettlementType, cash_settlement, physical_settlement,
        cds_settlement, option_settlement, futures_settlement,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum


# ---- Settlement types ----

class SettlementType(Enum):
    CASH = "cash"
    PHYSICAL = "physical"
    AUCTION = "auction"       # CDS credit event (ISDA auction)
    ELECT = "elect"           # holder chooses at exercise


# ---- Settlement conventions ----

SETTLEMENT_CONVENTIONS: dict[str, dict[str, object]] = {
    "ir_swap": {"type": SettlementType.CASH, "lag_days": 0},
    "swaption_physical": {"type": SettlementType.PHYSICAL, "lag_days": 0},
    "swaption_cash": {"type": SettlementType.CASH, "lag_days": 0},
    "cds_physical": {"type": SettlementType.PHYSICAL, "lag_days": 30},
    "cds_auction": {"type": SettlementType.AUCTION, "lag_days": 5},
    "equity_option_physical": {"type": SettlementType.PHYSICAL, "lag_days": 2},
    "equity_option_cash": {"type": SettlementType.CASH, "lag_days": 1},
    "bond_future": {"type": SettlementType.PHYSICAL, "lag_days": 3},
    "equity_future": {"type": SettlementType.CASH, "lag_days": 0},
    "commodity_future_physical": {"type": SettlementType.PHYSICAL, "lag_days": 0},
    "commodity_future_cash": {"type": SettlementType.CASH, "lag_days": 0},
    "fx_spot": {"type": SettlementType.PHYSICAL, "lag_days": 2},
    "fx_ndf": {"type": SettlementType.CASH, "lag_days": 2},
}


def get_convention(product: str) -> dict[str, object]:
    """Look up settlement convention for a product type."""
    if product not in SETTLEMENT_CONVENTIONS:
        raise KeyError(f"Unknown product type: {product}")
    return SETTLEMENT_CONVENTIONS[product]


# ---- Cash settlement ----

@dataclass
class CashSettlementResult:
    """Result of cash settlement."""
    settlement_type: SettlementType
    amount: float
    settlement_date: date
    currency: str


def cash_settlement(
    pv: float,
    exercise_date: date,
    lag_days: int = 0,
    currency: str = "USD",
) -> CashSettlementResult:
    """Generic cash settlement: pay/receive PV on settlement date."""
    settle = date.fromordinal(exercise_date.toordinal() + lag_days)
    return CashSettlementResult(SettlementType.CASH, pv, settle, currency)


# ---- CDS settlement ----

@dataclass
class CDSSettlementResult:
    """CDS credit event settlement."""
    settlement_type: SettlementType
    protection_payout: float
    recovery_value: float
    bond_delivered: bool
    settlement_date: date


def cds_settlement_physical(
    notional: float,
    recovery: float,
    event_date: date,
    lag_days: int = 30,
) -> CDSSettlementResult:
    """Physical CDS settlement: deliver bond, receive par.

    Protection buyer delivers defaulted bond (worth recovery × notional)
    and receives full notional.
    """
    settle = date.fromordinal(event_date.toordinal() + lag_days)
    payout = notional  # buyer receives par
    recovery_val = recovery * notional  # bond delivered is worth this
    return CDSSettlementResult(
        SettlementType.PHYSICAL, payout, recovery_val, True, settle,
    )


def cds_settlement_cash(
    notional: float,
    recovery: float,
    event_date: date,
    lag_days: int = 5,
) -> CDSSettlementResult:
    """Cash (auction) CDS settlement: pay par minus recovery.

    Protection buyer receives (1 - recovery) × notional.
    """
    settle = date.fromordinal(event_date.toordinal() + lag_days)
    payout = (1 - recovery) * notional
    recovery_val = recovery * notional
    return CDSSettlementResult(
        SettlementType.AUCTION, payout, recovery_val, False, settle,
    )


# ---- Option settlement ----

@dataclass
class OptionSettlementResult:
    """Option exercise settlement."""
    settlement_type: SettlementType
    intrinsic: float
    cash_amount: float
    shares_delivered: float
    settlement_date: date


def option_settlement_cash(
    spot: float,
    strike: float,
    is_call: bool,
    contracts: float,
    exercise_date: date,
    lag_days: int = 1,
) -> OptionSettlementResult:
    """Cash-settled option exercise."""
    intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    cash = intrinsic * contracts
    settle = date.fromordinal(exercise_date.toordinal() + lag_days)
    return OptionSettlementResult(
        SettlementType.CASH, intrinsic, cash, 0.0, settle,
    )


def option_settlement_physical(
    spot: float,
    strike: float,
    is_call: bool,
    contracts: float,
    exercise_date: date,
    lag_days: int = 2,
) -> OptionSettlementResult:
    """Physical-settled option exercise.

    Call: buyer pays strike × contracts, receives shares.
    Put: buyer delivers shares, receives strike × contracts.
    """
    intrinsic = max(spot - strike, 0.0) if is_call else max(strike - spot, 0.0)
    settle = date.fromordinal(exercise_date.toordinal() + lag_days)

    if is_call:
        cash = -strike * contracts  # buyer pays strike
        shares = contracts  # buyer receives shares
    else:
        cash = strike * contracts  # buyer receives strike
        shares = -contracts  # buyer delivers shares

    return OptionSettlementResult(
        SettlementType.PHYSICAL, intrinsic, cash, shares, settle,
    )


# ---- Futures settlement ----

@dataclass
class FuturesSettlementResult:
    """Futures final settlement."""
    settlement_type: SettlementType
    final_settlement_price: float
    cash_amount: float
    physical_delivery: bool
    settlement_date: date


def futures_settlement_cash(
    entry_price: float,
    final_price: float,
    contracts: int,
    multiplier: float,
    expiry: date,
) -> FuturesSettlementResult:
    """Cash-settled futures final settlement."""
    cash = (final_price - entry_price) * contracts * multiplier
    return FuturesSettlementResult(
        SettlementType.CASH, final_price, cash, False, expiry,
    )


def futures_settlement_physical(
    entry_price: float,
    invoice_price: float,
    contracts: int,
    multiplier: float,
    expiry: date,
    delivery_lag: int = 3,
) -> FuturesSettlementResult:
    """Physical-delivery futures settlement (e.g. bond futures).

    Buyer pays invoice price, receives the physical commodity/bond.
    """
    cash = (invoice_price - entry_price) * contracts * multiplier
    settle = date.fromordinal(expiry.toordinal() + delivery_lag)
    return FuturesSettlementResult(
        SettlementType.PHYSICAL, invoice_price, cash, True, settle,
    )


# ---- Settlement risk ----

@dataclass
class SettlementRiskResult:
    """Settlement risk exposure."""
    exposure: float
    days_at_risk: int
    settlement_type: SettlementType


def settlement_risk(
    trade_amount: float,
    trade_date: date,
    settlement_date: date,
    settlement_type: SettlementType = SettlementType.CASH,
) -> SettlementRiskResult:
    """Compute settlement risk: exposure between trade and settlement.

    For cash: risk = full amount during settlement window.
    For DvP: risk reduced to replacement cost.
    For PvP: risk eliminated (simultaneous exchange).
    """
    days = max((settlement_date - trade_date).days, 0)

    if settlement_type == SettlementType.PHYSICAL:
        # Gross settlement risk: full amount
        exposure = abs(trade_amount)
    else:
        # Net settlement: only replacement cost
        exposure = abs(trade_amount) * 0.1  # simplified: 10% replacement cost

    return SettlementRiskResult(exposure, days, settlement_type)
