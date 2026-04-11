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
from datetime import date, timedelta
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


# ---- Business-day-aware helpers ----

def add_business_days(
    start: date,
    n: int,
    calendar: object | None = None,
) -> date:
    """Advance *n* business days from *start*.

    If *calendar* is ``None``, only weekends are skipped. Otherwise
    the calendar's ``is_business_day()`` method is used.

    ``n`` can be negative (move backward).
    """
    step = 1 if n >= 0 else -1
    remaining = abs(n)
    current = start
    while remaining > 0:
        current += timedelta(days=step)
        if calendar is not None:
            if calendar.is_business_day(current):
                remaining -= 1
        else:
            if current.weekday() < 5:
                remaining -= 1
    return current


# FX settlement lag: T+1 pairs (same-day or T+1 settlement).
FX_T1_PAIRS: set[tuple[str, str]] = {
    ("USD", "CAD"), ("CAD", "USD"),
    ("USD", "TRY"), ("TRY", "USD"),
    ("USD", "RUB"), ("RUB", "USD"),
    ("USD", "PHP"), ("PHP", "USD"),
}


def fx_spot_date(
    trade_date: date,
    base: str,
    quote: str,
    calendar: object | None = None,
) -> date:
    """Compute the FX spot settlement date.

    T+2 for most pairs; T+1 for USD/CAD, USD/TRY, USD/RUB, USD/PHP.
    The result lands on a business day per the supplied *calendar*.
    """
    pair = (base.upper(), quote.upper())
    lag = 1 if pair in FX_T1_PAIRS else 2
    return add_business_days(trade_date, lag, calendar)


# Bond settlement lag by market.
BOND_SETTLEMENT_LAGS: dict[str, int] = {
    "US": 1,
    "USD": 1,
    "CA": 1,
    "CAD": 1,
    "EU": 2,
    "EUR": 2,
    "TARGET": 2,
    "UK": 1,
    "GBP": 1,
    "JP": 2,
    "JPY": 2,
    "CH": 2,
    "CHF": 2,
    "AU": 2,
    "AUD": 2,
}


def bond_settlement_date(
    trade_date: date,
    market: str,
    calendar: object | None = None,
) -> date:
    """Compute the bond settlement date for a given market.

    US/UK: T+1, Europe/Japan/Australia: T+2. Falls back to T+2
    for unrecognised markets.
    """
    lag = BOND_SETTLEMENT_LAGS.get(market.upper(), 2)
    return add_business_days(trade_date, lag, calendar)


def settlement_risk(
    trade_amount: float,
    trade_date: date,
    settlement_date: date,
    settlement_type: SettlementType = SettlementType.CASH,
    replacement_cost_pct: float = 0.10,
) -> SettlementRiskResult:
    """Compute settlement risk: exposure between trade and settlement.

    Args:
        replacement_cost_pct: fraction of notional at risk for cash/auction
            settlement (default 10%).

    PHYSICAL/AUCTION: gross exposure (full amount).
    CASH/ELECT: replacement cost approximation (default 10% of notional).
    """
    if settlement_date < trade_date:
        raise ValueError("settlement_date must be on or after trade_date")
    days = (settlement_date - trade_date).days

    if settlement_type in (SettlementType.PHYSICAL, SettlementType.AUCTION):
        exposure = abs(trade_amount)
    else:
        exposure = abs(trade_amount) * replacement_cost_pct

    return SettlementRiskResult(exposure, days, settlement_type)
