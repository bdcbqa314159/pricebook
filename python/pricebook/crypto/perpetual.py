"""Perpetual swap/futures pricing.

Perpetuals never expire — the funding rate mechanism keeps the
perp price anchored to the spot index. Three contract types:
linear (USD-margined), inverse (crypto-margined), quanto.

* :class:`PerpetualSwap` — perpetual swap instrument.
* :func:`fair_basis` — theoretical perp premium over spot.
* :func:`funding_payment` — funding payment for a position.
* :func:`mark_price` — mark price from index + EMA basis.
* :func:`liquidation_price` — price at which position is liquidated.

References:
    BitMEX, *Perpetual Contracts Guide*.
    Deribit, *Perpetual Contract Specification*.
    Cartea, Drissi & Monga, *Decentralised Finance and Automated Market
    Making*, 2023.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class ContractType(Enum):
    """Perpetual contract type."""
    LINEAR = "linear"           # USD-margined (USDT settled)
    INVERSE = "inverse"         # crypto-margined (BTC/ETH settled)
    QUANTO = "quanto"           # cross-currency


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class PerpetualSwap:
    """Perpetual swap contract specification.

    Attributes:
        symbol: e.g. "BTC-PERP", "ETH-USDT-PERP".
        contract_type: linear, inverse, or quanto.
        multiplier: contract size (1 for linear, varies for inverse).
        tick_size: minimum price increment.
        max_leverage: maximum allowed leverage.
        funding_interval_hours: funding period (typically 8h).
        maintenance_margin: maintenance margin rate.
        maker_fee: maker fee rate.
        taker_fee: taker fee rate.
    """
    symbol: str
    contract_type: ContractType = ContractType.LINEAR
    multiplier: float = 1.0
    tick_size: float = 0.01
    max_leverage: float = 100.0
    funding_interval_hours: float = 8.0
    maintenance_margin: float = 0.005
    maker_fee: float = 0.0002
    taker_fee: float = 0.0005

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "type": self.contract_type.value,
            "multiplier": self.multiplier,
            "max_leverage": self.max_leverage,
        }


# ═══════════════════════════════════════════════════════════════
# Pricing
# ═══════════════════════════════════════════════════════════════

@dataclass
class PerpPricingResult:
    """Perpetual pricing result."""
    mark_price: float
    index_price: float
    basis: float                # mark - index
    basis_pct: float            # basis as % of index
    annualised_basis: float     # basis annualised
    fair_funding_rate: float    # implied 8h funding rate

    def to_dict(self) -> dict:
        return vars(self)


def fair_basis(
    index_price: float,
    funding_rate: float,
    hours_to_funding: float = 8.0,
) -> float:
    """Theoretical fair basis (premium/discount).

    basis = index × funding_rate × (hours_to_funding / 24)

    When funding > 0: longs pay shorts → perp should trade above index.
    When funding < 0: shorts pay longs → perp should trade below.

    Args:
        index_price: spot index price.
        funding_rate: current funding rate (per interval).
        hours_to_funding: hours until next funding settlement.
    """
    return index_price * funding_rate * (hours_to_funding / 24)


def mark_price(
    index_price: float,
    last_price: float,
    basis_ema: float = 0.0,
    ema_window: int = 30,
) -> float:
    """Mark price: index + EMA of basis.

    Used for liquidation and unrealised P&L calculations.
    Prevents manipulation via last-traded price.

    mark = index + EMA(last − index)

    Args:
        index_price: multi-exchange spot index.
        last_price: last traded perp price.
        basis_ema: current EMA of (perp − index) basis.
        ema_window: EMA window in periods.
    """
    current_basis = last_price - index_price
    alpha = 2.0 / (ema_window + 1)
    new_ema = alpha * current_basis + (1 - alpha) * basis_ema
    return index_price + new_ema


def funding_rate_from_basis(
    perp_price: float,
    index_price: float,
    interval_hours: float = 8.0,
    clamp_rate: float = 0.0005,
) -> float:
    """Implied funding rate from perp-index basis.

    funding = clamp(premium / interval_fraction, ±clamp_rate)
    premium = (perp − index) / index

    Most exchanges clamp the funding rate to ±0.05% per interval.

    Args:
        perp_price: perpetual last price.
        index_price: spot index.
        interval_hours: funding interval (8h default).
        clamp_rate: maximum absolute funding rate.
    """
    if index_price <= 0:
        return 0.0
    premium = (perp_price - index_price) / index_price
    rate = premium * (24 / interval_hours)
    return max(-clamp_rate, min(rate, clamp_rate))


def price_perpetual(
    index_price: float,
    perp_price: float,
    funding_rate: float,
    hours_to_funding: float = 4.0,
    interval_hours: float = 8.0,
) -> PerpPricingResult:
    """Full perpetual pricing analytics.

    Args:
        index_price: spot index.
        perp_price: current perp price.
        funding_rate: current funding rate (per interval).
        hours_to_funding: hours until next funding.
    """
    basis = perp_price - index_price
    basis_pct = basis / index_price * 100 if index_price > 0 else 0
    intervals_per_year = 365 * 24 / interval_hours
    annualised = funding_rate * intervals_per_year * 100
    mk = mark_price(index_price, perp_price)

    return PerpPricingResult(
        mark_price=mk,
        index_price=index_price,
        basis=basis,
        basis_pct=basis_pct,
        annualised_basis=annualised,
        fair_funding_rate=funding_rate,
    )


# ═══════════════════════════════════════════════════════════════
# Funding Payment
# ═══════════════════════════════════════════════════════════════

@dataclass
class FundingPaymentResult:
    """Funding payment calculation."""
    payment: float              # positive = you pay, negative = you receive
    position_value: float
    funding_rate: float
    side: str

    def to_dict(self) -> dict:
        return vars(self)


def funding_payment(
    position_size: float,
    mark_price: float,
    funding_rate: float,
    side: PositionSide = PositionSide.LONG,
    contract_type: ContractType = ContractType.LINEAR,
) -> FundingPaymentResult:
    """Calculate funding payment for a position.

    Linear: payment = position_size × mark_price × funding_rate
    Inverse: payment = position_size × funding_rate / mark_price

    Longs pay shorts when funding > 0 (perp above index).
    Shorts pay longs when funding < 0 (perp below index).

    Args:
        position_size: number of contracts (positive).
        mark_price: current mark price.
        funding_rate: funding rate for this interval.
        side: LONG or SHORT.
    """
    if contract_type == ContractType.LINEAR:
        notional = position_size * mark_price
        payment = notional * funding_rate
    else:  # inverse
        notional = position_size
        payment = position_size * funding_rate / mark_price if mark_price > 0 else 0

    # Longs pay when positive, shorts receive (and vice versa)
    if side == PositionSide.SHORT:
        payment = -payment

    return FundingPaymentResult(
        payment=payment,
        position_value=notional if contract_type == ContractType.LINEAR else position_size / mark_price,
        funding_rate=funding_rate,
        side=side.value,
    )


# ═══════════════════════════════════════════════════════════════
# Liquidation
# ═══════════════════════════════════════════════════════════════

@dataclass
class LiquidationResult:
    """Liquidation price calculation."""
    liquidation_price: float
    margin: float
    leverage: float
    distance_pct: float         # how far current price is from liquidation

    def to_dict(self) -> dict:
        return vars(self)


def liquidation_price(
    entry_price: float,
    leverage: float,
    side: PositionSide = PositionSide.LONG,
    maintenance_margin: float = 0.005,
    contract_type: ContractType = ContractType.LINEAR,
) -> LiquidationResult:
    """Calculate liquidation price.

    Linear long: liq = entry × (1 − 1/leverage + maintenance_margin)
    Linear short: liq = entry × (1 + 1/leverage − maintenance_margin)

    Inverse long: liq = entry × leverage / (leverage + 1 − maintenance_margin × leverage)
    Inverse short: liq = entry × leverage / (leverage − 1 + maintenance_margin × leverage)

    Args:
        entry_price: position entry price.
        leverage: position leverage.
        maintenance_margin: maintenance margin rate.
    """
    mm = maintenance_margin
    margin = entry_price / leverage

    if contract_type == ContractType.LINEAR:
        if side == PositionSide.LONG:
            liq = entry_price * (1 - 1 / leverage + mm)
        else:
            liq = entry_price * (1 + 1 / leverage - mm)
    else:  # inverse
        if side == PositionSide.LONG:
            denom = leverage + 1 - mm * leverage
            liq = entry_price * leverage / denom if denom > 0 else 0
        else:
            denom = leverage - 1 + mm * leverage
            liq = entry_price * leverage / denom if denom > 0 else float('inf')

    distance = abs(entry_price - liq) / entry_price * 100

    return LiquidationResult(
        liquidation_price=max(liq, 0),
        margin=margin,
        leverage=leverage,
        distance_pct=distance,
    )
