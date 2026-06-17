"""Crypto basis arbitrage: spot-perp, cash-and-carry, cross-exchange.

* :func:`spot_perp_basis` — basis trade analytics.
* :func:`cash_and_carry_crypto` — futures basis arb.
* :func:`triangular_arb` — triangular arbitrage (BTC/ETH/USDT).
* :func:`cross_exchange_arb` — same asset, different exchanges.

References:
    Cartea, Drissi & Monga, *Decentralised Finance*, 2023.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BasisTradeResult:
    """Basis trade analysis."""
    annualised_yield: float
    daily_carry: float
    basis_bps: float
    funding_component: float
    spot_price: float
    perp_price: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def spot_perp_basis(
    spot: float,
    perp: float,
    funding_rate: float,
    funding_interval_hours: float = 8.0,
) -> BasisTradeResult:
    """Spot-perp basis trade: long spot, short perp.

    Earns funding rate when perp > spot (contango).
    basis = (perp − spot) / spot.
    annualised = funding × intervals_per_year.

    Args:
        spot: spot price.
        perp: perpetual price.
        funding_rate: current per-interval funding rate.
    """
    basis = (perp - spot) / spot if spot > 0 else 0
    intervals_per_year = 365 * 24 / funding_interval_hours
    ann = funding_rate * intervals_per_year * 100
    daily = funding_rate * (24 / funding_interval_hours)

    return BasisTradeResult(
        annualised_yield=ann,
        daily_carry=daily,
        basis_bps=basis * 10_000,
        funding_component=funding_rate,
        spot_price=spot,
        perp_price=perp,
    )


@dataclass
class TriangularArbResult:
    """Triangular arbitrage result."""
    profit_pct: float
    path: list[str]
    rates: list[float]
    profitable: bool

    def to_dict(self) -> dict:
        return dict(vars(self))


def triangular_arb(
    btc_usd: float,
    eth_usd: float,
    eth_btc: float,
    fee: float = 0.001,
) -> TriangularArbResult:
    """Triangular arbitrage: BTC/USD → ETH/BTC → ETH/USD.

    If BTC/USD × ETH/BTC ≠ ETH/USD, there's an arb.

    Path 1: USD → BTC → ETH → USD
    Path 2: USD → ETH → BTC → USD

    Args:
        btc_usd: BTC price in USD.
        eth_usd: ETH price in USD.
        eth_btc: ETH price in BTC.
        fee: per-trade fee.
    """
    # Path 1: USD → BTC → ETH → USD
    btc_qty = 1.0 / btc_usd * (1 - fee)
    eth_qty = btc_qty / eth_btc * (1 - fee)
    usd_out_1 = eth_qty * eth_usd * (1 - fee)
    profit_1 = (usd_out_1 - 1.0) * 100

    # Path 2: USD → ETH → BTC → USD
    eth_qty_2 = 1.0 / eth_usd * (1 - fee)
    btc_qty_2 = eth_qty_2 * eth_btc * (1 - fee)
    usd_out_2 = btc_qty_2 * btc_usd * (1 - fee)
    profit_2 = (usd_out_2 - 1.0) * 100

    if profit_1 > profit_2:
        return TriangularArbResult(profit_1, ["USD→BTC→ETH→USD"], [btc_usd, eth_btc, eth_usd], profit_1 > 0)
    return TriangularArbResult(profit_2, ["USD→ETH→BTC→USD"], [eth_usd, eth_btc, btc_usd], profit_2 > 0)


def cross_exchange_arb(
    price_a: float,
    price_b: float,
    fee_a: float = 0.001,
    fee_b: float = 0.001,
    transfer_cost: float = 0.0,
) -> dict:
    """Cross-exchange arb: buy on cheap exchange, sell on expensive.

    Args:
        price_a: price on exchange A.
        price_b: price on exchange B.
        fee_a: trading fee on A.
        fee_b: trading fee on B.
    """
    if price_a < price_b:
        buy_exchange, sell_exchange = "A", "B"
        buy_price, sell_price = price_a, price_b
        buy_fee, sell_fee = fee_a, fee_b
    else:
        buy_exchange, sell_exchange = "B", "A"
        buy_price, sell_price = price_b, price_a
        buy_fee, sell_fee = fee_b, fee_a

    gross_spread = (sell_price - buy_price) / buy_price
    costs = buy_fee + sell_fee + transfer_cost / buy_price
    net_profit_pct = (gross_spread - costs) * 100

    return {
        "buy_exchange": buy_exchange,
        "sell_exchange": sell_exchange,
        "buy_price": buy_price,
        "sell_price": sell_price,
        "gross_spread_bps": gross_spread * 10_000,
        "costs_bps": costs * 10_000,
        "net_profit_pct": net_profit_pct,
        "profitable": net_profit_pct > 0,
    }
