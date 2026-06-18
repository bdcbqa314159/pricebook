"""Crypto trading desk: book management, P&L, risk.

* :class:`CryptoPosition` — single crypto position.
* :class:`CryptoBook` — portfolio of crypto positions.
* :func:`crypto_pnl` — P&L decomposition (spot, funding, fees).
* :func:`crypto_risk_report` — aggregated risk metrics.

References:
    Internal pricebook desk protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum


class CryptoInstrument(Enum):
    SPOT = "spot"
    PERPETUAL = "perpetual"
    FUTURE = "future"
    OPTION = "option"
    LP_POSITION = "lp_position"
    STAKING = "staking"


@dataclass
class CryptoPosition:
    """Single crypto position."""
    symbol: str                     # BTC, ETH, etc.
    instrument: CryptoInstrument
    quantity: float                 # signed (positive = long)
    entry_price: float
    current_price: float
    exchange: str = ""
    leverage: float = 1.0
    funding_accumulated: float = 0.0  # cumulative funding paid/received

    @property
    def notional_usd(self) -> float:
        return abs(self.quantity * self.current_price)

    @property
    def unrealised_pnl(self) -> float:
        return self.quantity * (self.current_price - self.entry_price)

    @property
    def total_pnl(self) -> float:
        return self.unrealised_pnl - self.funding_accumulated

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "instrument": self.instrument.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "notional_usd": self.notional_usd,
            "unrealised_pnl": self.unrealised_pnl,
            "total_pnl": self.total_pnl,
            "exchange": self.exchange,
            "leverage": self.leverage,
        }


class CryptoBook:
    """Crypto trading book."""

    def __init__(self, name: str = "crypto"):
        self.name = name
        self._positions: list[CryptoPosition] = []

    def add(self, pos: CryptoPosition):
        self._positions.append(pos)

    @property
    def positions(self) -> list[CryptoPosition]:
        return list(self._positions)

    def total_notional(self) -> float:
        return sum(p.notional_usd for p in self._positions)

    def total_pnl(self) -> float:
        return sum(p.total_pnl for p in self._positions)

    def by_symbol(self) -> dict[str, list[CryptoPosition]]:
        result: dict[str, list[CryptoPosition]] = {}
        for p in self._positions:
            result.setdefault(p.symbol, []).append(p)
        return result

    def by_exchange(self) -> dict[str, list[CryptoPosition]]:
        result: dict[str, list[CryptoPosition]] = {}
        for p in self._positions:
            result.setdefault(p.exchange, []).append(p)
        return result

    def by_instrument(self) -> dict[str, list[CryptoPosition]]:
        result: dict[str, list[CryptoPosition]] = {}
        for p in self._positions:
            result.setdefault(p.instrument.value, []).append(p)
        return result

    def net_exposure(self) -> dict[str, float]:
        """Net USD exposure per symbol."""
        result: dict[str, float] = {}
        for p in self._positions:
            result[p.symbol] = result.get(p.symbol, 0) + p.quantity * p.current_price
        return result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_positions": len(self._positions),
            "total_notional": self.total_notional(),
            "total_pnl": self.total_pnl(),
            "net_exposure": self.net_exposure(),
        }


@dataclass
class CryptoPnLResult:
    """Crypto P&L decomposition."""
    spot_pnl: float             # from price move
    funding_pnl: float          # from funding payments
    fee_pnl: float              # trading fees
    total_pnl: float
    pnl_btc: float              # total P&L in BTC terms
    pnl_usd: float              # total P&L in USD terms

    def to_dict(self) -> dict:
        return dict(vars(self))


def crypto_pnl(
    book: CryptoBook,
    btc_price: float = 50_000,
) -> CryptoPnLResult:
    """P&L decomposition for a crypto book.

    Splits into: spot move + funding + fees.
    Reports in both BTC and USD.
    """
    spot = sum(p.unrealised_pnl for p in book.positions)
    funding = sum(p.funding_accumulated for p in book.positions)
    total = spot - funding

    return CryptoPnLResult(
        spot_pnl=spot,
        funding_pnl=-funding,
        fee_pnl=0,
        total_pnl=total,
        pnl_btc=total / btc_price if btc_price > 0 else 0,
        pnl_usd=total,
    )


def crypto_risk_report(book: CryptoBook) -> dict:
    """Aggregated risk metrics for crypto book."""
    positions = book.positions
    if not positions:
        return {"n_positions": 0}

    net = book.net_exposure()
    gross = book.total_notional()
    max_leverage = max(p.leverage for p in positions)

    return {
        "n_positions": len(positions),
        "gross_notional": gross,
        "net_exposure": net,
        "total_pnl": book.total_pnl(),
        "max_leverage": max_leverage,
        "n_exchanges": len(book.by_exchange()),
        "n_symbols": len(book.by_symbol()),
        "instruments": list(book.by_instrument().keys()),
    }


# ═══════════════════════════════════════════════════════════════
# CD14: Greeks Aggregation, Margin Monitor, Scenarios, Fees, Hedge
# ═══════════════════════════════════════════════════════════════

@dataclass
class CryptoGreeksAgg:
    """Aggregated Greeks for crypto options book."""
    net_delta_usd: float
    net_delta_btc: float
    net_gamma_usd: float
    net_vega: float
    net_theta: float
    gamma_pnl_1pct: float       # P&L from 1% spot move
    n_options: int

    def to_dict(self) -> dict:
        return dict(vars(self))


def aggregate_options_greeks(
    book: CryptoBook,
    spot: float,
) -> CryptoGreeksAgg:
    """Aggregate Greeks across all option positions in a crypto book.

    Requires positions to have greeks dict: {"delta", "gamma", "vega", "theta"}.
    """
    d_usd = 0.0
    g_usd = 0.0
    v = 0.0
    th = 0.0
    n_opt = 0

    for p in book.positions:
        if p.instrument != CryptoInstrument.OPTION:
            continue
        greeks = getattr(p, 'greeks', None)
        if greeks is None:
            continue
        qty = p.quantity
        d_usd += greeks.get("delta", 0) * qty * spot
        g_usd += greeks.get("gamma", 0) * qty * spot**2
        v += greeks.get("vega", 0) * qty
        th += greeks.get("theta", 0) * qty
        n_opt += 1

    gamma_1pct = 0.5 * g_usd * 0.01**2

    return CryptoGreeksAgg(d_usd, d_usd / spot if spot > 0 else 0,
                            g_usd, v, th, gamma_1pct, n_opt)


@dataclass
class MarginMonitorResult:
    """Cross-exchange margin monitoring."""
    total_equity: float
    total_used_margin: float
    total_available: float
    worst_exchange: str
    worst_margin_ratio: float
    alerts: list[str]

    def to_dict(self) -> dict:
        return dict(vars(self))


def margin_monitor(
    book: CryptoBook,
    exchange_equity: dict[str, float],
    warning_ratio: float = 0.70,
    danger_ratio: float = 0.90,
) -> MarginMonitorResult:
    """Monitor margin utilisation across exchanges.

    Args:
        book: crypto book with positions.
        exchange_equity: {exchange: equity} available margin per exchange.
        warning_ratio: margin ratio triggering warning.
        danger_ratio: margin ratio triggering danger alert.
    """
    by_exchange = book.by_exchange()
    alerts = []
    worst = ("", 0.0)
    total_equity = sum(exchange_equity.values())
    total_used = 0.0

    for exchange, positions in by_exchange.items():
        equity = exchange_equity.get(exchange, 0)
        used = sum(p.notional_usd / p.leverage for p in positions)
        total_used += used

        ratio = used / equity if equity > 0 else float('inf')
        if ratio > worst[1]:
            worst = (exchange, ratio)

        if ratio > danger_ratio:
            alerts.append(f"DANGER: {exchange} margin ratio {ratio:.1%}")
        elif ratio > warning_ratio:
            alerts.append(f"WARNING: {exchange} margin ratio {ratio:.1%}")

    return MarginMonitorResult(
        total_equity=total_equity,
        total_used_margin=total_used,
        total_available=total_equity - total_used,
        worst_exchange=worst[0],
        worst_margin_ratio=worst[1],
        alerts=alerts,
    )


@dataclass
class ScenarioResult:
    """Crypto scenario analysis result."""
    scenario_name: str
    spot_shock_pct: float
    vol_shock_pct: float
    pnl_impact: float
    new_margin_ratio: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def scenario_analysis(
    book: CryptoBook,
    scenarios: list[dict],
    current_equity: float,
) -> list[ScenarioResult]:
    """Run stress scenarios on crypto book.

    Each scenario: {"name", "spot_shock" (decimal), "vol_shock" (decimal)}.

    Args:
        book: crypto book.
        scenarios: list of scenario dicts.
        current_equity: current total equity.
    """
    results = []
    for s in scenarios:
        name = s.get("name", "unnamed")
        spot_shock = s.get("spot_shock", 0)
        vol_shock = s.get("vol_shock", 0)

        pnl = 0.0
        for p in book.positions:
            # Spot P&L
            price_change = p.current_price * spot_shock
            pnl += p.quantity * price_change

        new_equity = current_equity + pnl
        used = sum(p.notional_usd / p.leverage for p in book.positions)
        new_ratio = used / new_equity if new_equity > 0 else float('inf')

        results.append(ScenarioResult(name, spot_shock * 100, vol_shock * 100,
                                       pnl, new_ratio))

    return results


def fee_attribution(book: CryptoBook) -> dict:
    """Fee P&L attribution per exchange and instrument type."""
    by_exchange: dict[str, float] = {}
    by_instrument: dict[str, float] = {}

    for p in book.positions:
        # Estimate fees from notional × taker fee
        est_fee = p.notional_usd * 0.0005  # 5bps estimate
        by_exchange[p.exchange] = by_exchange.get(p.exchange, 0) + est_fee
        by_instrument[p.instrument.value] = by_instrument.get(p.instrument.value, 0) + est_fee

    return {
        "total_estimated_fees": sum(by_exchange.values()),
        "by_exchange": by_exchange,
        "by_instrument": by_instrument,
    }


def hedge_recommendations(
    book: CryptoBook,
    spot: float,
    delta_limit_usd: float = 50_000.0,
) -> list[dict]:
    """Hedge recommendations based on current exposure.

    If net delta exceeds limit, recommend hedging with perp.
    """
    net = book.net_exposure()
    recs = []

    for symbol, exposure in net.items():
        if abs(exposure) > delta_limit_usd:
            direction = "sell" if exposure > 0 else "buy"
            size_usd = abs(exposure) - delta_limit_usd
            size_qty = size_usd / spot if spot > 0 else 0
            recs.append({
                "symbol": symbol,
                "action": f"{direction} {symbol}-PERP",
                "size_usd": size_usd,
                "size_qty": size_qty,
                "reason": f"Net {symbol} exposure ${exposure:,.0f} exceeds limit ${delta_limit_usd:,.0f}",
            })

    return recs
