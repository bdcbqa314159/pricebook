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
        return vars(self)


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
