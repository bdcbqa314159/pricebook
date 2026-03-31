"""
Trade and Portfolio — first-class objects for position management.

A Trade wraps an instrument with direction, notional scaling, and metadata.
A Portfolio aggregates trades with a single PV call.

    trade = Trade(instrument=my_swap, direction=1, notional=10_000_000,
                  trade_date=date(2024,1,15), counterparty="ACME")
    pv = trade.pv(context)

    portfolio = Portfolio([trade1, trade2, trade3])
    total_pv = portfolio.pv(context)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook.pricing_context import PricingContext


@dataclass
class Trade:
    """A trade: instrument + direction + scaling + metadata.

    Args:
        instrument: any object with a pv_ctx(PricingContext) or
            pv(DiscountCurve, ...) method.
        direction: +1 for long, -1 for short.
        notional_scale: multiplier on the instrument's notional (default 1.0).
        trade_date: date the trade was executed.
        counterparty: counterparty name.
        trade_id: unique identifier.
    """

    instrument: object
    direction: int = 1
    notional_scale: float = 1.0
    trade_date: date | None = None
    counterparty: str = ""
    trade_id: str = ""

    def pv(self, ctx: PricingContext) -> float:
        """PV of the trade: direction * scale * instrument.pv_ctx(ctx)."""
        if hasattr(self.instrument, "pv_ctx"):
            raw_pv = self.instrument.pv_ctx(ctx)
        else:
            raise ValueError(
                f"Instrument {type(self.instrument).__name__} has no pv_ctx method"
            )
        return self.direction * self.notional_scale * raw_pv


class Portfolio:
    """A collection of trades.

    Args:
        trades: list of Trade objects.
        name: portfolio name.
    """

    def __init__(self, trades: list[Trade] | None = None, name: str = ""):
        self.trades = trades or []
        self.name = name

    def add(self, trade: Trade) -> None:
        self.trades.append(trade)

    def pv(self, ctx: PricingContext) -> float:
        """Aggregate PV: sum of all trade PVs."""
        return sum(t.pv(ctx) for t in self.trades)

    def pv_by_trade(self, ctx: PricingContext) -> list[tuple[str, float]]:
        """PV broken down by trade."""
        return [(t.trade_id or f"trade_{i}", t.pv(ctx))
                for i, t in enumerate(self.trades)]

    def __len__(self) -> int:
        return len(self.trades)
