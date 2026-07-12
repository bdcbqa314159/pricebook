"""Booking — the stateful shell over the stateless core.

L6: `book(trade)` begins a trade's life; `BookedTrade.value(...)` runs the
engine and STORES the result. The shell remembers; it never computes prices
itself (redesign/02_spine.md). The booked trade's state is its result history,
never a cached price — price is always recomputed statelessly.

ponytail: Slice 0 has no snapshot store, so `value()` takes the market and
engine directly instead of pulling `snapshot(date)`. The store arrives with the
data-spine slice.

Provenance:
  quarry: python/pricebook/core/book.py (re-homed core -> L6 shell, minimal)
  source: redesign/02_spine.md (stateful lifecycle over stateless core)
  oracle: shell result == direct engine price (Slice 0 shell-path oracle)
  slice:  S00
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.instruments.fixed_cashflow import FixedCashflowTrade
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel


@dataclass
class BookedTrade:
    """A trade under monitoring: its description + the results observed so far."""

    trade: FixedCashflowTrade
    results: list[PricingResult | PricingFailure] = field(default_factory=list)

    def value(
        self,
        market: MarketSnapshot,
        numerics: NumericalConfig,
        engine: DiscountingEngine,
    ) -> PricingResult | PricingFailure:
        # the shell builds/binds the model for this date's snapshot, then prices (A1)
        result = engine.price(self.trade, DiscountingModel(market), numerics)
        self.results.append(result)
        return result


def book(trade: FixedCashflowTrade) -> BookedTrade:
    """Begin a trade's life in the shell."""
    return BookedTrade(trade=trade)
