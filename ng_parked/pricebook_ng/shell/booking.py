"""Booking — the stateful shell over the stateless core (L6).

Domain hierarchy (Amendment A3): a `Trade` is a *collection of products* + a
start date (frozen description); a `Book` collects trades; a `BookedTrade` puts a
trade under monitoring. The shell **remembers** — realized P&L (the **benefit
table**: cashflows that already paid, actual cash, never discounted) — and calls
the core for the **mark** (the PV of what remains). Total economics = realized +
mark. The core never remembers; the shell never computes prices itself.

ponytail: `value()` takes the market + engine directly (no snapshot store yet);
the store arrives with the data-spine slice. A3 wires the benefit table for
fixed (known-amount) cashflows — realized P&L for float legs needs fixings and
lands with a seasoned-float slice.

Provenance:
  quarry: python/pricebook/core/book.py + pnl_history (re-homed core -> L6 shell)
  source: redesign/02_spine.md Amendment A3 (Product/Trade/Book + benefit table)
  oracle: realized (benefit table) + mark reconcile to total economics (A3); dirty =
          clean + accrued; book mark = Σ trade marks (linearity)
  slice:  S00; A3 (Trade/Book + benefit table); l6-trade-lifecycle (mark + reconciliation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pricebook_ng.engine.discounting import CashflowProduct, DiscountingEngine
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.discounting_model import DiscountingModel


def _combine(results: list[PricingResult | PricingFailure]) -> PricingResult | PricingFailure:
    """Sum marks into one (pv + accrued, same currency); the first failure short-circuits
    (failure is a value — engine contract 4). Shared by trade- and book-level marks."""
    pv = 0.0
    accrued = 0.0
    currency = None
    for result in results:
        if isinstance(result, PricingFailure):
            return result
        pv += result.pv.amount
        accrued += result.accrued.amount if result.accrued is not None else 0.0
        currency = result.pv.currency
    return PricingResult(pv=Money(pv, currency), accrued=Money(accrued, currency))


@dataclass(frozen=True)
class Trade:
    """A trade: a collection of products with a common start date (Amendment A3).
    Frozen description; its lifecycle state lives on the `BookedTrade`."""

    products: tuple[CashflowProduct, ...]
    start_date: date

    def _cashflows(self):
        for product in self.products:
            yield from product.cashflows

    def realized(self, as_of: date) -> Money:
        """Benefit table: total of cashflows that have paid by `as_of` — actual
        cash, undiscounted. (Fixed/known amounts; float legs need fixings.)"""
        paid = [cf for cf in self._cashflows() if cf.date <= as_of]
        currency = paid[0].amount.currency if paid else next(self._cashflows()).amount.currency
        return Money(sum(cf.amount.amount for cf in paid), currency)

    def mark(
        self, market: MarketSnapshot, numerics: NumericalConfig, engine: DiscountingEngine
    ) -> PricingResult | PricingFailure:
        """The mark: sum of the products' PVs + accrued as of the snapshot. The shell binds
        the model to the snapshot (A1); the engine excludes already-paid flows (A2, which the
        benefit table remembers) — so `realized + mark` is the trade's total economics."""
        model = DiscountingModel(market)
        return _combine([engine.price(product, model, numerics) for product in self.products])


@dataclass(frozen=True)
class Book:
    """A collection of trades (Amendment A3)."""

    trades: tuple[Trade, ...]

    def realized(self, as_of: date) -> Money:
        totals = [t.realized(as_of) for t in self.trades]
        return Money(sum(m.amount for m in totals), totals[0].currency)

    def value(
        self, market: MarketSnapshot, numerics: NumericalConfig, engine: DiscountingEngine
    ) -> PricingResult | PricingFailure:
        """The book's mark: the sum of its trades' marks (linearity), matching the sum of
        their realized P&L."""
        return _combine([t.mark(market, numerics, engine) for t in self.trades])


@dataclass
class BookedTrade:
    """A trade under monitoring: its description + the marks observed so far."""

    trade: Trade
    results: list[PricingResult | PricingFailure] = field(default_factory=list)

    def realized(self, as_of: date) -> Money:
        return self.trade.realized(as_of)

    def value(
        self,
        market: MarketSnapshot,
        numerics: NumericalConfig,
        engine: DiscountingEngine,
    ) -> PricingResult | PricingFailure:
        """The mark as of the snapshot, remembered on the booked trade (the shell records
        what it observes). Delegates the pricing to the frozen `Trade`."""
        mark = self.trade.mark(market, numerics, engine)
        self.results.append(mark)
        return mark


def book(trade: Trade) -> BookedTrade:
    """Begin a trade's life in the shell."""
    return BookedTrade(trade=trade)
