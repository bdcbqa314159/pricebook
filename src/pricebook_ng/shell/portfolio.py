"""The imperative shell — frozen Trade/Book + marking + portfolio risk (L6).

Functional core, imperative shell: the shell HOLDS descriptions and CALLS the core; it never
prices (there is no `pv()` on a product, trade, or book — pricing lives in L4) and the core never
remembers. `Trade`/`Book` are pure-data descriptions; `mark` is a shell FUNCTION that sums the
engine's PV over the products (the quarry's `Trade.pv(ctx)` self-pricing is realigned away here).
Portfolio DV01 is Σ of the L5 greek — the shell composes risk, adds no risk code. Single-model:
ONE model marks the whole book; a rich `BlackModel` (curves + surfaces) marks a MIXED book
(swaps + caplets + swaptions) via the registry's per-type dispatch — NO isinstance on a product.

The stateful half — `BookedTrade` + benefit table + realized P&L (the "shell remembers realized
cash") — is the next L6 slice; this slice's `mark` is the **future PV** (invariant 6: past cashflows
are excluded upstream). L6 depends DOWN on L4/L5/L2 and is imported by NOTHING (`verify.py acyclic`).

Provenance:
  quarry: python/pricebook/core/trade.py; core/book.py (Trade/Portfolio/Book — realigned, not cloned)
  source: CLAUDE.md §1 (L6 shell, no-isinstance) · §2 (trade vs mark vs realized; shell calls core) · §3
  oracle: trade-of-1 == engine PV; book == Σ trade marks; portfolio DV01 == Σ per-product ir_delta
  slice:  l6-shell (C3 / L6 opening)
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date

from pricebook_ng.engine import price
from pricebook_ng.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.market.curve_set import CurveKey
from pricebook_ng.risk import Priceable, ir_delta, priceable

Product = object  # a registered L2 product; the engine dispatches by type, so the shell stays type-blind


@dataclass(frozen=True)
class Trade:
    """A trade: ≥1 pure-data `products` and a `start_date` (CLAUDE.md §3). No `pv()` — the shell
    marks it by calling the engine. Frozen description; lifecycle/realized P&L land with `BookedTrade`."""

    products: tuple[Product, ...]
    start_date: date

    def __post_init__(self) -> None:
        if not self.products:
            raise ValueError("a Trade holds at least one product (CLAUDE.md §3).")


@dataclass(frozen=True)
class Book:
    """A book: a collection of `trades` under an optional `name` (CLAUDE.md §3). No `pv()`."""

    trades: tuple[Trade, ...]
    name: str = ""


def add_money(by_ccy: dict[Currency, Money], money: Money) -> None:
    """Accumulate `money` into the per-currency map (never mixes currencies — that's the point)."""
    existing = by_ccy.get(money.currency)
    by_ccy[money.currency] = money if existing is None else existing + money


def mark(trade: Trade, model: CalibratedModel) -> Mapping[Currency, Money] | PricingFailure:
    """The trade's mark (future PV) as a PER-CURRENCY map: Σ over its products of
    `engine.price(product, model).pv`, grouped by currency. A single-currency trade is the degenerate
    1-entry map; a mixed trade returns >1 entry, never a `TypeError` (invariant 4, #2). A product that
    fails to price propagates as a `PricingFailure`."""
    by_ccy: dict[Currency, Money] = {}
    for product in trade.products:
        result = price(product, model)
        if isinstance(result, PricingFailure):
            return result
        add_money(by_ccy, result.pv)
    return by_ccy


def mark_book(book: Book, model: CalibratedModel) -> Mapping[Currency, Money] | PricingFailure:
    """The book's mark: the per-currency merge of every trade's mark. A failing trade propagates;
    an empty book is a `PricingFailure` (no currency)."""
    by_ccy: dict[Currency, Money] = {}
    for trade in book.trades:
        m = mark(trade, model)
        if isinstance(m, PricingFailure):
            return m
        for money in m.values():
            add_money(by_ccy, money)
    if not by_ccy:
        return PricingFailure("empty book: mark is undefined (no currency).")
    return by_ccy


def book_priceable(
    book: Book, model_ctor: Callable[[MarketSnapshot], CalibratedModel]
) -> Priceable:
    """Adapt a `Book` to the L5 `Priceable` protocol (`snapshot → PV`), so EVERY L5 greek works on a
    whole portfolio for free. A `Priceable` carries ONE `Money`, so this is defined for a
    single-currency book; a multi-currency book returns a `PricingFailure` (base-currency conversion
    via `snapshot.fx_rate` is deferred — its trigger is the reporting consumer)."""

    @dataclass(frozen=True)
    class _BookPriceable:
        book: Book
        model_ctor: Callable[[MarketSnapshot], CalibratedModel]

        def price_at(self, market: MarketSnapshot) -> PricingResult | PricingFailure:
            m = mark_book(self.book, self.model_ctor(market))
            if isinstance(m, PricingFailure):
                return m
            if len(m) != 1:
                return PricingFailure(
                    f"book spans {sorted(c.code for c in m)} — a single-currency Priceable only "
                    f"(base-currency conversion deferred)"
                )
            (money,) = m.values()
            return PricingResult(pv=money)

    return _BookPriceable(book, model_ctor)


def book_dv01(book: Book, model: CalibratedModel, key: CurveKey) -> float | PricingFailure:
    """Portfolio DV01: `ir_delta` of the whole book (== Σ per-product DV01 by linearity). Reuses the
    L5 greek verbatim on the book-as-`Priceable`. (`key` names the curve to bump — same shape as L5
    `ir_delta(p, market, key)`; the ratified `book_dv01(book, model)` couldn't say WHICH curve.)"""
    return ir_delta(book_priceable(book, type(model)), model.market, key)
