"""BookedTrade + realized P&L + total economics — the stateful shell (L6).

The shell REMEMBERS realized cash; the engine COMPUTES the mark. `BookedTrade` wraps a frozen `Trade`
with the `FixingSource` needed to value its historical float coupons. `realized` is the benefit table
(Σ past undiscounted cash, §2), `total` = realized + mark. This is the "shell remembers realized" half
of "Total = realized + mark" — the mark (`mark`, from the frozen-shell slice) is the FUTURE PV, which
the engine now excludes historical flows from (invariant 6), so realized and mark never double-count.

Lifecycle EVENTS (exercise/amend/novate), accrued/clean-dirty reporting, and booking persistence are
deferred to their slices; `BookedTrade` carries only the fixings this slice needs.

Provenance:
  quarry: python/pricebook/core/trade.py (BookedTrade / realized-vs-mark — realigned, not cloned)
  source: CLAUDE.md §2 (Total = realized + mark; shell remembers, engine computes) · §3 (BookedTrade)
  oracle: seasoned realized == Σ past coupons (undiscounted); total == realized + mark; spot → realized 0
  slice:  l6-realized (L6 stateful; closes C3/T1)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from pricebook_ng.foundation import Currency, Money, PricingFailure
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.shell.benefit import benefit
from pricebook_ng.shell.portfolio import Trade, add_money, mark


@dataclass(frozen=True)
class BookedTrade:
    """A booked trade: the frozen `trade` (lifecycle events land with their slice). Fixings are NOT
    held here — they are market state on the `MarketSnapshot` (one source, A1); `realized` re-sources
    them from the model's market (#3b)."""

    trade: Trade


def realized(booked: BookedTrade, model: CalibratedModel) -> Mapping[Currency, Money] | PricingFailure:
    """The benefit table as a PER-CURRENCY map: Σ over the trade's products of already-paid,
    UNDISCOUNTED cash (the shell's memory, never a PV). Fixings + valuation date come from the model's
    market (one source, A1). Single-currency = the degenerate 1-entry map (#2). A product whose benefit
    can't be resolved (a missing historical fixing) propagates as a `PricingFailure` (invariant 4)."""
    by_ccy: dict[Currency, Money] = {}
    for product in booked.trade.products:
        b = benefit(product, model.market.valuation_date, model.market.fixings)
        if isinstance(b, PricingFailure):
            return b
        add_money(by_ccy, b)
    return by_ccy


def total(booked: BookedTrade, model: CalibratedModel) -> Mapping[Currency, Money] | PricingFailure:
    """Total economics = realized + mark (§2), per currency: the shell's realized cash plus the
    engine's future-PV mark (historical excluded by invariant 6, so no double count). Either failing
    propagates."""
    r = realized(booked, model)
    if isinstance(r, PricingFailure):
        return r
    m = mark(booked.trade, model)
    if isinstance(m, PricingFailure):
        return m
    by_ccy: dict[Currency, Money] = {}
    for money in (*r.values(), *m.values()):
        add_money(by_ccy, money)
    return by_ccy
