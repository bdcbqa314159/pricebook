"""`position_value` — a book's value and its ERROR (L4). R1.2.

`PV = Σ direction_i · units_i · price(product_i, model_i)`, which telescopes and is therefore the
guard rather than the test. The tranche's real content is the **standard error**:

    SE_book = √( Σ_groups ( Σ_within-group SE )² )      where a group is one `(product, model)` pair

**Why not `√(Σ SE_i²)` throughout, which is the form one reaches for first.** §3.1 makes every pricer
referentially transparent — identical inputs give identical output — and that is *verified* on a real
MC engine in the oracle, not assumed: dispatching the same `(product, model)` twice is **bit-identical**
in PV *and* SE. So two trades of the same product and model do not carry *independent* errors; they
carry **the same** error. Their bands add **linearly**, and treating them as independent would
**understate** the book — for the entirely ordinary book that holds two lots of one contract.

So the two halves have different standing, and the difference is worth keeping visible:

* **within a group the linear sum is a FACT** about this library, forced by §3.1's determinism;
* **across groups independence is an ASSUMPTION.** Different pricers use different fixed seeds and
  different payoffs, which makes their errors different but not provably uncorrelated. Books whose
  trades genuinely share Monte-Carlo paths would need the linear regime throughout; nothing in the
  tree does that today, and if a later unit shares paths across trades, this is where it must be
  revisited.

A band has **no sign**, so a long and an offsetting short of the same contract net to zero *value* with
their bands **adding**: the uncertainty about a netted zero is not zero. That falls out of
`trade_value`'s `abs(scale)` and is oracled explicitly, because a reader may reasonably expect the
cancellation to carry through.

Provenance:
  stock:  none — architecture-first
  source: CLAUDE.md §0/§1 — the representation contract; §3.1 (referential transparency) is what makes
          the within-group combination a fact rather than a convention, and §3.7 is why a book carries
          a band at all
  oracle: tests/L4/test_position.py — determinism asserted first, then both SE regimes against exact
          known values, the mixed exact/estimate book, netting on a non-symmetric payoff, order
          invariance, and the whole-book failure on one unpriceable trade
  crosscheck: none — no library prices an abstract book wrapper; the underlying engines carry theirs
  slice:  tranche 83 — R1.2, the Position type
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pricebook.engine.trade import trade_value
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.products.position import Position


def position_value(
    position: Position, models: Sequence[object]
) -> PricingResult | PricingFailure:
    """The book's value, or the failure of the first trade that could not be priced.

    **One unpriceable trade fails the whole book.** Skipping it and totalling the rest would return a
    *number* that silently omits a position, which is worse than no number — §3.4's reasoning one
    layer up. The failure names the product so the caller can act on it.

    `models` is one model per trade, positionally. Two parallel sequences are the honest minimum
    until R1.3's `PricingRequest` resolves models from the market; the length is **checked** rather
    than `zip`-ed, because `zip` truncates silently and would return a confident partial total.
    """
    if len(models) != len(position.trades):
        raise ValueError(
            f"position_value needs one model per trade: {len(position.trades)} trades, "
            f"{len(models)} models"
        )

    total = 0.0
    # `(product, model)` → the summed band of the trades sharing it, grouped **by equality and not
    # by identity**. The first draft keyed on `id(model)`, so two structurally equal models were
    # different groups and the within-group linear sum never fired — the grouping silently degraded
    # to the independent regime, which is the understating direction. Equality is the right test
    # because §3.1 is about identical *inputs*: a frozen dataclass compares structurally, and a
    # list scan rather than a dict is required because `MarketSnapshot` holds mappings and so is
    # unhashable.
    grouped: list[tuple[object, object, float]] = []

    for trade, model in zip(position.trades, models):
        priced = trade_value(trade, model)
        if isinstance(priced, PricingFailure):
            return PricingFailure(
                f"the book cannot be valued: {type(trade.product).__name__} — {priced.reason}"
            )
        if priced.pv.currency != position.currency:
            return PricingFailure(
                f"the book reports in {position.currency.code} but "
                f"{type(trade.product).__name__} prices in {priced.pv.currency.code} — unlike "
                "currencies cannot be summed, and cross-currency netting is not modelled yet"
            )
        total += priced.pv.amount
        if priced.standard_error is not None:
            for index, (product, grouped_model, band) in enumerate(grouped):
                if product == trade.product and grouped_model == model:
                    grouped[index] = (product, grouped_model, band + priced.standard_error)
                    break
            else:
                grouped.append((trade.product, model, priced.standard_error))

    # No band at all when nothing in the book is an estimate — including the empty book, which is
    # not an estimate either. A `0.0` would claim a *measured* zero uncertainty, a stronger and
    # different statement than "there is no error bar here".
    combined = (
        math.sqrt(sum(band * band for _, _, band in grouped)) if grouped else None
    )
    return PricingResult(pv=Money(total, position.currency), standard_error=combined)
