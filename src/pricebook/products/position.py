"""The `Position` — a collection of trades (L2). Frozen data; prices nothing.

**R1.2, the second representation piece.** T82's `Trade` is a holding of one contract; a `Position` is
a holding of several. That is the whole of it as *data* — the interesting content is in how the
aggregate's **error** combines, which lives with the function that computes it (`engine/position.py`).

**One field, and it stays one.** A position is its trades. A name, an id, a book label and a trade date
are all plausible and none is needed by anything yet, so none is here (§6 — nothing speculative); when
two consumers want them they bundle into a value object rather than widening this.

Provenance:
  stock:  none — architecture-first; the structural track anchors on the catalogue
  source: CLAUDE.md §0/§1 — the representation contract and the purpose-defined 1.0 ("represent, value
          and risk a position")
  oracle: tests/L4/test_position.py — the aggregation identity as a guard, and the standard-error
          combination in both regimes (linear within a `(product, model)` group, in quadrature across
          groups) checked against exact known values
  crosscheck: none — frozen representation data
  slice:  tranche 83 — R1.2, the Position type
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.money import Currency
from pricebook.products.trade import Trade


@dataclass(frozen=True)
class Position:
    """The trades held, and the currency the book reports in. Order carries no meaning.

    **The `currency` is forced by the empty book, not added speculatively.** The spec asked for an
    empty position to be worth `PV 0` — and a `Money` needs a currency that an empty collection of
    trades cannot supply. There is no honest default: a book with nothing in it still reports in
    *some* currency, and only the book knows which. Declaring it also gives the aggregator the
    reference it needs to **refuse** a book whose trades price in a different currency, which the
    first draft summed silently (dollars and euros added together). "Multi-currency aggregation is
    out of scope" licenses not *supporting* cross-currency netting; it does not license adding
    unlike currencies and reporting a number.
    """

    trades: tuple[Trade, ...]
    currency: Currency
