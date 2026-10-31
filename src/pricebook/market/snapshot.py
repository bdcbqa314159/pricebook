"""The market snapshot (L1) — immutable market state, carrying its own valuation date.

**The valuation date lives here, not in a pricing call.** There is no clock read anywhere in the
library: "today" is `model.market.valuation_date`, which is what makes a reprice reproducible
months later and makes `price` a pure function of its arguments.

Provenance:
  stock:  _stock/src/pricebook/market/snapshot.py (purified: valuation_date + curves. Its
          `scalars`/`surfaces`/`series` shapes and their key-unions grew one asset class at a
          time; each returns when a product pulls it)
  source: CLAUDE.md-era §2 invariant 2 (no ambient state) + doc 19 §2
  oracle: the snapshot is never mutated across a reprice (test_repricing_is_byte_identical)
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.market.curve_set import CurveSet


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the market said at `valuation_date`. Frozen; a bump produces a new snapshot."""

    valuation_date: date
    curves: CurveSet
