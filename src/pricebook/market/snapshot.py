"""The market snapshot (L1) — immutable market state, carrying its own valuation date.

**The valuation date lives here, not in a pricing call.** There is no clock read anywhere in the
library: "today" is `model.market.valuation_date`, which is what makes a reprice reproducible
months later and makes `price` a pure function of its arguments.

Provenance:
  stock:  _stock/src/pricebook/market/snapshot.py (purified: valuation_date + curves + fixing
          series. Its
          `scalars`/`surfaces`/`series` shapes and their key-unions grew one asset class at a
          time; each returns when a product pulls it)
  source: CLAUDE.md-era §2 invariant 2 (no ambient state) + doc 19 §2
  oracle: the snapshot is never mutated across a reprice (test_repricing_is_byte_identical)
  crosscheck: none — a frozen market-state value bag, no numerics
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date

from pricebook.foundation.inflation import InflationIndex
from pricebook.market.curve_set import CurveSet
from pricebook.market.fixing_series import FixingSeries


@dataclass(frozen=True)
class MarketSnapshot:
    """Everything the market said at `valuation_date`. Frozen; a bump produces a new snapshot."""

    valuation_date: date
    curves: CurveSet
    #: The published print history per index. **A series, not a resolved scalar, since tranche 7.**
    #: Tranche 2 carried `Mapping[InflationIndex, float]` — the already-resolved `I(t)` — which is
    #: enough for a zero-coupon swap at inception but cannot supply the *historical reset*
    #: `I(T_{ι(t)−1})` that a seasoned year-on-year period pays on (eq 2.5.12). The scalar is the
    #: degenerate one-point case of this, which is what makes the replacement provably faithful
    #: rather than a rewrite: see `test_a_one_point_series_reproduces_the_hand_fed_scalar`.
    #: *Which* print a date resolves to is `market.fixing_series.reference_level`, defined once and
    #: composed by every pricer — never re-derived per engine.
    fixings: Mapping[InflationIndex, FixingSeries] = field(default_factory=dict)
