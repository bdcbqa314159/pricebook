"""MarketSnapshot — the immutable market state a model carries (L1).

The valuation date, the `CurveSet`, and (this slice) the `scalars` shape — the first non-curve
shape of doc 19's closed-shapes × open-keys design. `scalars` holds FX spots keyed by pair, in
their DECLARED canonical quote order; `fx_rate(base, quote)` asserts the pair is declared,
inverts internally, and RAISES on an undeclared cross — never a bare pair-scalar (doc 19 §2.1).
Surfaces · series · schedules still arrive with their first consumer. Frozen (A1, invariant 3).

Provenance:
  quarry: python/pricebook/pricing/market_data_provider.py
  source: redesign/19 §1-§3 (snapshot shapes; CurveSet) · §2.1 (FX directional); CLAUDE.md §2 (A1)
  oracle: fx_rate(A,B)·fx_rate(B,A) == 1; an undeclared cross raises
  slice:  fx-spot (T1 slice 6a)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Mapping

from pricebook_ng.foundation import Currency, CurrencyPair, fx_pair
from pricebook_ng.market.curve_set import CurveSet


@dataclass(frozen=True)
class ScalarKey:
    """A key into the snapshot's `scalars` shape. This slice: an FX spot, keyed by its pair
    (a new asset adds keys, not fields). Carries the pair only — rule of two."""

    pair: CurrencyPair


@dataclass(frozen=True)
class MarketSnapshot:
    """The market state a model was calibrated to (A1). `valuation_date` is 'today';
    `curves` is the keyed `CurveSet`; `scalars` holds FX spots (and future single-number
    market data), reached through typed accessors — never as bare values."""

    valuation_date: date
    curves: CurveSet
    scalars: Mapping[ScalarKey, float] = field(default_factory=dict)

    def fx_rate(self, base: Currency, quote: Currency) -> float:
        """One unit of `base` in units of `quote`. Resolves the declared canonical pair
        (raising on an undeclared cross), reads the stored spot, and inverts if `base/quote`
        is the reverse of the declared direction — the rate is never ambiguous (doc 19 §2.1)."""
        canonical = fx_pair(base, quote)
        spot = self.scalars[ScalarKey(canonical)]
        if canonical.base == base and canonical.quote == quote:
            return spot
        return 1.0 / spot
