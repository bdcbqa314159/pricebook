"""MarketSnapshot — the immutable market state a model carries (L1).

The valuation date plus the `CurveSet` (doc 19 §1-§3). Doc 19's other closed shapes —
surfaces · scalars · series · schedules — arrive with their first consumer (vols, FX
spot, fixings); a new asset adds keys to the CurveSet, not fields here. Frozen and never
mutated (CLAUDE.md §2, invariant 3).

Provenance:
  quarry: python/pricebook/pricing/market_data_provider.py
  source: redesign/19 §1-§3 (snapshot = the state pricing reads; CurveSet); CLAUDE.md §2 (A1)
  oracle: carried unchanged through the engine; a EURIBOR swap prices to zero dual-curve
  slice:  dual-curve-euribor-estr (T1 slice 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.market.curve_set import CurveSet


@dataclass(frozen=True)
class MarketSnapshot:
    """The market state a model was calibrated to (A1). `valuation_date` is 'today';
    `curves` is the keyed `CurveSet` (discount + projection), reached through its
    typed accessors."""

    valuation_date: date
    curves: CurveSet
