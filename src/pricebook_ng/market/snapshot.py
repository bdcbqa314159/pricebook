"""MarketSnapshot — the immutable market state a model carries (L1).

Minimal for the single-curve slice: the valuation date and the discount curve.
Doc 19's full closed-shapes×open-keys `CurveSet` (discount·projection·survival·
surfaces·scalars·…) arrives with its SECOND curve family — the projection curve at
multicurve — the consumer that earns the typed-accessor abstraction (rule of two,
CLAUDE.md §6b). Frozen and never mutated (CLAUDE.md §2, invariant 3).

Provenance:
  quarry: python/pricebook/pricing/market_data_provider.py
  source: redesign/19 §1 (snapshot = the state pricing reads); CLAUDE.md §2 (A1, frozen)
  oracle: carried unchanged through the engine; reprice is stateless / byte-identical
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.market.curve import CurveHandle


@dataclass(frozen=True)
class MarketSnapshot:
    """The market state a model was calibrated to (A1). `valuation_date` is 'today';
    `discount_curve` is the numeraire curve, reached through its `CurveHandle`."""

    valuation_date: date
    discount_curve: CurveHandle
