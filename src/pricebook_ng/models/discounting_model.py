"""DiscountingModel — the degenerate calibrated model for linear products (L3).

A model carries the `MarketSnapshot` it was calibrated to (A1); the engine reaches the
market ONLY through it, never as a peer argument, so a model/market mismatch cannot be
expressed. `DiscountingModel` has no free parameters and no dynamics — its "calibration"
is curve construction, so it is BUILT, not solved (doc 22 Q1). Single-curve: the discount
curve is also the projection curve. Model-level `df`/`forward` capabilities under
dynamics arrive with the first dynamics model, each with its second consumer and its
semantic contract (doc 22 Q3′, rule of two).

Provenance:
  quarry: python/pricebook/pricing/ (discounting model)
  source: redesign/22 Q1 (CalibratedModel · DiscountingModel); CLAUDE.md §2 (A1, frozen)
  oracle: model.market carried through the engine unchanged; par swap → zero NPV
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class DiscountingModel:
    """The linear model: it carries `market` (A1) and nothing else. The discounting
    capability is the snapshot's curve, reached as `model.market.discount_curve`."""

    market: MarketSnapshot
