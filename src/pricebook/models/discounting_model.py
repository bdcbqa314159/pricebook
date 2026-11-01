"""The discounting model (L3).

**A model carries the `MarketSnapshot` it was built on.** The engine reaches market data only
through `model.market` — never as a second argument — which is what makes a model/market mismatch
structurally impossible to express rather than merely discouraged.

`DiscountingModel` is the degenerate case: no free parameters, no dynamics. Its "calibration" is
curve construction, so it is *built*, not solved.

Provenance:
  stock:  _stock/src/pricebook/models/discounting_model.py (purified as-is — 30 lines and
          already exactly this)
  source: CLAUDE.md-era §2 (A1: the model carries the market) + doc 22 Q1
  oracle: the snapshot passes through the engine unchanged (test_repricing_is_byte_identical)
  crosscheck: none — carries the MarketSnapshot, computes nothing
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class DiscountingModel:
    """Carries `market` and nothing else; the discounting capability is the snapshot's curves."""

    market: MarketSnapshot
