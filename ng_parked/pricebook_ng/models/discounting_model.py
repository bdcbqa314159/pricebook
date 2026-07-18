"""Calibrated-model protocol and the discounting model for linear products (L3).

Amendment A1: the engine depends on the model, and a model **carries the
`MarketSnapshot` it was calibrated to** (`model.market`) — market is upstream of
the model (`market -> calibrate -> model -> price`), never a peer argument to the
engine. This makes a model/market mismatch structurally impossible.

`DiscountingModel` is the thin model linear products (cashflows, bonds, swaps)
use: its "calibration" trivially adopts the snapshot. Real consumers: every
linear product (rule of two) — not ceremony.

Provenance:
  quarry: python/pricebook/core/pricing_context.py (built-market-state input)
  source: redesign/02_spine.md Amendment A1
  oracle: engine-model binding test + unchanged PVs (A1)
  slice:  A1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pricebook_ng.market.snapshot import MarketSnapshot


@runtime_checkable
class CalibratedModel(Protocol):
    """A model bound to the market it was calibrated to. The engine reaches the
    economy (curves, fixings, valuation date) through `market`."""

    @property
    def market(self) -> MarketSnapshot: ...


@dataclass(frozen=True)
class DiscountingModel:
    """The degenerate model for linear products: deterministic discounting on the
    snapshot's curve. Its dynamics are 'none'; it exists to carry the market."""

    market: MarketSnapshot
