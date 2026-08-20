"""Priceable — the `snapshot → PV` capability risk depends on (L5).

Risk lands at L5, above the engine, and depends DOWN on it through this protocol — it NEVER
inspects the instrument's type (the engine registry already dispatches by `type(instrument)`;
CLAUDE.md §1 no-isinstance). `priceable(instrument, model_ctor)` closes over an instrument and a
model constructor, so a bumped `MarketSnapshot` reprices it: `market ↦ price(instrument,
model_ctor(market))`. Generic over every product the engine knows — a new product needs no risk code.

Provenance:
  quarry: python/pricebook/risk/priceable.py (the snapshot→PV protocol — design reference only)
  source: CLAUDE.md §1 (risk@L5, Priceable, no-isinstance); redesign/18 §C3 (risk cluster)
  oracle: DV01 vs analytic curve delta; vega vs closed-form Black vega — priced through this
  slice:  risk-greeks (C3 opening)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pricebook_ng.engine import price
from pricebook_ng.foundation import PricingFailure, PricingResult
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.protocols import CalibratedModel


@runtime_checkable
class Priceable(Protocol):
    """The capability risk bumps against: reprice at a (possibly bumped) `MarketSnapshot`."""

    def price_at(self, market: MarketSnapshot) -> PricingResult | PricingFailure: ...


@dataclass(frozen=True)
class _InstrumentPriceable:
    """Closes over an instrument + a model constructor; `price_at` rebuilds the model from the
    (bumped) market and prices through the engine registry — type-blind."""

    instrument: object
    model_ctor: Callable[[MarketSnapshot], CalibratedModel]

    def price_at(self, market: MarketSnapshot) -> PricingResult | PricingFailure:
        return price(self.instrument, self.model_ctor(market))


def priceable(
    instrument: object, model_ctor: Callable[[MarketSnapshot], CalibratedModel]
) -> Priceable:
    """A `Priceable` for `instrument` under the model built by `model_ctor` (e.g. `DiscountingModel`
    for a swap, `BlackModel` for a caplet). Risk stays instrument-type-blind."""
    return _InstrumentPriceable(instrument, model_ctor)
