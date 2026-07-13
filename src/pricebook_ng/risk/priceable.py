"""Priceable — what the risk layer consumes (L5).

The structural fix (spine): risk depends only on a **`Priceable`** — a function
`MarketSnapshot -> PV` — never on concrete product or model classes. Greeks bump
the snapshot and call the priceable; XVA/RWA (later) simulate snapshots and call
it. This is why risk needs no `isinstance`-on-product ladders.

A priceable is the closure `snapshot ↦ price(product, build(snapshot))` (Amendment
A1: a market bump flows through re-building the model). The factories bind a
product + a model-build recipe + an engine into that closure.

Provenance:
  quarry: python/pricebook/risk/ (greeks were on L3, switching on instrument type)
  source: redesign/02_spine.md (risk at L5 on a Priceable protocol)
  oracle: generic dv01 matches analytic across products (L5 greeks slice)
  slice:  L5-risk
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.models.credit_model import CreditModel
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.models.hull_white import HullWhite


@dataclass(frozen=True)
class Priceable:
    """Reprice under a (bumped) market snapshot: `snapshot -> PV`. Risk sees only
    this — no product/model type knowledge."""

    price_at: Callable[[MarketSnapshot], float]

    def __call__(self, snapshot: MarketSnapshot) -> float:
        return self.price_at(snapshot)


def _pv(result) -> float:
    if isinstance(result, PricingFailure):
        raise ValueError(f"priceable priced to a failure: {result.reason}")
    return result.pv.amount


def discounting_priceable(product: object, engine: object, numerics: NumericalConfig) -> Priceable:
    """Bind a linear product to a `DiscountingModel` built from the snapshot,
    priced by `engine` (any engine that consumes a DiscountingModel)."""
    return Priceable(lambda snap: _pv(engine.price(product, DiscountingModel(snap), numerics)))


def hull_white_priceable(
    product: object,
    a: float,
    sigma: float,
    engine: object,
    numerics: NumericalConfig,
) -> Priceable:
    """Bind a product to a Hull-White model (mean reversion `a`, vol `sigma`)
    calibrated to the snapshot; a rate bump rebuilds the model."""
    return Priceable(lambda snap: _pv(engine.price(product, HullWhite(a, sigma, snap), numerics)))


def credit_priceable(
    product: object, recovery: float, engine: object, numerics: NumericalConfig
) -> Priceable:
    """Bind a credit product to a `CreditModel` built from the snapshot (discount +
    survival curve) and `recovery`; a rate or hazard bump rebuilds the model."""
    return Priceable(lambda snap: _pv(engine.price(product, CreditModel(snap, recovery), numerics)))


def fx_forward_priceable(product: object, engine: object, numerics: NumericalConfig) -> Priceable:
    """Bind an FX forward to the snapshot (which carries the FX curves + spots); a
    spot or rate bump rebuilds the discounting model and reprices."""
    return Priceable(lambda snap: _pv(engine.price(product, DiscountingModel(snap), numerics)))
