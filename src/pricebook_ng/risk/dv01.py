"""DV01 — sensitivity to a 1bp parallel rate move, by bumping the snapshot.

L5: risk lives above the engine and works by re-running it under bumped markets
(CLAUDE.md 1, spine). Central finite difference on the rate gives dPV/dr; DV01
is that times 1bp. The bump step is the explicit `NumericalConfig.fd_bump`.

Bumping builds NEW frozen curves/snapshots — the input market is never mutated,
so statelessness holds.

Provenance:
  quarry: python/pricebook/risk/ (bump-and-reprice greeks)
  source: redesign/02_spine.md (risk on the engine, not on instruments)
  oracle: analytic DV01 = -N t exp(-r t) * 1e-4 vs central FD < 1e-6 (Slice 0)
  slice:  S00
"""

from __future__ import annotations

from dataclasses import replace

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.products.fixed_cashflow import FixedCashflow
from pricebook_ng.models.discounting_model import DiscountingModel

_ONE_BP = 1e-4


def _pv(engine, trade, model, numerics) -> float:
    result = engine.price(trade, model, numerics)
    if not isinstance(result, PricingResult):
        raise ValueError(f"DV01 bump priced to a failure: {result}")
    return result.pv.amount


def dv01(
    engine,
    trade: FixedCashflow,
    model: DiscountingModel,
    numerics: NumericalConfig,
) -> float:
    """Central-difference DV01 (PV change per 1bp rate rise).

    A market bump flows through the model (Amendment A1): bump the snapshot's
    curve, rebuild the `DiscountingModel`, reprice.

    ponytail: Slice 0 assumes a flat, single-rate curve — the bump shifts that
    one rate. The general parallel-shift over a bootstrapped curve arrives with
    a later risk slice, behind the same CurveHandle.
    """
    h = numerics.fd_bump
    market = model.market
    curve = market.discount_curve
    up = DiscountingModel(replace(market, discount_curve=replace(curve, rate=curve.rate + h)))
    down = DiscountingModel(replace(market, discount_curve=replace(curve, rate=curve.rate - h)))
    dpv_dr = (_pv(engine, trade, up, numerics) - _pv(engine, trade, down, numerics)) / (2 * h)
    return dpv_dr * _ONE_BP
