"""Greeks by bump-and-reprice on the Priceable protocol (L5).

Generic over the product/model: a greek central-differences a `Priceable` under a
bump of the snapshot, which rebuilds the model inside the priceable (Amendment
A1). With the keyed market registry (Amendment A5) the spot/vol greeks are
**generic** — one `spot_delta`/`vol_vega` for FX, equity, commodity, … keyed by
`MarketKey`, not one pair per asset class. Curve greeks stay curve-specific:
`dv01` bumps the home discount rate, `credit01` bumps a survival curve.

Provenance:
  quarry: python/pricebook/risk/ (bump-and-reprice greeks; ex-L3)
  source: redesign/02_spine.md (risk at L5); Amendment A5 (generic greeks)
  oracle: dv01/credit01 analytic; spot_delta/vol_vega match FX & equity analytics
  slice:  S00 (dv01); L5-risk; survival-in-snapshot; A5 (generic on MarketKey)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.keys import MarketKey
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.survival_curve import SurvivalCurve
from pricebook_ng.risk.priceable import Priceable

_ONE_BP = 1e-4
_CURVE_DC = DayCountConvention.ACT_365_FIXED
_Bump = Callable[[MarketSnapshot, float], MarketSnapshot]


def _central_diff(
    priceable: Priceable, snapshot: MarketSnapshot, bump: _Bump, numerics: NumericalConfig
) -> float:
    """Raw central finite difference of `priceable` under `bump`."""
    h = numerics.fd_bump
    return (priceable(bump(snapshot, h)) - priceable(bump(snapshot, -h))) / (2.0 * h)


# ---- rate risk (home discount curve) ------------------------------------------
def bump_rate(snapshot: MarketSnapshot, dr: float) -> MarketSnapshot:
    """Parallel shift of the (flat) home discount curve by `dr`.

    ponytail: single-rate parallel shift for the flat curve; a pillar-wise shift of
    a bootstrapped curve is a later greek slice, behind the same protocol."""
    curve = snapshot.discount_curve
    return replace(snapshot, discount_curve=replace(curve, rate=curve.rate + dr))


def dv01(priceable: Priceable, snapshot: MarketSnapshot, numerics: NumericalConfig) -> float:
    """PV change per 1bp parallel rate rise."""
    return _central_diff(priceable, snapshot, bump_rate, numerics) * _ONE_BP


# ---- credit risk (a survival curve in the registry) ---------------------------
def bump_hazard(snapshot: MarketSnapshot, key: MarketKey, dh: float) -> MarketSnapshot:
    """Parallel hazard shift of the survival curve at `key`: scale each pillar by
    exp(-dh*t), as a new snapshot."""
    survival = snapshot.curves[key]
    assert isinstance(survival, SurvivalCurve)
    v = survival.valuation_date
    pillars = tuple(
        (d, q * math.exp(-dh * year_fraction(v, d, _CURVE_DC))) for d, q in survival.pillars
    )
    curves = dict(snapshot.curves)
    curves[key] = replace(survival, pillars=pillars)
    return replace(snapshot, curves=curves)


def credit01(
    priceable: Priceable, snapshot: MarketSnapshot, key: MarketKey, numerics: NumericalConfig
) -> float:
    """PV change per 1bp parallel credit-spread (hazard) widening on `key` (CS01)."""
    return _central_diff(priceable, snapshot, lambda s, d: bump_hazard(s, key, d), numerics) * _ONE_BP


# ---- spot & vol risk (generic over MarketKey, A5) -----------------------------
def bump_spot(snapshot: MarketSnapshot, key: MarketKey, ds: float) -> MarketSnapshot:
    """Shift the spot at `key` by `ds`, as a new snapshot (only that spot moves)."""
    spots = dict(snapshot.spots)
    spots[key] = spots[key] + ds
    return replace(snapshot, spots=spots)


def bump_vol(snapshot: MarketSnapshot, key: MarketKey, dvol: float) -> MarketSnapshot:
    """Shift the vol at `key` by `dvol`, as a new snapshot (only that vol moves)."""
    vols = dict(snapshot.vols)
    vols[key] = vols[key] + dvol
    return replace(snapshot, vols=vols)


def spot_delta(
    priceable: Priceable, snapshot: MarketSnapshot, key: MarketKey, numerics: NumericalConfig
) -> float:
    """PV change per unit move in the spot at `key` (FX delta, equity delta, …)."""
    return _central_diff(priceable, snapshot, lambda s, d: bump_spot(s, key, d), numerics)


def vol_vega(
    priceable: Priceable, snapshot: MarketSnapshot, key: MarketKey, numerics: NumericalConfig
) -> float:
    """PV change per unit move in the vol at `key` (FX vega, equity vega, …)."""
    return _central_diff(priceable, snapshot, lambda s, d: bump_vol(s, key, d), numerics)
