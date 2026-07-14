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

from collections.abc import Callable
from dataclasses import replace

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.market.keys import MarketKey
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.risk.priceable import Priceable

_ONE_BP = 1e-4
_Bump = Callable[[MarketSnapshot, float], MarketSnapshot]


def _central_diff(
    priceable: Priceable, snapshot: MarketSnapshot, bump: _Bump, numerics: NumericalConfig
) -> float:
    """Raw central finite difference of `priceable` under `bump`."""
    h = numerics.fd_bump
    return (priceable(bump(snapshot, h)) - priceable(bump(snapshot, -h))) / (2.0 * h)


# ---- rate risk on the home discount curve -------------------------------------
def bump_rate(snapshot: MarketSnapshot, dr: float) -> MarketSnapshot:
    """Parallel shift of the home discount curve by `dr` (uses the curve's own
    `bumped` — a rate shift for the flat curve, a uniform zero shift across every
    pillar for a bootstrapped curve)."""
    return replace(snapshot, discount_curve=snapshot.discount_curve.bumped(dr))


def dv01(priceable: Priceable, snapshot: MarketSnapshot, numerics: NumericalConfig) -> float:
    """PV change per 1bp parallel rate rise on the home discount curve."""
    return _central_diff(priceable, snapshot, bump_rate, numerics) * _ONE_BP


# ---- generic curve risk on any keyed curve (A5-style: FX foreign, dividend,
#      real/breakeven, survival) ------------------------------------------------
def bump_curve(snapshot: MarketSnapshot, key: MarketKey, shift: float) -> MarketSnapshot:
    """Parallel-shift the curve at `key` by `shift`, as a new snapshot — polymorphic
    via the curve's own `bumped` (rate for a discount curve, hazard for a survival
    curve)."""
    curves = dict(snapshot.curves)
    curves[key] = snapshot.curves[key].bumped(shift)
    return replace(snapshot, curves=curves)


def curve01(
    priceable: Priceable, snapshot: MarketSnapshot, key: MarketKey, numerics: NumericalConfig
) -> float:
    """PV change per 1bp parallel shift of the curve at `key` — rate risk on the FX
    foreign / dividend / real (breakeven) curve, or CS01 on a survival curve."""
    return _central_diff(priceable, snapshot, lambda s, d: bump_curve(s, key, d), numerics) * _ONE_BP


def credit01(
    priceable: Priceable, snapshot: MarketSnapshot, key: MarketKey, numerics: NumericalConfig
) -> float:
    """CS01 — PV change per 1bp hazard widening. A named alias of `curve01` on a
    survival curve (bumping a survival curve shifts its hazard)."""
    return curve01(priceable, snapshot, key, numerics)


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
