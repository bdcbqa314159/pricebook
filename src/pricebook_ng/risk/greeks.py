"""Generic bump-and-reprice greeks (L5).

`central_diff` is the KEY-TYPE-BLIND finite-difference core: it applies a `Bump`, reprices through a
`Priceable`, and central-differences — with NO isinstance and NO key-type branch (`bump.apply` is the
only shape-aware call). `ir_delta` and `vol_vega` are thin wrappers supplying a `CurveBump` /
`SurfaceBump` — the two present key kinds that earn the generic core (rule of two, §6b). Failure is a
value: if a bumped reprice returns `PricingFailure`, it propagates (never raised, invariant 4). The
returned sensitivities are the raw derivatives (`∂PV/∂r`, `∂PV/∂σ`); DV01 = `ir_delta · 1bp`,
vega-per-vol-point = `vol_vega · 0.01`.

Provenance:
  quarry: python/pricebook/risk/greeks.py; core/greeks.py (bump-and-reprice — design reference only)
  source: CLAUDE.md §1 (generic greeks, no-isinstance) · §3d (one core, no type-switch); doc 19 §2
  oracle: DV01 vs −Σ cf·t·DF (<1e-6); vega vs df·N·τ·black_vega (<1e-6); one core, two strategies
  slice:  risk-greeks (C3 opening)
"""

from __future__ import annotations

from pricebook_ng.foundation import PricingFailure
from pricebook_ng.market.curve_set import CurveKey
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import SurfaceKey
from pricebook_ng.risk.bump import Bump, CurveBump, SurfaceBump
from pricebook_ng.risk.priceable import Priceable


def central_diff(
    p: Priceable, base: MarketSnapshot, bump: Bump, shift: float
) -> float | PricingFailure:
    """`(PV(base⊕shift) − PV(base⊖shift)) / (2·shift)` — the key-blind FD core. `bump.apply` is the
    ONLY shape-aware call; a failed reprice propagates as a value (invariant 4)."""
    up = p.price_at(bump.apply(base, shift))
    if isinstance(up, PricingFailure):
        return up
    down = p.price_at(bump.apply(base, -shift))
    if isinstance(down, PricingFailure):
        return down
    return (up.pv.amount - down.pv.amount) / (2.0 * shift)


def ir_delta(
    p: Priceable, market: MarketSnapshot, key: CurveKey, shift: float = 1e-4
) -> float | PricingFailure:
    """Parallel interest-rate delta `∂PV/∂r` for the curve at `key` (DV01 = this · 1bp)."""
    return central_diff(p, market, CurveBump(key), shift)


def vol_vega(
    p: Priceable, market: MarketSnapshot, key: SurfaceKey, shift: float = 1e-4
) -> float | PricingFailure:
    """Parallel volatility vega `∂PV/∂σ` for the surface at `key` (per vol point = this · 0.01)."""
    return central_diff(p, market, SurfaceBump(key), shift)
