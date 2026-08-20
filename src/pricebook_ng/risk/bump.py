"""Bump strategies — the per-shape market perturbation risk applies (L5).

The ONLY shape-aware part of the risk layer: a `Bump` knows how to perturb ONE snapshot shape
(a curve, a surface) and return a NEW `MarketSnapshot` (invariant 3 — the base is never mutated).
The finite-difference core is key-TYPE-BLIND; the curve-vs-surface split lives HERE, in polymorphic
`apply`, never as an `isinstance`/`if-elif` in the core (§1 no-isinstance; §3d no type-switch). A
third key kind (FX-spot `ScalarBump`) arrives with its first consumer — a new strategy, not a core edit.

Provenance:
  quarry: python/pricebook/risk/ (curve_bumper — design reference only, built on the superseded shape)
  source: CLAUDE.md §1 (no-isinstance) · §2 (invariant 3, no mutation); redesign/19 §2 (per-shape keys)
  oracle: bumped market reprices; base snapshot/curve/surface unchanged after the bump
  slice:  risk-greeks (C3 opening)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from pricebook_ng.market.curve_set import CurveKey
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.vol_surface import SurfaceKey


@runtime_checkable
class Bump(Protocol):
    """Perturb one snapshot shape by `shift`, returning a new snapshot (never mutates the base)."""

    def apply(self, market: MarketSnapshot, shift: float) -> MarketSnapshot: ...


@dataclass(frozen=True)
class CurveBump:
    """Parallel-bumps the discount/projection curve at `key` (sticky-model market delta): replace
    it with `.bumped(shift)`, rebuild the snapshot. The unbumped curves/surfaces are shared as-is."""

    key: CurveKey

    def apply(self, market: MarketSnapshot, shift: float) -> MarketSnapshot:
        bumped = market.curves.curves[self.key].bumped(shift)
        return replace(market, curves=market.curves.with_curve(self.key, bumped))


@dataclass(frozen=True)
class SurfaceBump:
    """Parallel-bumps the vol surface at `key` (a flat vega): replace it with `.bumped(shift)`,
    rebuild the snapshot."""

    key: SurfaceKey

    def apply(self, market: MarketSnapshot, shift: float) -> MarketSnapshot:
        bumped = market.surfaces[self.key].bumped(shift)
        return replace(market, surfaces={**market.surfaces, self.key: bumped})
