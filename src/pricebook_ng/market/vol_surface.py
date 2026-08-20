"""Volatility surface + its key — the snapshot's `surfaces` shape (L1).

doc 19 §2: a surface is value-over-(strike × expiry), keyed by the underlying/index. This slice
is MINIMAL — a FLAT surface (one lognormal vol at every expiry/strike), the leanest thing a
vol-read (not solved) `BlackModel` needs. A gridded surface + 2D smile interpolation arrive with
their consumer (a calibrated/quoted surface — rule of two). `at(expiry, strike)` is the accessor
upper layers depend on; the `black_vol` capability reaches it, never a bare value.

Provenance:
  quarry: python/pricebook/models/vol_surface.py (surface concept)
  source: redesign/19 §2 (surfaces shape: strike×expiry, keyed by underlying); ScalarKey precedent
  oracle: a caplet reprices to Black-76 off the surface's flat vol
  slice:  black-caplet (C2 slice 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation import Interpolation, RateIndex, Tenor, interpolate


@dataclass(frozen=True)
class SurfaceKey:
    """A key into the snapshot's `surfaces` shape for an OPTIONLET (caplet) vol, carrying the index
    the vol is quoted on (a new asset adds keys, not fields — the ScalarKey precedent)."""

    index: RateIndex


@dataclass(frozen=True)
class SwaptionSurfaceKey:
    """A key for a SWAPTION vol surface — keyed by `index` AND `swap_tenor` (a 2Y5Y vol ≠ a 2Y10Y
    vol), so it never collides with an optionlet `SurfaceKey(index)`. Distinct capability, distinct
    key (doc 19 §2: the key carries the asset dimension)."""

    index: RateIndex
    swap_tenor: Tenor


@dataclass(frozen=True)
class Surface:
    """A Black (lognormal) volatility surface — a TERM STRUCTURE over expiry: `vols[i]` at
    `expiries[i]` (ascending). `at(expiry, strike)` interpolates in **variance** (σ² linear over
    expiry — exact at pillars, and needs no valuation origin), then takes the root; `strike` is
    ignored until the smile axis lands (its consumer). The FLAT case (`Surface.flat(v)`, one vol) is
    the degenerate 1-pillar surface slices 1–2 use — `at` returns the constant without interpolating.
    Grid strike-smile / the swaption cube arrive with their consumers (rule of two)."""

    vols: tuple[float, ...]
    expiries: tuple[date, ...] = ()
    interpolation: Interpolation = Interpolation.LINEAR

    @classmethod
    def flat(cls, vol: float) -> Surface:
        """The degenerate flat surface: one vol at every `(expiry, strike)`."""
        return cls((vol,))

    def at(self, expiry: date, strike: float) -> float:
        """The lognormal vol at `(expiry, strike)`. Flat (1 pillar): the constant. Otherwise σ²
        is interpolated linearly over expiry and rooted (RAISE outside the pillar range — no silent
        extrapolation, consistent with the curve)."""
        if len(self.vols) == 1:
            return self.vols[0]
        xs = tuple(float(d.toordinal()) for d in self.expiries)
        variances = tuple(v * v for v in self.vols)
        return math.sqrt(interpolate(xs, variances, float(expiry.toordinal()), self.interpolation))

    def bumped(self, shift: float) -> Surface:
        """A new frozen surface with every vol shifted by `shift` in parallel (a flat vega bump).
        The base surface is never mutated (invariant 3); risk (L5) reprices off the copy."""
        return Surface(tuple(v + shift for v in self.vols), self.expiries, self.interpolation)
