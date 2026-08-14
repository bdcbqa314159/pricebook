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

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation import RateIndex


@dataclass(frozen=True)
class SurfaceKey:
    """A key into the snapshot's `surfaces` shape, carrying the index the vol is quoted on (a
    new asset adds keys, not fields — the ScalarKey precedent)."""

    index: RateIndex


@dataclass(frozen=True)
class Surface:
    """A Black (lognormal) volatility surface. MINIMAL: FLAT — one `flat_vol` at every
    `(expiry, strike)`. Grid storage + 2D smile interpolation arrive with their consumer."""

    flat_vol: float

    def at(self, expiry: date, strike: float) -> float:
        """The lognormal vol at `(expiry, strike)`. Flat surface: the constant, at every point."""
        return self.flat_vol
