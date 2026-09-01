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
from collections.abc import Mapping
from datetime import date

from pricebook_ng.foundation import Interpolation, RateIndex, Tenor, TimeMeasure, interpolate


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
    `expiries[i]` (ascending). `at(expiry, strike, time_measure)` interpolates **total variance**
    `w = σ²·T` LINEAR in `T` (arb-free — `w` cannot decrease between pillars), then returns
    `σ = √(w/T)`; `T` comes from the pricer's canonical `TimeMeasure`, so the surface axis and the
    priced expiry-`t` are one source (§3d, #4). `strike` is ignored until the smile axis lands. The
    FLAT case (`Surface.flat(v)`, one vol) is the degenerate slices 1–2 use — `at` returns the
    constant. Only LINEAR interpolation is ratified (arb-free); a non-linear scheme is rejected (its
    arb-free form is deferred to a smile/cube consumer). Grid strike-smile arrives with its consumer."""

    vols: tuple[float, ...]
    expiries: tuple[date, ...] = ()
    interpolation: Interpolation = Interpolation.LINEAR

    def __post_init__(self) -> None:
        # a term structure carries one expiry per vol; the flat case is the 1-vol / no-expiry degenerate
        if len(self.vols) != 1 and len(self.expiries) != len(self.vols):
            raise ValueError(
                f"Surface needs one expiry per vol (or a single flat vol); "
                f"got {len(self.vols)} vols, {len(self.expiries)} expiries."
            )
        if self.interpolation is not Interpolation.LINEAR:  # #4: only linear-in-T total variance is arb-free
            raise ValueError(
                f"Surface only supports LINEAR (total-variance) interpolation; got {self.interpolation.name} "
                f"(arb-free non-linear vol interpolation is deferred to a smile/cube consumer)."
            )

    @classmethod
    def flat(cls, vol: float) -> Surface:
        """The degenerate flat surface: one vol at every `(expiry, strike)`."""
        return cls((vol,))

    def at(self, expiry: date, strike: float, time_measure: TimeMeasure) -> float:
        """The lognormal vol at `(expiry, strike)`. Flat (1 pillar): the constant. Otherwise total
        variance `w = σ²·T` is interpolated linearly in `T` (RAISE outside the pillar range) and
        rooted: `σ = √(w/T)`. `time_measure` is the pricer's canonical clock (§3d)."""
        if len(self.vols) == 1:
            return self.vols[0]
        ts = tuple(time_measure.year_fraction(d) for d in self.expiries)
        variances = tuple(v * v * t for v, t in zip(self.vols, ts))  # total variance at each pillar
        t_q = time_measure.year_fraction(expiry)
        w = interpolate(ts, variances, t_q, self.interpolation)
        return math.sqrt(w / t_q)

    def bumped(self, shift: float) -> Surface:
        """A new frozen surface with every vol shifted by `shift` in parallel (a flat vega bump).
        The base surface is never mutated (invariant 3); risk (L5) reprices off the copy."""
        return Surface(tuple(v + shift for v in self.vols), self.expiries, self.interpolation)


@dataclass(frozen=True)
class SabrParams:
    """The SABR parameters for one expiry (Hagan 2002): `alpha` (ATM vol level), `beta` (the CEV
    backbone, a FIXED input this slice), `rho` (spot/vol correlation → skew), `nu` (vol-of-vol →
    convexity). The strike-dependent Black vol is computed by `models.sabr.sabr_vol`."""

    alpha: float
    beta: float
    rho: float
    nu: float


@dataclass(frozen=True)
class SabrSurface:
    """A SABR vol representation — `SabrParams` per expiry pillar, keyed by index (stored in the
    snapshot's `surfaces` map under `SurfaceKey(index)`, the closed-shapes×open-keys design). This is
    an alternative to the flat/term-structure `Surface`: it produces a STRIKE-dependent (smile) vol.
    Params are read at-pillar; cross-expiry interpolation is deferred to its first off-pillar consumer."""

    params: Mapping[date, SabrParams]

    def at_expiry(self, expiry: date) -> SabrParams:
        """The SABR params at `expiry` (at-pillar; `KeyError` off-pillar → the caller fails as a value)."""
        return self.params[expiry]


def flat_surface(surface: Surface | SabrSurface) -> Surface:
    """Narrow a `surfaces` entry to a flat/term `Surface`. A `SabrSurface` here means a flat-vol
    consumer (BlackModel, vega bump) was handed a smile surface — a configuration error raised as a
    value by the caller's failure path. Keeps the union narrowing out of `risk/` (§1 no type-switch)."""
    if not isinstance(surface, Surface):
        raise ValueError(f"expected a flat/term Surface, got {type(surface).__name__}")
    return surface
