"""Discount curve + the CurveHandle capability (L1).

A `DiscountCurve` holds discount factors at pillar times and interpolates them
LOG-LINEAR — a constant continuously-compounded forward between pillars, the
market-standard minimal scheme, and exact for a flat curve (`df(t) = exp(-r·t)`).
`CurveHandle` is the capability upper layers depend on — `df(date)` — never the
concrete curve (redesign/19 §3). The date→t map is the curve's `TimeMeasure` (one
map, ruling A1). doc 19's typed `CurveSet` (discount·projection·survival·…) arrives
with its second curve family at multicurve (rule of two).

Provenance:
  quarry: python/pricebook/core/discount_curve.py
  source: redesign/19 (CurveHandle · CurveSet); log-linear discount-factor interpolation
  oracle: flat-curve df(t) = exp(-r·t) to 1e-12; par swap reprices to zero NPV
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from functools import cached_property
from typing import Protocol

from pricebook_ng.foundation import (
    Interpolation,
    MonotoneConvex,
    TimeMeasure,
    interpolate,
    monotone_convex,
)


class CurveHandle(Protocol):
    """The discounting capability — a discount factor to a date, and a `bumped` copy for risk.
    Depend on this, not the concrete curve (redesign/19 §3); risk (L5) bumps through it."""

    def df(self, d: date) -> float: ...

    def bumped(self, shift: float) -> CurveHandle: ...


@dataclass(frozen=True)
class DiscountCurve:
    """Discount factors `dfs` at pillar `times` (ascending, `times[0] == 0.0`,
    `dfs[0] == 1.0`), interpolated `LOG_LINEAR`. Reached only through `df(date)`;
    a date past the last pillar raises (the default RAISE extrapolation), which the
    engine turns into a `PricingFailure`."""

    time_measure: TimeMeasure
    times: tuple[float, ...]
    dfs: tuple[float, ...]
    interpolation: Interpolation = Interpolation.LOG_LINEAR

    def df(self, d: date) -> float:
        t = self.time_measure.year_fraction(d)
        if self.interpolation is Interpolation.HAGAN_WEST:
            return math.exp(-self._forward_reconstruction.integral(t))
        return interpolate(self.times, self.dfs, t, self.interpolation)

    def bumped(self, shift: float) -> DiscountCurve:
        """A new frozen curve with the zero rate shifted by `shift` in parallel:
        `df'(tᵢ) = dfᵢ·exp(−shift·tᵢ)` (df at t=0 is unchanged). The base curve is never mutated
        (invariant 3); risk (L5) reprices off the copy."""
        dfs = tuple(df * math.exp(-shift * t) for t, df in zip(self.times, self.dfs))
        return DiscountCurve(self.time_measure, self.times, dfs, self.interpolation)

    @cached_property
    def _forward_reconstruction(self) -> MonotoneConvex:
        """Hagan–West: the average instantaneous forward over each pillar interval,
        `(ln df_{i-1} − ln df_i)/Δt`, fed to the L0 monotone-convex primitive; then
        `df(t) = exp(−∫₀ᵗ f)`. Reproduces the pillar DFs exactly. It is NON-LOCAL — its home
        is the simultaneous solve (sequential-HW is deferred to the paper's terminal-interval
        convention). Cached per curve (the reconstruction is O(n); rebuilding per `df()` would
        make a full-curve reval O(n²)) — the frozen curve is immutable, so the cache is honest
        (`cached == uncached`, finding #8)."""
        averages = tuple(
            (math.log(self.dfs[i - 1]) - math.log(self.dfs[i])) / (self.times[i] - self.times[i - 1])
            for i in range(1, len(self.times))
        )
        return monotone_convex(self.times, averages)

    @classmethod
    def flat(cls, time_measure: TimeMeasure, rate: float, until: date) -> DiscountCurve:
        """A flat continuously-compounded curve: `df(t) = exp(-rate·t)` exactly
        (log-linear between the anchor and `until`)."""
        t = time_measure.year_fraction(until)
        return cls(time_measure, (0.0, t), (1.0, math.exp(-rate * t)))
