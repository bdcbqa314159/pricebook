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
from typing import Protocol

from pricebook_ng.foundation import Interpolation, TimeMeasure, interpolate


class CurveHandle(Protocol):
    """The discounting capability — a discount factor to a date. Depend on this, not
    the concrete curve (redesign/19 §3)."""

    def df(self, d: date) -> float: ...


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
        return interpolate(
            self.times, self.dfs, self.time_measure.year_fraction(d), self.interpolation
        )

    @classmethod
    def flat(cls, time_measure: TimeMeasure, rate: float, until: date) -> DiscountCurve:
        """A flat continuously-compounded curve: `df(t) = exp(-rate·t)` exactly
        (log-linear between the anchor and `until`)."""
        t = time_measure.year_fraction(until)
        return cls(time_measure, (0.0, t), (1.0, math.exp(-rate * t)))
