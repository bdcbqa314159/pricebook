"""Immutable market snapshot and the flat discount curve behind a CurveHandle.

L1: a snapshot is a frozen picture of the market. Higher layers reach a curve
through the `CurveHandle` capability (`df(date)`), not a concrete class
(vocabulary ratification 5) — and curves are never mutated in place; a bump
produces a new curve.

Slice 0 ships only the flat, continuously-compounded curve `df(t)=exp(-r t)`.
The bootstrapped curve lands at S2 behind the same handle.

Provenance:
  quarry: python/pricebook/core/discount_curve.py (re-homed core -> L1, minimal)
  source: Hull, Options Futures & Other Derivatives — continuous discounting
  oracle: df(t) = exp(-r t) closed form (drives the Slice 0 PV oracle)
  slice:  S00; A1 (FixingHistory); survival-in-snapshot (§5.1); A5 (keyed registry)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.keys import MarketKey


@runtime_checkable
class CurveHandle(Protocol):
    """The capability higher layers depend on: a factor per date. `df` is the
    discount factor for a discount curve, the survival probability for a hazard
    curve — the same shape, so both live in the snapshot's `curves` map."""

    def df(self, d: date) -> float: ...


@dataclass(frozen=True)
class FlatDiscountCurve:
    """Flat continuously-compounded curve: df(d) = exp(-rate * t)."""

    rate: float
    anchor: date
    day_count: DayCountConvention

    def df(self, d: date) -> float:
        t = year_fraction(self.anchor, d, self.day_count)
        return math.exp(-self.rate * t)


@dataclass(frozen=True)
class FixingHistory:
    """Realized index fixings by observation date. First-class in the snapshot so
    the core can resolve current-period amounts (the economy = curves + fixings).

    A1 introduces the type + empty default; the seasoned-period lookup that
    consumes it lands with the temporal slice (A2)."""

    fixings: dict[date, float] = field(default_factory=dict)

    def get(self, observation: date) -> float | None:
        return self.fixings.get(observation)


@dataclass(frozen=True)
class MarketSnapshot:
    """Immutable market state as of `valuation_date` — the economy (curves +
    fixings). "Today" lives here; a model is calibrated to a snapshot and carries
    it (Amendment A1)."""

    valuation_date: date
    discount_curve: CurveHandle          # HOME NUMERAIRE — stays special (A5.1)
    fixings: FixingHistory = field(default_factory=FixingHistory)
    # Keyed registry (A5): all other market data, keyed by MarketKey(asset, id).
    # curves = survival / dividend / foreign-discount / projection; spots + vols per key.
    curves: dict[MarketKey, CurveHandle] = field(default_factory=dict)
    spots: dict[MarketKey, float] = field(default_factory=dict)
    vols: dict[MarketKey, float] = field(default_factory=dict)
