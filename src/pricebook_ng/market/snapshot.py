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
  slice:  S00
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from pricebook_ng.foundation.time import DayCountConvention, year_fraction


@runtime_checkable
class CurveHandle(Protocol):
    """The capability higher layers depend on: a discount factor per date."""

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
class MarketSnapshot:
    """Immutable market state as of `valuation_date` ("today" lives here)."""

    valuation_date: date
    discount_curve: CurveHandle
