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
  slice:  S00; A1 (FixingHistory first-class); survival-in-snapshot (§5.1 credit curve)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

from pricebook_ng.foundation.money import Currency
from pricebook_ng.foundation.time import DayCountConvention, year_fraction


@runtime_checkable
class CurveHandle(Protocol):
    """The capability higher layers depend on: a discount factor per date."""

    def df(self, d: date) -> float: ...


@runtime_checkable
class SurvivalHandle(Protocol):
    """The credit-curve capability: a survival probability per date. Kept as a
    protocol (not the concrete `SurvivalCurve`) so the snapshot avoids importing
    the credit module — same pattern as `CurveHandle`."""

    def survival(self, d: date) -> float: ...


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
    discount_curve: CurveHandle          # the home / valuation-currency curve
    fixings: FixingHistory = field(default_factory=FixingHistory)
    survival_curve: SurvivalHandle | None = None  # the credit curve (market data, §5.1)
    # FX market data (§5.1): foreign-currency discount curves + spots (home units per
    # foreign unit), keyed by foreign currency. A full currency->curve map is the
    # eventual clean shape; this keeps the home curve as `discount_curve`.
    fx_curves: dict[Currency, CurveHandle] = field(default_factory=dict)
    fx_spots: dict[Currency, float] = field(default_factory=dict)
