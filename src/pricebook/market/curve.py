"""The discount curve and the capability the engine depends on (L1).

**`CurveHandle` is a protocol, and the engine depends on it rather than on `DiscountCurve`.**
A pricer needs one thing — `df(date)` — so that is what it asks for. Depending on the concrete
curve would mean every later curve shape (bootstrapped, monotone-convex, a shifted copy for a
bump) either subclasses it or forces a type-switch in the engine.

**Flat and log-linear only.** The stock's curve also carries Hagan–West monotone-convex
reconstruction; it is non-local and its home is a simultaneous curve solve, so it arrives with the
tranche that builds one.

Provenance:
  stock:  _stock/src/pricebook/market/curve.py (purified: flat + log-linear; the monotone-convex
          reconstruction stays in the stock until a curve-build consumer needs it)
  source: standard continuously-compounded discounting, `df(t) = exp(−r·t)`
  oracle: `DiscountCurve.flat` reproduces `exp(−r·t)` exactly at every horizon, which the
          tranche-1 closed form asserts independently
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.time_measure import TimeMeasure


@runtime_checkable
class CurveHandle(Protocol):
    """What a pricer needs of a discount curve: a discount factor to a date, and the time measure
    it is anchored on (a model that integrates over the curve needs the curve's own clock, and
    reaching for it with `getattr` is the inspect-the-object pattern a protocol exists to avoid)."""

    def df(self, d: date) -> float: ...

    @property
    def time_measure(self) -> TimeMeasure: ...


@dataclass(frozen=True)
class DiscountCurve:
    """Discount factors from continuously-compounded zero rates at pillar times.

    `times` are year fractions on `time_measure` (ascending, starting at 0.0) and `zero_rates`
    the continuously-compounded rates at them; between pillars the **zero rate** is interpolated
    linearly, which is log-linear in the discount factor. Frozen: a bumped curve is a new object,
    never a mutation, so a risk calculation cannot disturb the base market."""

    time_measure: TimeMeasure
    times: tuple[float, ...]
    zero_rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.zero_rates):
            raise ValueError("times and zero_rates must have equal length")
        if not self.times:
            raise ValueError("a curve needs at least one pillar")
        if self.times[0] != 0.0:
            raise ValueError(f"the first pillar must be t=0.0, got {self.times[0]}")
        if any(b <= a for a, b in zip(self.times, self.times[1:])):
            raise ValueError("times must be strictly ascending")

    @classmethod
    def flat(cls, anchor: date, rate: float, day_count: DayCountConvention) -> DiscountCurve:
        """A flat curve: `df(t) = exp(−rate·t)` exactly, at every horizon and with no
        interpolation error, because one pillar extrapolates flat."""
        return cls(TimeMeasure(anchor, day_count), (0.0,), (rate,))

    def zero_rate(self, t: float) -> float:
        """The continuously-compounded zero rate at `t`, linearly interpolated between pillars and
        held flat beyond the ends. Flat extrapolation rather than a raise, because a one-pillar
        flat curve is the degenerate case this tranche is built on."""
        if t <= self.times[0]:
            return self.zero_rates[0]
        if t >= self.times[-1]:
            return self.zero_rates[-1]
        # `bisect` rather than a scan, so the bracketing is TOTAL: the two guards above exclude
        # the ends, so `i` always indexes a real interval and there is no unreachable branch to
        # excuse with a pragma. (The first draft had one, and the debt gate caught it — the right
        # answer to dead code is to remove the deadness, not to ledger the suppression.)
        i = bisect.bisect_right(self.times, t) - 1
        t0, t1 = self.times[i], self.times[i + 1]
        r0, r1 = self.zero_rates[i], self.zero_rates[i + 1]
        return r0 + (r1 - r0) * (t - t0) / (t1 - t0)

    def df(self, d: date) -> float:
        t = self.time_measure.year_fraction(d)
        return math.exp(-self.zero_rate(t) * t)

    def bumped(self, shift: float) -> DiscountCurve:
        """A new curve with every zero rate shifted by `shift`. The base is never mutated."""
        return DiscountCurve(
            self.time_measure, self.times, tuple(r + shift for r in self.zero_rates)
        )
