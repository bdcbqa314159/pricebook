"""The time measure — the ONE `date → t` map (L0).

Every `t` in the library comes from here. That is a convention with a cost of being wrong: the
stock's conventions register records that a vol clock read by ten modules was never asserted to be
*one* clock, and the predecessor's 5.14% error was the same shape — a convention several modules
had to share, with nothing comparing them. One type, constructed once per market, is the
structural answer: there is nothing to disagree about.

Provenance:
  stock:  _stock/src/pricebook/foundation/time_measure.py (purified as-is; already minimal)
  source: CLAUDE.md-era ruling A1 (one date→t map); `_stock/docs/conventions_register.md` C2
  oracle: year_fraction(anchor) == 0; agrees with the L0 primitive (test_time_measure)
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.day_count import DayCountConvention, year_fraction


@dataclass(frozen=True)
class TimeMeasure:
    """`anchor` plus a `day_count`: the map a curve's axis and a pricer's `t` both read."""

    anchor: date
    day_count: DayCountConvention

    def year_fraction(self, d: date) -> float:
        return year_fraction(self.anchor, d, self.day_count)


def is_historical(pay: date, valuation_date: date) -> bool:
    """`True` when a flow paying on `pay` is historical — invariant 6's boundary, as ONE atom.

    *A flow on or before `valuation_date` is historical*: excluded from the mark, PV 0 (the L6
    shell's business), never discounted with a non-positive `t`. The comparison is **strict on the
    live side** — a payment landing exactly today is historical — and that boundary case is not
    academic: tranche 24 measured a `>` -> `>=` slip as a **41% mispricing** in the inflation
    cap/floor (5,929 -> 8,364), while the same slip was a proved *equivalent* mutant in two other
    engines, because an expired Hull-White caplet is worth exactly zero.

    That asymmetry is precisely why this is an atom rather than a convention. Five engines inlining
    one comparison makes a divergence between them **expressible**, which is all register row C10
    could ever claim (`per-module`, measured). Routing every consumer through here makes it
    inexpressible — C10's stated trigger, and the C9 standing.

    Finance-free in code (§2): a date comparison. The finance is in the docstring, where L0 allows
    it.
    """
    return pay <= valuation_date
