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
