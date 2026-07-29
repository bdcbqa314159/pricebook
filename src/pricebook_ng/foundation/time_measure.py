"""TimeMeasure — the sanctioned date→year-fraction map (L0).

A curve, a model and an engine must all turn a `date` into a year fraction `t` the
SAME way, or two of them measure one interval differently. `TimeMeasure` is that one
map: an anchor date plus a day count. It is the only `date → t` in the tree (ruling
A1, redesign/19 §7) — nothing re-derives it.

Provenance:
  quarry: python/pricebook/core/day_count.py (t-from-date helpers)
  source: redesign/19 §7 (TimeMeasure anchor, ruling A1); ISDA 2006 §4.16 day counts
  oracle: year_fraction(anchor) == 0; agrees with the L0 year_fraction primitive
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.day_count import DayCountConvention, year_fraction


@dataclass(frozen=True)
class TimeMeasure:
    """The one `date → t` map: `year_fraction(d)` is the fraction from `anchor` to
    `d` under `day_count`. A date before the anchor raises — historical dates are
    excluded upstream (the shell), never mapped here."""

    anchor: date
    day_count: DayCountConvention

    def year_fraction(self, d: date) -> float:
        return year_fraction(self.anchor, d, self.day_count)
