"""Day count and the year fraction (L0).

**ACT/365F only, deliberately.** The stock carries the full ISDA set plus ICMA coupon-period
machinery in 332 lines; a fixed cashflow on a flat curve needs exactly one convention, and the
others arrive when a product's contract specifies them. Adding them now would be building for
products that have not asked — and each one needs its own published test vectors to land honestly,
which is a tranche of its own, not a footnote to this one.

Provenance:
  stock:  _stock/src/pricebook/foundation/day_count.py (purified: ACT/365F only)
  source: ISDA 2006 Definitions §4.16 (Act/365 (Fixed))
  oracle: the tranche-1 closed form states `t = days/365` in the test, independently of this
          module, so a wrong year fraction moves the PV
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from datetime import date
from enum import Enum


class DayCountConvention(Enum):
    """How a date interval becomes a year fraction.

    An `Enum` here and NOT in `Currency`, for a reason worth stating: day-count conventions are a
    **closed, published set** (ISDA defines them; a new one is a standards event, not a market
    event), whereas currencies are open-ended. The shape follows the domain."""

    ACT_365_FIXED = "ACT/365F"


def year_fraction(start: date, end: date, convention: DayCountConvention) -> float:
    """The year fraction between `start` and `end` under `convention`.

    Signed: `end < start` gives a negative fraction rather than raising, because callers need to
    *detect* a past date (invariant: a flow on or before the valuation date is historical) and a
    raise would force them to pre-check the thing this function already knows."""
    if convention is DayCountConvention.ACT_365_FIXED:
        return (end - start).days / 365.0
    raise ValueError(f"unsupported day count: {convention}")  # unreachable while the enum has one
