"""Day-count conventions and year fractions.

Slice 0 needs only ACT/365F. The remaining conventions (ACT/360, 30/360,
ACT/ACT ISDA/ICMA, BUS/252) migrate in S1 against ISDA/ICMA test vectors —
declaring only what is complete keeps this from being a partial abstraction
(CLAUDE.md 6b).

Provenance:
  quarry: python/pricebook/core/day_count.py
  source: ISDA 2006 Definitions, s.4.16 (day count fractions)
  oracle: ACT/365F = actual_days / 365 — exact; extended vectors land in S1
  slice:  S00
"""

from __future__ import annotations

from datetime import date
from enum import Enum


class DayCountConvention(Enum):
    ACT_365_FIXED = "ACT/365F"


def year_fraction(start: date, end: date, convention: DayCountConvention) -> float:
    """Year fraction between two dates. `start` must be on or before `end`."""
    if start > end:
        raise ValueError(f"start ({start}) must be on or before end ({end})")
    if convention is DayCountConvention.ACT_365_FIXED:
        return (end - start).days / 365.0
    raise ValueError(f"Unsupported convention: {convention}")
