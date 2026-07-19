"""Tenor — a period as a value type (L0).

`Tenor(count, unit)` is the one primitive behind index tenors (`"3M"`), schedule steps
and curve-pillar tenors, so `"28D"` is parsed once, not in three modules. This overturns
the earlier "Tenor stays a string" ruling (gate audit S3/S7: a month-int `Frequency`
can't express 28-day, daily or single-period).

Provenance:
  quarry: python/pricebook/core/ (tenor helpers)
  source: standard market tenor notation (D/W/M/Y)
  oracle: parse/str round-trip; months() for M/Y; garbage rejected
  slice:  tenor-frequency (Topic 0 gate rework, S7)
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum


class TenorUnit(Enum):
    DAY = "D"
    WEEK = "W"
    MONTH = "M"
    YEAR = "Y"


_BY_CODE = {u.value: u for u in TenorUnit}


@dataclass(frozen=True)
class Tenor:
    count: int
    unit: TenorUnit

    @classmethod
    def parse(cls, s: str) -> Tenor:
        """Parse `"3M"`, `"28D"`, `"2W"`, `"1Y"`."""
        s = s.strip().upper()
        if len(s) < 2 or s[-1] not in _BY_CODE or not s[:-1].lstrip("-").isdigit():
            raise ValueError(f"cannot parse tenor {s!r} (expected e.g. '3M', '28D')")
        return cls(int(s[:-1]), _BY_CODE[s[-1]])

    def __str__(self) -> str:
        return f"{self.count}{self.unit.value}"

    def __radd__(self, d: date) -> date:
        """`date + Tenor` — the raw shifted date (the most-used curve-building op, S7).
        Day/week tenors are exact; month/year tenors clamp the day to the target month's
        length (31 Jan + 1M → 28/29 Feb). Business-day rolling is a separate `RollRule`
        concern — this stays finance-free date arithmetic."""
        if self.unit is TenorUnit.DAY:
            return d + timedelta(days=self.count)
        if self.unit is TenorUnit.WEEK:
            return d + timedelta(weeks=self.count)
        total = d.month - 1 + self.months()
        year, month = d.year + total // 12, total % 12 + 1
        return date(year, month, min(d.day, monthrange(year, month)[1]))

    def months(self) -> int:
        """Whole months for a month/year tenor; day/week tenors have no fixed month count."""
        if self.unit is TenorUnit.MONTH:
            return self.count
        if self.unit is TenorUnit.YEAR:
            return self.count * 12
        raise ValueError(f"{self} has no fixed month count (it is {self.unit.value}-based)")
