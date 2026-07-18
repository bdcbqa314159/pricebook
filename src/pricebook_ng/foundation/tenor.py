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

from dataclasses import dataclass
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

    def months(self) -> int:
        """Whole months for a month/year tenor; day/week tenors have no fixed month count."""
        if self.unit is TenorUnit.MONTH:
            return self.count
        if self.unit is TenorUnit.YEAR:
            return self.count * 12
        raise ValueError(f"{self} has no fixed month count (it is {self.unit.value}-based)")
