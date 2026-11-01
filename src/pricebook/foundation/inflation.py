"""Inflation index identity (L0).

§3c's corollary: a new asset class adds **one** L0 object — its identity — and nothing else. Spot
levels, curves and vols are market state keyed *by* this identity; the contract terms are L2.

**The publication lag and the interpolation convention live here — and they arrived in tranche 7,
not tranche 2.** The stock's `inflation_indices.py` carries a single `publication_lag_months`: 3 for
most indices and **8 for UK index-linked gilts**. Those are two different quantities conflated under
one name — UK RPI is *published* about a month after its reference month, and the 8 months is the
**indexation lag the gilt's contract specifies**. Two linkers on the same index can specify
different indexation lags, so that one stays contract data (L2, `InflationLinkedBond`); the
publication lag is the index's own and belongs here.

Tranche 2 left both fields out because nothing read them, and tranche 7 brought them back when
seasoned resolution became their first consumer. **That is the pull discipline working forwards** —
the field arrives *with* the code that reads it, not ahead of it — and it is why they are
**required, not defaulted**: "3 months, linear" is right for most markets and wrong for UK gilts,
so a default would silently price the wrong convention. The same reasoning made `JYCorrelations`
required in tranche 5.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/inflation_indices.py (purified to the bare
          identity; the 8-index registry, interpolation enum, deflation-floor notes and the
          conflated lag field are left behind — see above)
  source: BERHAN-2012 (docs/references.md) §1.2, where the index `I(t)` is the FX-analogy spot
  oracle: satisfies value-keying (two identical `InflationIndex` are one snapshot key); the
          lag/interpolation fields are oracled through resolution in tests/L1/test_fixing_series.py
          (FLAT vs LINEAR disagree on a sloped series read mid-month) and round-trip in
          tests/L0/test_representation.py
  slice:  tranche 2 — the inflation spine; the lag + interpolation in tranche 7 (seasoned pricing)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IndexInterpolation(Enum):
    """How a published monthly index is read *within* a month — the index's own convention.

    `FLAT` uses the reference month's print unchanged (UK index-linked gilts). `LINEAR`
    interpolates between the reference month and the next by day-of-month (US TIPS and most
    markets). They are not interchangeable: on a sloped series read mid-month they give different
    numbers, which is what makes the convention worth carrying rather than assuming."""

    FLAT = "flat"
    LINEAR = "linear"


@dataclass(frozen=True)
class InflationIndex:
    """An inflation index: its `name`, its publication lag in months, and how it interpolates.

    Frozen and hashable so it keys the snapshot by VALUE — two identical `InflationIndex` are one
    key, which is what a curve lookup and a fixing lookup both need. Note the corollary of keying
    by value: two indices that differ only in `publication_lag` are *different keys*, which is
    correct — they resolve to different prints and are therefore different market objects."""

    name: str
    publication_lag: int
    interpolation: IndexInterpolation

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("InflationIndex needs a non-empty name")
        if self.publication_lag < 0:
            raise ValueError(
                f"publication_lag is a number of months and cannot be negative, "
                f"got {self.publication_lag}"
            )

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        """The index's own serialised form. It became a nested object in tranche 7: before that a
        product wrote `"index": "HICPxt"`, a bare name, which cannot carry the conventions that
        decide which print `I(t)` is. A payload naming an index without them is not a complete
        description of the trade."""
        return {
            "name": self.name,
            "publication_lag": self.publication_lag,
            "interpolation": self.interpolation.value,
        }

    @classmethod
    def from_dict(cls, d: dict | str) -> InflationIndex:
        """Rebuild from `to_dict`. **A bare string is refused, not defaulted.**

        Pre-tranche-7 payloads wrote the index as a name only. Reading one by assuming "3 months,
        linear" would silently price a UK gilt on the TIPS convention — the exact silent-wrong-number
        this tranche exists to remove — so the absent conventions are an error the caller must
        resolve, not a gap this code fills in."""
        if isinstance(d, str):
            raise ValueError(
                f"index {d!r} is serialised as a bare name (pre-tranche-7 form) and carries no "
                f"publication lag or interpolation convention; refusing to assume them"
            )
        return cls(
            str(d["name"]),
            int(d["publication_lag"]),
            IndexInterpolation(d["interpolation"]),
        )


class CapFloorKind(Enum):
    """Cap or floor — the `omega = +/-1` of BERHAN 1.2.22.

    An enum and **not** a boolean. `InflationCapFloor(..., True)` is unreadable at the call site
    and silently reversible: swap the argument and you have priced the other instrument with no
    error anywhere. The enum cannot be passed backwards."""

    CAP = "cap"
    FLOOR = "floor"

    @property
    def omega(self) -> float:
        """`+1` for a cap, `-1` for a floor — the sign as the thesis writes it."""
        return 1.0 if self is CapFloorKind.CAP else -1.0
