"""Zero-coupon inflation swap as pure data (L2).

A ZCIS on a price `index`: at maturity one leg pays realized inflation
`notional*(I(T) - 1)`, the other a fixed compounded rate `notional*((1+K)^T - 1)`.
`receive_inflation=True` receives the inflation leg and pays fixed. Pure data — no
`pv` method (CLAUDE.md 2); pricing is L4.

Provenance:
  quarry: python/pricebook/inflation/ (ZCIS); python/pricebook/fixed_income/inflation.py
          (ZCInflationSwap — this supersedes only the ZCIS; the module is held a partial cross:
          YoY dead, InflationLinkedBond deferred → its un-crossed consumers)
  source: zero-coupon inflation swap; Fisher relation
  oracle: par ZCIS reprices to zero (inflation-zcis slice); to_dict/from_dict round-trip
  slice:  inflation-zcis; serialisation-zcis (CP-3 tail, build-early per §4.5)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook_ng.foundation.money import Money

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ZeroCouponInflationSwap:
    """A ZCIS on `index`: inflation leg vs fixed compounded rate `fixed_rate` (K) on
    `face` (notional + currency)."""

    index: str
    face: Money
    fixed_rate: float
    maturity: date
    receive_inflation: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable wire form (+ `schema_version`); `face` via the shared `Money` encoder."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "index": self.index,
            "face": self.face.to_dict(),
            "fixed_rate": self.fixed_rate,
            "maturity": self.maturity.isoformat(),
            "receive_inflation": self.receive_inflation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ZeroCouponInflationSwap":
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"ZeroCouponInflationSwap schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        return cls(
            index=data["index"],
            face=Money.from_dict(data["face"]),
            fixed_rate=data["fixed_rate"],
            maturity=date.fromisoformat(data["maturity"]),
            receive_inflation=data["receive_inflation"],
        )
