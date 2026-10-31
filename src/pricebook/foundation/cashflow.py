"""The instrument atom: a cashflow (L0).

Provenance:
  stock:  _stock/src/pricebook/foundation/cashflow.py (purified: the flow itself; `Leg` and the
          `accrual` field stay in the stock — accrual belongs with the first coupon-bearing
          product, which can state what it means)
  source: CLAUDE.md-era §3 vocabulary (Cashflow as the instrument atom)
  oracle: round-trips through its serialised form; the tranche-1 closed form prices one
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money

#: Bump when the serialised shape changes incompatibly. Present from the first product because a
#: format with no version cannot change without silently breaking a reader — and the scope line
#: makes this representation the library's durable public contract.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Cashflow:
    """A single payment of `amount` on `date`. Pure data — it does not discount itself."""

    date: date
    amount: Money

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "date": self.date.isoformat(),
            "amount": self.amount.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cashflow:
        return cls(date.fromisoformat(d["date"]), Money.from_dict(d["amount"]))
