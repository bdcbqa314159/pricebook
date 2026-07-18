"""FixedCashflow — the Slice 0 product: one fixed cashflow.

L2: products are pure data. This describes a single payment and holds NO
`pv` method — pricing lives in the L4 engine (CLAUDE.md 2).

A zero-coupon bond is exactly this — face at maturity, priced `Face·DF(T)` by the
engine; its money-market yield analytics are shed dead (see the CP-3 #5 retire note).

Provenance:
  quarry: python/pricebook/fixed_income/ (zero-coupon / single cashflow);
          python/pricebook/fixed_income/zero_coupon_bond.py (CP-3 #5 retire)
  source: redesign/04_slice_plan.md (Slice 0 product)
  oracle: Slice 0 closed form (priced by DiscountingEngine); to_dict/from_dict round-trip
  slice:  S00; A3 (instrument -> product rename); serialisation-zcb (CP-3 #5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pricebook_ng.foundation.cashflow import Cashflow

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FixedCashflow:
    """A product that pays a single fixed `Cashflow`."""

    cashflow: Cashflow

    @property
    def cashflows(self) -> tuple[Cashflow, ...]:
        """The engine's `CashflowProduct` view: a one-cashflow leg."""
        return (self.cashflow,)

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable wire form (+ `schema_version`); the `Cashflow` uses its
        shared encoder."""
        return {"schema_version": _SCHEMA_VERSION, "cashflow": self.cashflow.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixedCashflow":
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"FixedCashflow schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        return cls(Cashflow.from_dict(data["cashflow"]))
