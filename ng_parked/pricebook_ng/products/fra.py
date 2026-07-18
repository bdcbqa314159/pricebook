"""ForwardRateAgreement — a single-period FRA as pure data (L2).

Pay `fixed_rate`, receive the simply-compounded forward over `[accrual_start,
accrual_end]`, settled at the end. Pure data — no `pv` (pricing lives in the L4
`FRAEngine`, which resolves the forward from the curve; the float amount is not
knowable without one, so — like the swap's float leg — it stays structural).

Provenance:
  quarry: python/pricebook/fixed_income/fra.py
  source: standard single-curve FRA
  oracle: par FRA (K = forward) reprices to zero; closed-form off-par PV;
          to_dict/from_dict round-trip (CP-3 #3, retires quarry fra)
  slice:  fra-spine (CP-2b #3); serialisation-fra (CP-3 #3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pricebook_ng.foundation.cashflow import Accrual
from pricebook_ng.foundation.money import Money

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ForwardRateAgreement:
    """A vanilla FRA: `pay_fixed` pays `fixed_rate` and receives the forward over the
    `accrual` period on `face` (notional + currency)."""

    face: Money
    fixed_rate: float
    accrual: Accrual
    pay_fixed: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable wire form (+ `schema_version`: absent = legacy v1, newer =
        loud reject). `Money` and `Accrual` use their shared encoders."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "face": self.face.to_dict(),
            "fixed_rate": self.fixed_rate,
            "accrual": self.accrual.to_dict(),
            "pay_fixed": self.pay_fixed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ForwardRateAgreement":
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"ForwardRateAgreement schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        return cls(
            face=Money.from_dict(data["face"]),
            fixed_rate=data["fixed_rate"],
            accrual=Accrual.from_dict(data["accrual"]),
            pay_fixed=data["pay_fixed"],
        )
