"""OvernightIndexSwap — fixed vs compounded-overnight float, as pure data (L2).

An OIS pays a fixed rate against the compounded overnight rate (SOFR/SONIA/ESTR).
Single-curve, the compounded overnight rate over a period equals the curve's forward,
so structurally it is the vanilla IRS (it reuses the swap's `FixedLeg`/`FloatLeg`). The
distinct type carries the RFR semantics; the multi-curve OIS/IBOR basis and daily-fixing
compounding are a later slice.

Provenance:
  quarry: python/pricebook/fixed_income/ois.py
  source: standard single-curve OIS (RFR compounded in arrears)
  oracle: par OIS -> 0; OIS == vanilla IRS (single-curve);
          to_dict/from_dict round-trip (CP-3 #4, retires quarry ois)
  slice:  ois-spine (CP-2c #4); serialisation-ois (CP-3 #4)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.schedule import generate_schedule
from pricebook_ng.foundation.time import DayCountConvention
from pricebook_ng.products.leg import fixed_coupon_cashflows
from pricebook_ng.products.swap import FixedLeg, FloatLeg, SwapTerms

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OvernightIndexSwap:
    """A fixed-vs-compounded-overnight swap: same leg shape as a vanilla IRS."""

    fixed_leg: FixedLeg
    float_leg: FloatLeg
    pay_fixed: bool

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable wire form (+ `schema_version`). The `FixedLeg`/`FloatLeg`
        encoding is inlined — OIS is their only serialising consumer so far; lift a
        shared leg encoder when the vanilla swap serialises (rule of two)."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "fixed_leg": [cf.to_dict() for cf in self.fixed_leg.cashflows],
            "float_leg": {
                "face": self.float_leg.face.to_dict(),
                "schedule": [d.isoformat() for d in self.float_leg.schedule],
                "day_count": self.float_leg.day_count.value,
            },
            "pay_fixed": self.pay_fixed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OvernightIndexSwap":
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"OvernightIndexSwap schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        fixed_leg = FixedLeg(tuple(Cashflow.from_dict(c) for c in data["fixed_leg"]))
        f = data["float_leg"]
        float_leg = FloatLeg(
            face=Money.from_dict(f["face"]),
            schedule=tuple(date.fromisoformat(d) for d in f["schedule"]),
            day_count=DayCountConvention(f["day_count"]),
        )
        return cls(fixed_leg=fixed_leg, float_leg=float_leg, pay_fixed=data["pay_fixed"])


def overnight_index_swap(
    face: Money, fixed_rate: float, start: date, maturity: date, terms: SwapTerms
) -> OvernightIndexSwap:
    """Build an OIS: fixed coupons + a structural compounded-overnight float leg."""
    fixed_leg = FixedLeg(
        fixed_coupon_cashflows(face, fixed_rate, start, maturity, terms.fixed_schedule)
    )
    float_dates = generate_schedule(
        start, maturity, terms.float_schedule.frequency, terms.float_schedule.roll
    )
    float_leg = FloatLeg(
        face=face, schedule=tuple(float_dates), day_count=terms.float_schedule.day_count
    )
    return OvernightIndexSwap(fixed_leg=fixed_leg, float_leg=float_leg, pay_fixed=terms.pay_fixed)
