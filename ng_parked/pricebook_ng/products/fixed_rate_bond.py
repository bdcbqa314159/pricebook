"""FixedRateBond — a fixed-coupon bond as pure data (L2).

The bond describes a leg of coupon cashflows plus the redemption; it holds NO
`pv` method (CLAUDE.md 2 — pricing lives in the L4 engine). The builder
`fixed_rate_bond(...)` expands schedule + day-count into explicit `Cashflow`s at
construction, so the frozen instrument stays pure data.

Coupons are dropped into a plain `tuple[Cashflow, ...]`; a shared `Leg` type is
not introduced until a second consumer needs it (rule of two, CLAUDE.md 6b).

Provenance:
  quarry: python/pricebook/fixed_income/ (fixed-rate bond / fixed leg);
          python/pricebook/fixed_income/bond.py (CP-3 tail retire)
  source: standard fixed-coupon bond; ISDA 2006 accrual
  oracle: closed-form discounted-cashflow PV < 1e-12 (S04); to_dict/from_dict round-trip
  slice:  S04; S05 (Money face + ScheduleTerms — 5-arg ceiling);
          S06 (coupons via shared fixed_coupon_cashflows); serialisation-bond (CP-3 tail)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import ScheduleTerms
from pricebook_ng.products.leg import fixed_coupon_cashflows

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class FixedRateBond:
    """A fixed-rate bond: its coupon+redemption cashflows plus identifying data."""

    notional: float
    coupon_rate: float
    currency: Currency
    cashflows: tuple[Cashflow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable wire form (+ `schema_version`); cashflows via the shared
        `Cashflow` encoder."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "notional": self.notional,
            "coupon_rate": self.coupon_rate,
            "currency": self.currency.value,
            "cashflows": [cf.to_dict() for cf in self.cashflows],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FixedRateBond":
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"FixedRateBond schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        return cls(
            notional=data["notional"],
            coupon_rate=data["coupon_rate"],
            currency=Currency(data["currency"]),
            cashflows=tuple(Cashflow.from_dict(c) for c in data["cashflows"]),
        )


def fixed_rate_bond(
    face: Money,
    coupon_rate: float,
    start: date,
    maturity: date,
    terms: ScheduleTerms,
) -> FixedRateBond:
    """Build a fixed-rate bond: fixed coupons plus the notional redemption.

    `face` is the notional as `Money(amount, currency)`. The notional redeems at
    maturity as a separate cashflow on that date.
    """
    coupons = fixed_coupon_cashflows(face, coupon_rate, start, maturity, terms)
    redemption = Cashflow(date=maturity, amount=face)
    return FixedRateBond(
        notional=face.amount,
        coupon_rate=coupon_rate,
        currency=face.currency,
        cashflows=(*coupons, redemption),
    )
