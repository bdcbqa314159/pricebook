"""Instrument atoms — Accrual, Cashflow, Leg (L0).

Finance-free pure data: an `Accrual` is a day-count period, a `Cashflow` a dated
payment, a `Leg` an ordered run of cashflows on one convention. `Accrual.year_fraction`
is the ergonomic entry point over the Slice-2 `year_fraction` primitive — it bundles
start + end + day_count, so callers pass one object, not three loose args.

Provenance:
  quarry: python/pricebook/fixed_income/fixed_leg.py (Cashflow atom)
  source: promoted to L0 per redesign/16 §2.3 (instrument atoms)
  oracle: Accrual.year_fraction matches the S2 primitive; leg construction
  slice:  money-quantity (Topic 0 S4)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.day_count import Accrual, DayCountConvention
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.settlement import Delivery

# A leg's flow is cash OR physical: a `Cashflow(Money)` or a `Delivery(Quantity)`
# (gate audit S2). Pay/receive is the SIGN of the amount — no direction field (S13).
# `Accrual` lives in `day_count.py` (A1 — it is an applied day count, not a cashflow concept).

# The serialisation-schema version for a `Leg` (the serialised root, S8). Bump on a
# breaking layout change; `from_dict` refuses versions it does not understand.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Cashflow:
    """A single payment of `amount` on `date`; a coupon additionally carries its
    `accrual` period (for accrued interest at valuation), a bullet leaves it `None`."""

    date: date
    amount: Money
    accrual: Accrual | None = None

    def to_dict(self) -> dict:
        return {
            "kind": "cash",
            "date": self.date.isoformat(),
            "amount": self.amount.to_dict(),
            "accrual": self.accrual.to_dict() if self.accrual is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cashflow:
        accrual = d["accrual"]
        return cls(
            date.fromisoformat(d["date"]),
            Money.from_dict(d["amount"]),
            Accrual.from_dict(accrual) if accrual is not None else None,
        )


# The flow union's `kind` discriminator → its decoder (S2: cash or physical).
_FLOW_FROM_DICT = {"cash": Cashflow.from_dict, "delivery": Delivery.from_dict}


@dataclass(frozen=True)
class Leg:
    """An ordered run of flows (cash `Cashflow` or physical `Delivery`) sharing one
    day-count convention. Direction is carried by the sign of each amount, not a flag."""

    flows: tuple[Cashflow | Delivery, ...]
    day_count: DayCountConvention

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "day_count": self.day_count.value,
            "flows": [f.to_dict() for f in self.flows],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Leg:
        version = d["schema_version"]
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unknown Leg schema_version {version} (this build reads {SCHEMA_VERSION})"
            )
        flows = tuple(_FLOW_FROM_DICT[f["kind"]](f) for f in d["flows"])
        return cls(flows, DayCountConvention(d["day_count"]))
