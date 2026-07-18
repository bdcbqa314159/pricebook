"""Deposit — a money-market deposit as pure data (L2).

Place `face` at `start`, receive `face·(1 + rate·τ)` at `maturity`. The builder
expands it into two cashflows (`−N` at start, the redemption at maturity), so the
frozen product is a `CashflowProduct` priced by the existing L4 `DiscountingEngine`
— no bespoke engine. A2 handles the temporal split: a spot deposit's principal-today
is realized cash (excluded from the mark), a forward deposit reprices to zero at par.

Provenance:
  quarry: python/pricebook/fixed_income/deposit.py
  source: standard money-market deposit (simple interest)
  oracle: forward par -> 0; spot par -> principal; closed-form off-par;
          to_dict/from_dict round-trip (CP-3 #2, retires quarry deposit)
  slice:  deposit-spine (CP-2c #3); serialisation-deposit (CP-3 #2)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.time import DayCountConvention, year_fraction

_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Deposit:
    """A money-market deposit: `face` placed at simple `rate`, expanded to its two
    cashflows (principal out + redemption in)."""

    face: Money
    rate: float
    cashflows: tuple[Cashflow, ...]

    def to_dict(self) -> dict[str, Any]:
        """Round-trippable wire form. The cashflows are the load-bearing state (a
        deposit's bullet flows carry no accrual), so they serialise directly.
        `schema_version` makes a future breaking change a loud reject, not a misread.

        ponytail: cashflow/Money encoding is inlined — deposit is the first product to
        serialise. Lift a shared `Money.to_dict`/`Cashflow.to_dict` at the second
        product (FRA/swap, CP-3 #3), per rule of two.
        """
        return {
            "schema_version": _SCHEMA_VERSION,
            "face": {"amount": self.face.amount, "currency": self.face.currency.value},
            "rate": self.rate,
            "cashflows": [
                {"date": cf.date.isoformat(), "amount": cf.amount.amount,
                 "currency": cf.amount.currency.value}
                for cf in self.cashflows
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Deposit":
        """Reconstruct from `to_dict`. No version is legacy v1; a newer version is refused."""
        version = data.get("schema_version", 1)
        if version > _SCHEMA_VERSION:
            raise ValueError(
                f"Deposit schema v{version} newer than reader v{_SCHEMA_VERSION}"
            )
        face = Money(data["face"]["amount"], Currency(data["face"]["currency"]))
        cashflows = tuple(
            Cashflow(date.fromisoformat(c["date"]), Money(c["amount"], Currency(c["currency"])))
            for c in data["cashflows"]
        )
        return cls(face=face, rate=data["rate"], cashflows=cashflows)


def deposit(
    face: Money, rate: float, start: date, maturity: date, day_count: DayCountConvention
) -> Deposit:
    """Build a deposit: `−face` at `start`, `face·(1 + rate·τ)` at `maturity`."""
    tau = year_fraction(start, maturity, day_count)
    principal = Cashflow(date=start, amount=Money(-face.amount, face.currency))
    redemption = Cashflow(date=maturity, amount=Money(face.amount * (1.0 + rate * tau), face.currency))
    return Deposit(face=face, rate=rate, cashflows=(principal, redemption))
