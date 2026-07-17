"""Deposit — a money-market deposit as pure data (L2).

Place `face` at `start`, receive `face·(1 + rate·τ)` at `maturity`. The builder
expands it into two cashflows (`−N` at start, the redemption at maturity), so the
frozen product is a `CashflowProduct` priced by the existing L4 `DiscountingEngine`
— no bespoke engine. A2 handles the temporal split: a spot deposit's principal-today
is realized cash (excluded from the mark), a forward deposit reprices to zero at par.

Provenance:
  quarry: python/pricebook/fixed_income/deposit.py
  source: standard money-market deposit (simple interest)
  oracle: forward par -> 0; spot par -> principal; closed-form off-par
  slice:  deposit-spine (CP-2c #3 — fixed_income spine to parity)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.time import DayCountConvention, year_fraction


@dataclass(frozen=True)
class Deposit:
    """A money-market deposit: `face` placed at simple `rate`, expanded to its two
    cashflows (principal out + redemption in)."""

    face: Money
    rate: float
    cashflows: tuple[Cashflow, ...]


def deposit(
    face: Money, rate: float, start: date, maturity: date, day_count: DayCountConvention
) -> Deposit:
    """Build a deposit: `−face` at `start`, `face·(1 + rate·τ)` at `maturity`."""
    tau = year_fraction(start, maturity, day_count)
    principal = Cashflow(date=start, amount=Money(-face.amount, face.currency))
    redemption = Cashflow(date=maturity, amount=Money(face.amount * (1.0 + rate * tau), face.currency))
    return Deposit(face=face, rate=rate, cashflows=(principal, redemption))
