"""Settlement — how a flow settles (L0).

Finance-free: it describes the *shape* of settlement, not its value. `Cashflow(date,
Money)` (S4) covers cash in the contract currency; this completes the vocabulary — a
physical `Delivery(date, Quantity)` alongside it, a settlement currency that may differ
from the contract currency (quanto / NDF), and settlement-date arithmetic (trade + lag
business days under a calendar).

`AUCTION` is a **marker** only: the CDS credit-event / recovery / auction-price
mechanics belong to the credit topic (higher layer), never L0. Collateral/CSA
discounting is a *numeraire* — an L3 model choice — and is deliberately absent here.

Provenance:
  quarry: python/pricebook/core/settlement.py (zero-fan-in orphan — content mined, structure ignored)
  source: ISDA settlement definitions; ACI Model Code (FX T+2)
  oracle: settlement date = trade + lag under a calendar; cash vs physical flow types;
          settlement currency ≠ contract currency
  slice:  settlement (Topic 0 S4b)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum

from pricebook_ng.foundation.calendar import Calendar
from pricebook_ng.foundation.money import Currency, Quantity


class SettlementType(Enum):
    CASH = "cash"
    PHYSICAL = "physical"
    AUCTION = "auction"      # CDS credit event (marker only — mechanics live in the credit topic)


@dataclass(frozen=True)
class Delivery:
    """A physical settlement flow: a `Quantity` delivered on a date — the `Cashflow`
    of the physical world (barrels, MWh, a security)."""

    date: date
    quantity: Quantity


@dataclass(frozen=True)
class SettlementTerms:
    """How a contract settles: the type, the settlement currency (which may differ from
    the contract currency for quanto/NDF; `None` for physical delivery), and the lag
    (T+`lag` business days)."""

    settlement_type: SettlementType
    currency: Currency | None
    lag: int = 2

    def __post_init__(self) -> None:
        if self.settlement_type is SettlementType.PHYSICAL:
            if self.currency is not None:
                raise ValueError("physical settlement delivers a Quantity, not a currency")
        elif self.currency is None:
            raise ValueError(f"{self.settlement_type.value} settlement requires a currency")


def settlement_date(trade_date: date, terms: SettlementTerms, calendar: Calendar) -> date:
    """The settlement date: `trade_date` + `terms.lag` business days on `calendar`."""
    return calendar.add_business_days(trade_date, terms.lag)
