"""The geometric-average commodity option (L2) - pure data. `CLARK-2014` ch 2.7.3.

A **sibling** of `CommodityOption`, not a subclass. They share the word "option" and not the payoff:
one settles on the terminal price, the other on an average over a schedule, which is a different
idea - the charter's real-leg-vs-year-on-year-leg case, and the same call tranche 60 made for the
two Black models. No base class, no polymorphism; the engines dispatch on the concrete type.

Five fields, the §7 ceiling exactly. The **fixing schedule is one field** (the averaging dates), and
it is constitutive: two Asians differing only in their schedule average different dates and are
different instruments, so it is carried and the round trip exercises it (§6).

**Settlement is at the last fixing.** Clark allows `t_n <= T` - an averaging window that ends before
payment - but a separate settlement date is a sixth field and nothing pulls it yet (§6: nothing
speculative). A trade that pays later than its last fixing is `to_go_deeper`.

Provenance:
  stock:  none - the archived predecessor had no Asian product
  source: CLARK-2014 (docs/references.md) ch 2.7.3 - discrete-fixing geometric average options
          (Kemna and Vorst, 1990)
  oracle: tests/L4/test_commodity_geometric_asian.py - the empty-schedule, ascending-order and
          duplicate-fixing refusals; the strike guard; the schema round trip and its refusal of a
          future version; the schedule seen by equality
  slice:  tranche 61 - geometric Asian options, exact closed form
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.money import Currency, Money
from pricebook.market.commodity import CommodityAsset
from pricebook.products.commodity_option import CommodityOptionKind


@dataclass(frozen=True)
class CommodityGeometricAsian:
    """A European-style option on the **geometric average** of one commodity over a fixing schedule."""

    asset: CommodityAsset
    strike: float
    fixings: tuple[date, ...]
    notional: Money
    kind: CommodityOptionKind

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"the strike must be positive, got {self.strike}")
        if not self.fixings:
            raise ValueError("an average-rate option needs at least one fixing date")
        # Strictly ascending, and the strictness matters twice: a schedule that doubles back is not
        # a schedule, and the source's own `sum t_i [2(n-i)+1]` evaluation of the variance double
        # sum is only valid for a sorted schedule (CLARK-2014 p.64). A duplicate is a legitimate
        # *mathematical* input - the variance formula handles coincident times - but it is not a
        # real averaging schedule, so it is refused at the product rather than silently averaged.
        if any(b <= a for a, b in zip(self.fixings, self.fixings[1:])):
            raise ValueError(
                f"the fixing dates must be strictly ascending, got {self.fixings}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset.symbol,
            "strike": self.strike,
            "fixings": [d.isoformat() for d in self.fixings],
            "notional": {
                "amount": self.notional.amount,
                "currency": self.notional.currency.code,
            },
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CommodityGeometricAsian:
        # Before `kind`: in an unknown format, `kind`'s meaning is unknown too.
        check_schema_version(d)
        return cls(
            asset=CommodityAsset(str(d["asset"])),
            strike=float(d["strike"]),
            fixings=tuple(date.fromisoformat(x) for x in d["fixings"]),
            notional=Money(
                float(d["notional"]["amount"]), Currency(str(d["notional"]["currency"]))
            ),
            kind=CommodityOptionKind(d["kind"]),
        )
