"""The European commodity option (L2) - pure data. `CLARK-2014` ch 2.5.

A frozen description of the trade and nothing more: it does not price itself, and it carries no
market or model data. Five fields, which is the §7 ceiling exactly - nothing here is bundled, since
each is an independent dimension of the contract.

**Its own `kind` enum, following the precedent the tree already set three times.** `CapFloorKind`,
`SwaptionKind` and `EquityOptionKind` each name their own direction, because a cap is not a call in
anyone's vocabulary and neither is a commodity option an equity one. What they *share* is the
`omega` shape, which is why `black_price` depends on that shape and not on any one enum - and why a
fourth enum costs nothing and keeps the vocabulary honest.

Provenance:
  stock:  none - the archived predecessor had no commodity option product
  source: CLARK-2014 (docs/references.md) ch 2.5 - European options on spot and on futures
  oracle: tests/L4/test_commodity_vanilla.py - the strike guard, the schema round trip, and the
          refusal of a future schema version
  slice:  tranche 60 - the commodity forward and option; the commodity arc opens
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.money import Currency, Money
from pricebook.market.commodity import CommodityAsset


class CommodityOptionKind(Enum):
    """`omega = +1` (call) or `omega = -1` (put)."""

    CALL = "call"
    PUT = "put"

    @property
    def omega(self) -> float:
        return 1.0 if self is CommodityOptionKind.CALL else -1.0


@dataclass(frozen=True)
class CommodityOption:
    """A European option on one commodity: strike, expiry, notional, direction."""

    asset: CommodityAsset
    strike: float
    expiry: date
    notional: Money
    kind: CommodityOptionKind

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"the strike must be positive, got {self.strike}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset.symbol,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "notional": {
                "amount": self.notional.amount,
                "currency": self.notional.currency.code,
            },
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CommodityOption:
        # Before `kind`: in an unknown format, `kind`'s meaning is unknown too.
        check_schema_version(d)
        return cls(
            asset=CommodityAsset(str(d["asset"])),
            strike=float(d["strike"]),
            expiry=date.fromisoformat(d["expiry"]),
            notional=Money(
                float(d["notional"]["amount"]), Currency(str(d["notional"]["currency"]))
            ),
            kind=CommodityOptionKind(d["kind"]),
        )
