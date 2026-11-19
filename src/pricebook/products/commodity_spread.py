"""The commodity spread option (L2) - pure data. `CLARK-2014` ch 2.9.

An option on `S(1) - S(2) - K`: the crack, spark and calendar spreads a commodity desk trades. A
sibling of the single-asset options, not a generalisation of them - two underlyings is a different
instrument, not a vanilla with a parameter.

**The legs are bundled because the limit compelled it, and that is the §7 case exactly.** Six loose
fields (two assets, strike, expiry, notional, direction) is over the ceiling, and the rule says the
fix is always to group cohesive parameters into a frozen value object, never to raise the limit. The
cohesive pair is the two assets, which belong together for a reason stronger than arithmetic: a
spread is defined by *which* leg is received and which surrendered, so they are an **ordered pair**
and equality must see the order. `SpreadLegs` is concrete and frozen, with no polymorphism - the §6
"compelled bundling is vocabulary, not abstraction" reading.

**The strike may be zero or negative, unlike every other option in this tree.** Zero strike is the
Margrabe exchange option (surrender one asset, receive the other, no payment), and a negative strike
just means the holder is paid to enter - both are ordinary spread trades. So there is no
positive-strike guard here. The constraint that does exist is `F(2) + K > 0`, since Kirk matches a
lognormal to `S(2) + K`; that needs market data, so it is a `PricingFailure` from the engine rather
than a `ValueError` at construction (§3.4 - failure is a value, and a precondition that needs the
market cannot be checked without it).

Provenance:
  stock:  none - the archived predecessor had no spread option
  source: CLARK-2014 (docs/references.md) ch 2.9 - two-asset spread options, 2.9.1 Margrabe,
          2.9.2 Kirk
  oracle: tests/L4/test_commodity_spread.py - the legs asserted an ORDERED pair seen by equality;
          the same-leg refusal; the schema round trip and its refusal of a future version; and a
          negative strike asserted LEGITIMATE (worth more than the price it prices)
  slice:  tranche 63 - spread options: Margrabe and Kirk
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
class SpreadLegs:
    """The two assets of a spread, as an **ordered** pair: `long - short`.

    Ordered rather than a set, because `WTI - BRENT` is not `BRENT - WTI` and equality has to see
    the difference. Named `long`/`short` rather than `first`/`second` so the sign convention is in
    the vocabulary instead of a comment.
    """

    long: CommodityAsset
    short: CommodityAsset

    def __post_init__(self) -> None:
        if self.long == self.short:
            raise ValueError(
                f"the two legs of a spread must differ, got {self.long.symbol} on both sides"
            )


@dataclass(frozen=True)
class CommoditySpreadOption:
    """A European option on the spread `S_long - S_short - K`."""

    legs: SpreadLegs
    strike: float
    expiry: date
    notional: Money
    kind: CommodityOptionKind

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "long": self.legs.long.symbol,
            "short": self.legs.short.symbol,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "notional": {
                "amount": self.notional.amount,
                "currency": self.notional.currency.code,
            },
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CommoditySpreadOption:
        # Before `kind`: in an unknown format, `kind`'s meaning is unknown too.
        check_schema_version(d)
        return cls(
            legs=SpreadLegs(
                long=CommodityAsset(str(d["long"])), short=CommodityAsset(str(d["short"]))
            ),
            strike=float(d["strike"]),
            expiry=date.fromisoformat(d["expiry"]),
            notional=Money(
                float(d["notional"]["amount"]), Currency(str(d["notional"]["currency"]))
            ),
            kind=CommodityOptionKind(d["kind"]),
        )
