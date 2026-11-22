"""An option on realized variance (L2). KWOK-ZHENG ch 5.

Pays `notional · (realized variance − strike)⁺` for a call, the mirror for a put. Pure data.

**The strike is in VARIANCE units**, for T31's reason: the payoff is linear in variance, and
converting at the boundary invites the squaring to happen twice or not at all.

**The asset is carried although the Heston pricer does not read it** (§6 allows that only when the
field is constitutive of identity, and it is): an option on realized variance is an option on *some
asset's* realized variance, and a payload without it cannot say what the trade is on. `HestonModel`
carries its parameters directly rather than keyed by asset, which is why the pricer does not need
it — not a reason for the contract to omit it. The round trip exercises it.

Provenance:
  stock:  none — the migration tree carried no option on realized variance
  source: KWOK-ZHENG (docs/references.md) ch 5 — "Options on Discrete Realized Variance"; the
          payoff is the `(RV − K)+` of §5's opening, with §5.2's normal approximation the pricing
          route (see engine/realized_variance_option.py)
  oracle: tests/L4/test_realized_variance_option.py — the round trip, the refusals, and the kind
  crosscheck: none — pure data, no numerical convention to cross-check
  slice:  tranche 75 — options on realized variance under Heston
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset


@dataclass(frozen=True)
class RealizedVarianceOption:
    """`notional · [ω(RV − strike)]⁺` at `expiry`, with `ω = +1` for a call on variance."""

    asset: EquityAsset
    expiry: date
    variance_strike: float
    variance_notional: Money
    kind: CapFloorKind

    def __post_init__(self) -> None:
        if self.variance_strike < 0.0:
            raise ValueError(f"a variance strike cannot be negative, got {self.variance_strike}")

    def to_dict(self) -> dict:
        return {
            "kind_": "realized_variance_option",
            "schema_version": SCHEMA_VERSION,
            "asset": self.asset.ticker,
            "expiry": self.expiry.isoformat(),
            "variance_strike": self.variance_strike,
            "variance_notional": self.variance_notional.to_dict(),
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> RealizedVarianceOption:
        check_schema_version(d)
        if d.get("kind_") != "realized_variance_option":
            raise ValueError(f"not a realized_variance_option payload: {d.get('kind_')!r}")
        return cls(
            EquityAsset(d["asset"]),
            date.fromisoformat(d["expiry"]),
            float(d["variance_strike"]),
            Money.from_dict(d["variance_notional"]),
            CapFloorKind(d["kind"]),
        )
