"""The nominal interest-rate cap/floor (L2). MCINERNEY-ZASTAWNIAK §2.7, §3.5.

A strip of caplets or floorlets on the **spot rate** over each period — the first instrument in this
library that is not an inflation product. It arrives because BERHAN-2012 §5.3.1 sources `σn` by
fitting `(κn, σn)` to ATM *nominal* cap volatilities, and `σn` has been a hand-supplied input since
tranche 6.

Payout per period: `N·τᵢ·[ω·(L(Tᵢ₋₁,Tᵢ) − K)]⁺`, paid at `Tᵢ`, with `ω = +1` for a cap and `−1` for
a floor. Pure data — the pricing is L4's business and the Hull-White analytics are L3's.

**Deliberately the same shape as `InflationCapFloor`, and deliberately NOT sharing a type with it.**
Both carry a schedule, a strike, a notional and a cap/floor flag — but one strikes the *year-on-year
index ratio* and the other a *nominal rate*, which are different underlyings valued under different
measures. That is the shared-*word* trap §6 names: "cap" is the word, and the payoff is not the
shape. Two frozen types, no base class, and the rule of two does not fire on a coincidence of
fields.

Provenance:
  stock:  — built fresh; the quarry's rates caps carry a conventions/calendar apparatus this
          tranche has no consumer for
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §2.7 (the caplet payoff and its
          put-on-a-zero-coupon-bond decomposition), §3.5 eq 3.21 (the Hull-White price);
          BERHAN-2012 §5.3.1 for why this instrument is pulled in at all
  oracle: tests/L4/test_nominal_cap.py — the strip sums its caplets, invariant 6 excludes settled
          periods, round-trip + version refusal; the caplet analytics are oracled at
          tests/L3/test_hull_white_caplet.py against an independent transcription
  crosscheck: none — a frozen cap description, no numerics
  slice:  tranche 10(a) — the Hull-White nominal cap
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule


@dataclass(frozen=True)
class NominalCap:
    """A nominal interest-rate cap or floor: its `schedule`, `strike` rate, `notional` and `kind`."""

    schedule: CouponSchedule
    strike: float
    notional: Money
    kind: CapFloorKind

    def to_dict(self) -> dict:
        return {
            "kind_": "nominal_cap",
            "schema_version": SCHEMA_VERSION,
            "schedule": self.schedule.to_dict(),
            "strike": self.strike,
            "notional": self.notional.to_dict(),
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NominalCap:
        check_schema_version(d)
        if d.get("kind_") != "nominal_cap":
            raise ValueError(f"not a nominal_cap payload: {d.get('kind_')!r}")
        return cls(
            CouponSchedule.from_dict(d["schedule"]),
            float(d["strike"]),
            Money.from_dict(d["notional"]),
            CapFloorKind(d["kind"]),
        )
