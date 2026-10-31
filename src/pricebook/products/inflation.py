"""Zero-coupon inflation swap (L2) — pure data.

Five fields, which is the limit, and reaching it is why `Term` exists: the contract needs the
index, the base level, the break-even, the notional **and both ends of its term**, and the rule's
answer to six loose fields is to bundle the cohesive ones rather than raise the limit.

Provenance:
  stock:  _stock/quarry/parked/ng/pricebook_ng/products/inflation.py (mined for CONTENT and
          CORRECTED: that version had `index: str` — a bare string where an identity belongs —
          and **no base index**, so it could only ever have priced eq 1.2.14 at inception, never
          the seasoned 1.2.13. Both fixed here; its shape carries no authority)
  source: BERHAN-2012 (docs/references.md) §1.2.2 — inflation leg 1.2.6, fixed leg 1.2.7
  oracle: priced against 1.2.14 and the 1.2.13 reduction; par identity at the break-even;
          round-trips through its serialised form with the version enforced
  slice:  tranche 2 — the inflation spine
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.cashflow import SCHEMA_VERSION, check_schema_version
from pricebook.foundation.inflation import InflationIndex
from pricebook.foundation.money import Money
from pricebook.foundation.term import Term


@dataclass(frozen=True)
class ZeroCouponInflationSwap:
    """Receive `N[I(TM)/I0 − 1]`, pay `N[(1+b)^TM − 1]`, both at `TM` — a single exchange.

    `base_index` is `I0`, the index level at inception. It is **contract data**, fixed in the
    confirmation, not something to read off a curve: deriving it would be inventing the trade's
    history. PV is taken from the **inflation receiver's** side; the payer's is its negative."""

    index: InflationIndex
    base_index: float
    fixed_rate: float
    term: Term
    notional: Money

    def __post_init__(self) -> None:
        if self.base_index <= 0.0:
            raise ValueError(f"base_index must be positive, got {self.base_index}")

    def to_dict(self) -> dict:
        return {
            "kind": "zc_inflation_swap",
            "schema_version": SCHEMA_VERSION,
            "index": self.index.name,
            "base_index": self.base_index,
            "fixed_rate": self.fixed_rate,
            "term": self.term.to_dict(),
            "notional": self.notional.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ZeroCouponInflationSwap:
        check_schema_version(d)  # before `kind`: in an unknown format, `kind`'s meaning is unknown
        if d.get("kind") != "zc_inflation_swap":
            raise ValueError(f"not a zc_inflation_swap payload: {d.get('kind')!r}")
        return cls(
            InflationIndex(d["index"]),
            float(d["base_index"]),
            float(d["fixed_rate"]),
            Term.from_dict(d["term"]),
            Money.from_dict(d["notional"]),
        )
