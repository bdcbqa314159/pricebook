"""Zero-coupon inflation swap (L2) — pure data.

Five fields, which is the limit, and reaching it is why `Term` exists: the contract needs the
index, the base level, the break-even, the notional **and both ends of its term**, and the rule's
answer to six loose fields is to bundle the cohesive ones rather than raise the limit.

Provenance:
  stock:  _stock/quarry/pricebook/fixed_income/inflation.py (the linker, mined for CONTENT: its
          8 mutable ctor params and bound schedule generator carry no authority, but its placing
          `cpi_lag_months` ON THE INSTRUMENT — "3 for US TIPS, 8 for UK ILG" — is the insight kept)
          and _stock/quarry/parked/ng/pricebook_ng/products/inflation.py (mined for CONTENT and
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
from pricebook.foundation.schedule import RealLeg, YoYLeg
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


@dataclass(frozen=True)
class InflationLinkedBond:
    """A linker: real coupons and principal, both uplifted by `I(t)/I0`.

    `ILB(t,TM) = I(t)/I0 · N[ Σ C·ψᵢ·Pr(t,Tᵢ) + Pr(t,TM) ]` — the paper's 1.2.5 generalised with
    the accrual fraction a real linker pays. **No nominal curve appears**: a linker is a real bond
    scaled by the index ratio, which is the foreign-currency analogy at its clearest.

    **`indexation_lag` is contract data, and it is constitutive of the contract.** Two linkers on
    the *same* index with different indexation lags are different instruments — 3-month for US
    TIPS, 8-month for UK index-linked gilts — so a serialised linker without it is ambiguous, and
    the round-trip is its present consumer. It is distinct from an index's *publication* lag (when
    a print becomes known), which is market metadata and lives nowhere here; the stock conflates
    the two under one name, and this field is the half that genuinely belongs on a contract.

    The lag is **carried, not yet resolved.** Deciding *which monthly print* is `I(t)` needs a
    monthly index series and an interpolation convention (TIPS interpolate daily, UK ILGs do not) —
    `deferred→` the index-series tranche. Until then the snapshot supplies `I(t)` already resolved.
    """

    index: InflationIndex
    base_index: float
    leg: RealLeg
    notional: Money
    indexation_lag: int = 3

    def __post_init__(self) -> None:
        if self.base_index <= 0.0:
            raise ValueError(f"base_index must be positive, got {self.base_index}")
        if self.indexation_lag < 0:
            raise ValueError(f"indexation_lag cannot be negative, got {self.indexation_lag}")

    def to_dict(self) -> dict:
        return {
            "kind": "inflation_linked_bond",
            "schema_version": SCHEMA_VERSION,
            "index": self.index.name,
            "base_index": self.base_index,
            "leg": self.leg.to_dict(),
            "notional": self.notional.to_dict(),
            "indexation_lag": self.indexation_lag,
        }

    @classmethod
    def from_dict(cls, d: dict) -> InflationLinkedBond:
        check_schema_version(d)
        if d.get("kind") != "inflation_linked_bond":
            raise ValueError(f"not an inflation_linked_bond payload: {d.get('kind')!r}")
        return cls(
            InflationIndex(d["index"]),
            float(d["base_index"]),
            RealLeg.from_dict(d["leg"]),
            Money.from_dict(d["notional"]),
            int(d["indexation_lag"]),
        )


@dataclass(frozen=True)
class YearOnYearInflationSwap:
    """Receive `Σ N·ψᵢ·[I(Tᵢ)/I(Tᵢ₋₁) − 1]`, pay `Σ N·ψᵢ·C_fix` (eqs 1.2.17, 1.2.18).

    Three fields, and notably **no base index**: a YoY swap's floating side is a chain of
    *period-on-period* ratios, so it has no single `I0` the way a ZC swap or a linker does. The
    index is named because it is the contract's underlying, not because the price reads a level —
    at inception eq 2.5.13 contains no index level at all."""

    index: InflationIndex
    leg: YoYLeg
    notional: Money

    def to_dict(self) -> dict:
        return {
            "kind": "yoy_inflation_swap",
            "schema_version": SCHEMA_VERSION,
            "index": self.index.name,
            "leg": self.leg.to_dict(),
            "notional": self.notional.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> YearOnYearInflationSwap:
        check_schema_version(d)
        if d.get("kind") != "yoy_inflation_swap":
            raise ValueError(f"not a yoy_inflation_swap payload: {d.get('kind')!r}")
        return cls(
            InflationIndex(d["index"]),
            YoYLeg.from_dict(d["leg"]),
            Money.from_dict(d["notional"]),
        )
