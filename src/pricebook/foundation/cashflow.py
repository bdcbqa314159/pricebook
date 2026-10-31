"""The instrument atom: a cashflow (L0).

Provenance:
  stock:  _stock/src/pricebook/foundation/cashflow.py (purified: the flow itself; `Leg` and the
          `accrual` field stay in the stock — accrual belongs with the first coupon-bearing
          product, which can state what it means)
  source: CLAUDE.md-era §3 vocabulary (Cashflow as the instrument atom)
  oracle: round-trips through its serialised form; the tranche-1 closed form prices one
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money

#: Bump when the serialised shape changes incompatibly. Present from the first product because a
#: format with no version cannot change without silently breaking a reader — and the scope line
#: makes this representation the library's durable public contract.
SCHEMA_VERSION = 1


def check_schema_version(payload: dict) -> int:
    """Read and validate a payload's `schema_version`. **The whole point of stamping one.**

    Written as its own function, called by every `from_dict`, because a version that is *written
    and never read* is strictly worse than no version at all: it looks like a guarantee and is
    not. The first cut of this module stamped the field, asserted in a test that it was
    *present*, and never dereferenced it — so a v1 payload would have loaded silently under v2
    code, defeating the field's stated purpose. That is this project's named defect shape
    (resolve-don't-just-exist), and the presence test could not have caught it.

    A **future** version is refused rather than best-effort parsed: a reader that guesses at a
    format it does not know produces a plausible wrong object, which is worse than a failure.
    Older versions are readable today only because v1 is the only version; when v2 lands it
    brings its own migration for v1, and this is the one place that decision is expressed."""
    raw = payload.get("schema_version")
    if raw is None:
        raise ValueError("payload carries no schema_version; refusing to guess at its format")
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise ValueError(f"schema_version must be a positive integer, got {raw!r}")
    if raw > SCHEMA_VERSION:
        raise ValueError(
            f"payload is schema_version {raw}; this build reads up to {SCHEMA_VERSION}"
        )
    return raw


@dataclass(frozen=True)
class Cashflow:
    """A single payment of `amount` on `date`. Pure data — it does not discount itself."""

    date: date
    amount: Money

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "date": self.date.isoformat(),
            "amount": self.amount.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Cashflow:
        check_schema_version(d)
        return cls(date.fromisoformat(d["date"]), Money.from_dict(d["amount"]))
