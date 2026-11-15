"""The first-to-default basket (L2). `OKANE-2008` ch 15.1-15.2.

Protection that triggers on the **first** of `N` names to default: the protection leg pays `(1−R)`
once, at that first default if it falls before maturity, and the premium leg pays a spread on a
risky annuity that knocks out there. Frozen data; it prices nothing.

**Homogeneous loss**, which is the scope: every name carries the same `(1−R)`, so the contract does
not need to know *which* name defaulted. Per-name recovery (and so the defaulter's identity) is
`docs/to_go_deeper.md`.

**No recovery field.** `R` is a market assumption and rides on the model, as tranche 38 ruled for
the single-name CDS — a recovery on the product would let two baskets on the same names carry
different ones, which is not a thing the market quotes.

Provenance:
  stock:  none — the archived predecessor had no basket products
  source: OKANE-2008 ch 15.1 (default baskets), 15.2 (the first-to-default basket)
  oracle: tests/L4/test_first_to_default_basket.py — par-reprice-to-zero, linearity in the
          contractual spread, and the spread bounded by the widest name and the sum of names
  crosscheck: none — frozen data, no numerics
  slice:  tranche 42 — the first-to-default basket
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule
from pricebook.market.survival_curve import ReferenceEntity


@dataclass(frozen=True)
class FirstToDefaultBasket:
    """The names protected, the premium schedule, the contractual spread and the notional."""

    entities: tuple[ReferenceEntity, ...]
    schedule: CouponSchedule
    spread: float
    notional: Money

    def __post_init__(self) -> None:
        if len(self.entities) < 2:
            raise ValueError(
                f"a first-to-default basket needs at least two names, got "
                f"{len(self.entities)}. One name is the degenerate reduction — which the suite "
                f"asserts on the basket *curve* — not a contract shape worth encoding")
        if len(set(self.entities)) != len(self.entities):
            raise ValueError(
                f"a basket may not name the same entity twice: {[e.name for e in self.entities]}. "
                f"A duplicate would enter the survival product twice, which is a wrong basket "
                f"rather than a wider one")
