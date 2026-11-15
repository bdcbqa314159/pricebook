"""The single-tranche synthetic CDO (L2). `OKANE-2008` ch 16.

Protection on the portfolio losses between an attachment `K₁` and a detachment `K₂`. Frozen data;
it prices nothing.

**No entity set, and no index handle either.** The tranche-43 spec asked which of the two the
homogeneous pool wants; the question dissolves. The LHP is the *infinite homogeneous limit*, so
there is no per-name structure to reference — the pool's identity is its one default curve, which
`CurveSet.survivals` already keys by `ReferenceEntity` (tranche 41b). An index is a reference entity
for keying purposes, so the existing type carries it and a new one would have no second consumer
(§6).

Provenance:
  stock:  none — the archived predecessor had no tranche products
  source: OKANE-2008 ch 16 (the single-tranche synthetic CDO)
  oracle: tests/L4/test_lhp_tranche.py — par-reprice-to-zero, linearity in the contractual spread,
          the equity/mezzanine/senior spread ordering, and the notional scaling
  crosscheck: none — frozen data, no numerics
  slice:  tranche 43 — the LHP synthetic-CDO tranche
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.money import Money
from pricebook.foundation.schedule import CouponSchedule
from pricebook.market.survival_curve import ReferenceEntity


@dataclass(frozen=True)
class TrancheBounds:
    """Where the tranche attaches and detaches, as fractions of the pool's notional.

    Bundled because an attachment without a detachment is not a thing — and compelled by §7: the
    reference entity would otherwise put this product at six fields. Validating them together is
    also the only place the ordering *can* be checked.
    """

    attachment: float
    detachment: float

    def __post_init__(self) -> None:
        for name, value in (("attachment", self.attachment), ("detachment", self.detachment)):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"the {name} must lie in the unit interval as a fraction of the pool's "
                    f"notional, got {value}")
        if self.detachment <= self.attachment:
            raise ValueError(
                f"the detachment {self.detachment} must exceed the attachment "
                f"{self.attachment}: a tranche of zero or negative width has no notional to "
                f"protect, and its survival would divide by zero")

    @property
    def width(self) -> float:
        """`K₂ − K₁` — the tranche's notional as a fraction of the pool's."""
        return self.detachment - self.attachment


@dataclass(frozen=True)
class SyntheticCDOTranche:
    """A tranche of a synthetic CDO on one homogeneous pool."""

    bounds: TrancheBounds
    schedule: CouponSchedule
    spread: float
    notional: Money
    reference_entity: ReferenceEntity
