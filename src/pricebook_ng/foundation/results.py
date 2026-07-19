"""Engine I/O value types — PricingResult, PricingFailure, DiscountBasis (L0).

A result is a **decomposition**, not a scalar (A2): dirty PV + accrued ⇒ clean. It also
records the **basis** it was computed on — the value currency (in `pv`) and the
collateral/discounting basis — so a PV is never ambiguous about what it is expressed in
(settlement ruling §1). The numeraire *choice* is an L3 model concern; this only records
it. Failure is a value, never an exception.

Provenance:
  quarry: python/pricebook/core/ (pricing result / failure)
  source: redesign/02 spine (A2 decomposition); settlement ruling §1 (basis on the result)
  oracle: clean = pv − accrued; the result records currency and collateral basis
  slice:  numerics-config (Topic 0 S6)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pricebook_ng.foundation.money import Currency, Money


@dataclass(frozen=True)
class DiscountBasis:
    """What a PV is discounted on: the collateral (CSA) currency. `None` = discounted in
    the value's own currency / uncollateralised. Records the numeraire; it does not
    choose it (that is L3)."""

    collateral: Currency | None = None


@dataclass(frozen=True)
class PricingResult:
    """A successful valuation — the full decomposition vocabulary, fixed once and
    populated on demand (design-complete, so no asset class retouches L0 to add a field).
    `pv` is the dirty PV; `accrued` the earned-but-unpaid interest; `clean = pv − accrued`;
    `basis` the collateral/discounting basis; `cashflow_breakdown` the per-flow PVs;
    `sensitivities` named greeks (L5 fills these); `diagnostics` solver/engine notes."""

    # fields-exempt: output record — a valuation's full decomposition (A2/A4.4), like XvaReport

    pv: Money
    accrued: Money | None = None
    basis: DiscountBasis | None = None
    cashflow_breakdown: tuple[Money, ...] | None = None
    sensitivities: Mapping[str, float] | None = None
    diagnostics: Mapping[str, Any] | None = None

    @property
    def clean(self) -> Money:
        if self.accrued is None:
            return self.pv
        return Money(self.pv.amount - self.accrued.amount, self.pv.currency)


@dataclass(frozen=True)
class PricingFailure:
    """A valuation that could not be produced — carried as a value, never raised."""

    reason: str
