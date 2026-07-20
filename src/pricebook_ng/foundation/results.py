"""Engine I/O value types — PricingResult, PricingFailure (L0).

A result is a **decomposition**, not a scalar (A2): dirty PV + accrued ⇒ clean. It also
records the **basis** it was computed on — the collateral/discounting (CSA) currency — so
a PV is never ambiguous about what it is discounted on (settlement ruling §1). `basis`
is `None` for a value discounted in its own currency / uncollateralised. The numeraire
*choice* is an L3 model concern; this only records it. Failure is a value, never raised.

Greeks, per-flow breakdowns and solver diagnostics are **not** fields here (A3): they had
no consumer and guessed a schema. Each returns — validated — with the engine/risk layer
that produces it (L4/L5), one field at a time.

Provenance:
  quarry: python/pricebook/core/ (pricing result / failure)
  source: redesign/02 spine (A2 decomposition); settlement ruling §1 (basis on the result)
  oracle: clean = pv − accrued; the result records the collateral basis
  slice:  numerics-config (Topic 0 S6); trimmed audit-closure Phase 4 (A3)
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation.money import Currency, Money


@dataclass(frozen=True)
class PricingResult:
    """A successful valuation. `pv` is the dirty PV; `accrued` the earned-but-unpaid
    interest; `clean = pv − accrued`; `basis` the collateral/discounting currency
    (`None` = the value's own currency / uncollateralised)."""

    pv: Money
    accrued: Money | None = None
    basis: Currency | None = None

    @property
    def clean(self) -> Money:
        if self.accrued is None:
            return self.pv
        return self.pv - self.accrued  # Money.__sub__ carries the currency-mixing guard (B2)


@dataclass(frozen=True)
class PricingFailure:
    """A valuation that could not be produced — carried as a value, never raised."""

    reason: str
