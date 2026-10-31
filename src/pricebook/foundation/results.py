"""Engine output: a result or a failure (L0).

**Failure is a VALUE.** A pricer that cannot price returns `PricingFailure` — it does not raise
into a caller that cannot handle it, and it never returns a silent `NaN` that propagates into a
report. That is a contract, so it is tested as one.

**The result is a DECOMPOSITION, not a scalar.** `pv`/`accrued`/`clean` is kept from the first
product even though a bullet cashflow has no accrual: the scope commits to the decomposition, and
trimming it here would mean re-widening a public type at the first coupon-bearing product.

Provenance:
  stock:  _stock/src/pricebook/foundation/results.py (purified as-is)
  source: CLAUDE.md-era §2 invariant 4 (failure is a value) + §3 (PricingResult decomposition)
  oracle: a missing curve and an unregistered product both return a failure, not an exception
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook.foundation.money import Currency, Money


@dataclass(frozen=True)
class PricingResult:
    """A successful valuation. `pv` is the dirty PV; `accrued` the earned-but-unpaid interest;
    `clean = pv − accrued`; `basis` the discounting currency (`None` = the value's own)."""

    pv: Money
    accrued: Money | None = None
    basis: Currency | None = None

    @property
    def clean(self) -> Money:
        if self.accrued is None:
            return self.pv
        return self.pv - self.accrued  # carries Money's currency-mixing guard


@dataclass(frozen=True)
class PricingFailure:
    """Why a valuation could not be produced. `reason` is for a human reading a report."""

    reason: str
