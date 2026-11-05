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
    `clean = pv − accrued`; `basis` the discounting currency (`None` = the value's own);
    `standard_error` the PV's own error bar when the PV is an estimate.

    **`standard_error` is §3 invariant 7, and `None` is load-bearing.** For a Monte Carlo price the
    error *is* part of the result — `100 ± 0.3`, not `100` — and an engine that discards it forces
    every downstream oracle to re-simulate or invent a tolerance. `None` means *this is not an
    estimate*, which is different information from `0.0`: zero would claim a perfectly precise
    simulation, and an exact pricer conflated with an infinitely-accurate one is exactly how an
    invented tolerance gets justified. Exact pricers inherit `None` from the default and are not
    touched, so the field rides along only where it is meaningful.

    It is scaled like the PV: an engine multiplying an estimator's mean by a discount factor and a
    notional multiplies the standard error by the same deterministic factor."""

    pv: Money
    accrued: Money | None = None
    basis: Currency | None = None
    standard_error: float | None = None

    @property
    def clean(self) -> Money:
        if self.accrued is None:
            return self.pv
        return self.pv - self.accrued  # carries Money's currency-mixing guard


@dataclass(frozen=True)
class PricingFailure:
    """Why a valuation could not be produced. `reason` is for a human reading a report."""

    reason: str
