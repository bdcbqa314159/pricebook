"""The credit model (L3) - the snapshot plus the assumed recovery rate.

Thin by design: all the survival machinery is tranche 25's `SurvivalCurve`, reused whole rather than
re-declared, so the library has one definition of what a hazard rate means. What this adds is the
one thing the curve cannot carry - the **recovery assumption** - and the §3 binding to the snapshot
the curves were quoted against.

**Why `R` is a model parameter and not market data.** A survival curve is bootstrapped *from* quotes;
a recovery rate is assumed *in order to* bootstrap one. The market quotes spreads, not recoveries,
and the standard 40% is a convention every desk applies to back out hazards. Putting it on the model
keeps the thing that is assumed next to the thing that assumes it, and makes a recovery/curve
mismatch inexpressible.

Provenance:
  stock:  none
  source: OKANE-2008 (docs/references.md) ch 6 - the valuation inputs: the risky curve `Q`, the
          discount curve `Z`, and the assumed recovery `R`
  oracle: tests/L4/test_credit_default_swap.py - the guards here are reached directly (recovery
          range, the shared-clock refusal, a snapshot with no survival curve)
  crosscheck: quantlib IsdaCdsEngine - deferred to the optional QuantLib tier
  slice:  tranche 38 - the credit default swap: project five opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import SurvivalCurve


@dataclass(frozen=True)
class CreditModel:
    """The snapshot (carrying `Q` and `Z`) and the assumed recovery rate."""

    market: MarketSnapshot
    recovery: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.recovery) or not 0.0 <= self.recovery <= 1.0:
            raise ValueError(f"the recovery rate must lie in [0, 1], got {self.recovery}")
        if not self.market.curves.survivals:
            raise ValueError(
                "a credit model needs a survival curve in the snapshot; none was supplied"
            )
        anchor = self.market.valuation_date
        for currency, curve in self.market.curves.survivals.items():
            if curve.time_measure.anchor != anchor:
                raise ValueError(
                    f"the survival curve for {currency} and the snapshot must share one time "
                    f"measure anchor: {curve.time_measure.anchor} vs {anchor}"
                )

    def survival(self, currency) -> SurvivalCurve:
        """`Q` for `currency` - reached through the snapshot, never passed alongside it (§3)."""
        try:
            return self.market.curves.survival(currency)
        except KeyError as exc:
            raise KeyError(f"no survival curve for {currency} in the snapshot") from exc
