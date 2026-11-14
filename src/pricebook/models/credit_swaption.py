"""The credit-swaption model (L3) — a `CreditModel` plus the forward-spread vol. `OKANE-2008` 9.3.

Thin by composition: everything about the credit curves is tranche 38's `CreditModel`, held whole
rather than re-declared. The only thing a swaption needs beyond a CDS is a **volatility for the
forward spread**, because the forward CDS itself is curve arithmetic and needs none.

**Why it composes rather than copies.** A flat `(market, recovery, volatility)` would restate the
recovery validation and the survival lookup, and the two could then disagree about either. Holding
a `CreditModel` means a forward CDS and the swaption written on it are guaranteed to be looking at
the same curves and the same recovery — and `market` is exposed as a property so §3's "the engine
reaches market data only through `model.market`" holds unchanged.

**`σ` is a model input, flat, exactly as tranche 30's equity Black vol is.** A quoted CDS-swaption
vol surface is a different capability and nothing pulls it yet (`docs/to_go_deeper.md`).

Provenance:
  stock:  none
  source: OKANE-2008 ch 9.3 — Black on the forward spread under the risky-annuity measure
  oracle: tests/L4/test_forward_cds_default_swaption.py — the refusals, and the vol's effect
          (zero-vol intrinsic, monotonicity, material time value)
  crosscheck: quantlib CdsOption / BlackCdsOptionEngine — deferred, dev-only
  slice:  tranche 40 — the forward CDS and the default swaption
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.foundation.money import Currency
from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.credit import CreditModel


@dataclass(frozen=True)
class CreditSwaptionModel:
    """`CreditModel` + the Black volatility of the forward spread."""

    credit: CreditModel
    volatility: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValueError(
                f"the forward-spread volatility must be finite and non-negative, got "
                f"{self.volatility}; zero is the deterministic case and prices the intrinsic")

    @property
    def market(self) -> MarketSnapshot:
        """§3 — the engine reaches market data only through this."""
        return self.credit.market

    @property
    def recovery(self) -> float:
        return self.credit.recovery

    def survival(self, currency: Currency) -> SurvivalCurve:
        return self.credit.survival(currency)
