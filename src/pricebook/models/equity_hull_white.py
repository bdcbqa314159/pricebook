"""The equity-rate hybrid model: equity under one-factor Hull-White rates (L3).

Four fields, and each is a thing the market quotes separately: the snapshot (§3), the **reused**
`HullWhiteParams` for the rate factor, the equity's own volatility, and the equity-rate correlation.

**`HullWhiteParams` is reused rather than re-declared**, so the library has exactly one Hull-White
convention. Jarrow-Yildirim (tranche 11) and the nominal cap (tranche 10a) already carry the same
bundle, and a second `kappa, sigma` pair spelled differently is how two parts of a library come to
disagree about what `kappa` means.

**The equity volatility here is `sigma_{S/B}`, not the total.** It is the volatility of `S/B` - the
equity *relative to the cash bond* - which is what a market quotes and what survives when the rate
factor is switched off. The **total** volatility of `log S`, which is what Black's `d+/-` needs, is
assembled by the engine from this plus the bond volatility plus their cross term (eq 11.2). Naming
the field for the quoted quantity rather than the derived one keeps the distinction where a reader
meets it.

**No second product type.** A European equity option is one product whatever prices it - the
tranche-33 ruling - and this model is the third pricer for it, alongside Black (30) and Heston (33).
That coexistence exists only because tranche 33b keyed the registry on the model.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 11.1 - the equity-interest rate hybrid; §11.1.2
          specialises to one-factor Hull-White
  oracle: tests/L4/test_equity_rate_hybrid.py - exercised through the engine: the `sigma_r -> 0`
          reduction is bit-identically tranche 30's Black, and the guards here are reached directly
  crosscheck: quantlib AnalyticBSMHullWhiteEngine - deferred to the optional QuantLib tier
  slice:  tranche 34 - the equity-rate hybrid, the arc's first
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import HullWhiteParams


@dataclass(frozen=True)
class EquityHullWhiteModel:
    """Equity under Hull-White rates: the snapshot, the rate factor, the equity vol, and `rho`."""

    market: MarketSnapshot
    rate_params: HullWhiteParams
    equity_vol: float
    correlation: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.equity_vol) or self.equity_vol < 0.0:
            raise ValueError(
                f"the equity volatility must be finite and non-negative, got {self.equity_vol}"
            )
        if self.rate_params.sigma < 0.0:
            raise ValueError(
                f"the rate volatility must be non-negative, got {self.rate_params.sigma}"
            )
        if not -1.0 <= self.correlation <= 1.0:
            raise ValueError(f"the correlation must lie in [-1, 1], got {self.correlation}")
