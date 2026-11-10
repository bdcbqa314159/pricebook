"""Flat Black-Scholes volatility on an equity underlying (L3).

The minimum model the equity vanilla needs: one volatility, plus the snapshot it was quoted against
(§3 - the model carries its `MarketSnapshot`, so a model/market mismatch cannot be expressed).

**Flat on purpose, and only for now.** The implied-vol *surface* and local/stochastic volatility are
tranches 32 and 33; a flat vol is what a single European option at one strike actually needs, and
carrying surface machinery before a product pulls it is the speculation §6 forbids.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 5 - Black-Scholes on a dividend-paying equity; the
          option formula itself is Black (verified since tranche 5) on the dividend forward
  oracle: tests/L4/test_equity_vanilla.py - the zero-dividend reduction is bit-identically
          `black_price`, so this model adds no numerical content of its own beyond the vol
  crosscheck: quantlib BlackConstantVol - deferred to the optional QuantLib tier
  slice:  tranche 30 - the equity underlying: the equity & hybrids arc opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class EquityBlackModel:
    """A flat lognormal volatility, carrying the snapshot it belongs to."""

    market: MarketSnapshot
    volatility: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValueError(f"the volatility must be finite and non-negative, got {self.volatility}")
