"""The variance swap (L2) - frozen data, prices nothing.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 3.7 / 5.8 - the variance swap and its payoff
  oracle: tests/L4/test_variance_swap.py - exercised through the replication engine; a swap struck
          AT the fair variance is worth zero, and the payoff is linear in the variance difference
  crosscheck: none - a frozen product description
  slice:  tranche 31 - the variance swap by static replication
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset


@dataclass(frozen=True)
class VarianceSwap:
    """Pays `notional * (realised variance - variance_strike)` at `expiry`.

    The strike is quoted in **variance**, not volatility: `0.06` is a 24.5% vol. Keeping it in
    variance units is the faithful representation - the payoff is linear in variance and the
    replication produces variance - and converting at the boundary would invite the squaring to
    happen twice or not at all.
    """

    asset: EquityAsset
    expiry: date
    variance_strike: float
    variance_notional: Money

    def __post_init__(self) -> None:
        if self.variance_strike < 0.0:
            raise ValueError(f"a variance strike may not be negative, got {self.variance_strike}")
