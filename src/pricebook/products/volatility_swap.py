"""The volatility swap (L2) — frozen data, prices nothing. `GATHERAL-2006` ch 11.

The sibling of `VarianceSwap` and deliberately **not** a flag on it. They share the word "swap"
and the observation window, and nothing else that matters: the variance swap's payoff is linear in
realized variance and is statically replicable, while this one pays `√V_T` and has no replicating
portfolio at all (ch 11: *"there is no replicating portfolio for a volatility swap and the
magnitude of the convexity adjustment is model dependent"*). §6's rule-of-two triggers on a shared
*shape*, and the shape here is the payoff — which differs — so two types, no base class.

**The strike is quoted in volatility, as the name says.** `VarianceSwap` keeps its strike in
variance because its payoff is linear in variance; this one keeps its strike in volatility for the
same reason, and converting at the boundary is how the squaring happens twice or not at all.

Provenance:
  stock:  none — the archived predecessor had no volatility swap
  source: GATHERAL-2006 (docs/references.md) ch 11 ("Volatility Swaps", p. 143) — realized
          volatility as `√V_T` with `V_T := x_T/T`, and the convexity adjustment against the
          variance swap
  oracle: tests/L4/test_volatility_swap.py — exercised through the Heston engine; a swap struck AT
          its own fair volatility is worth zero, the payoff is linear in the volatility difference,
          and an expired IN-THE-MONEY swap is PV 0
  crosscheck: none — a frozen product description
  slice:  tranche 48 — volatility derivatives
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook.foundation.money import Money
from pricebook.market.equity import EquityAsset


@dataclass(frozen=True)
class VolatilitySwap:
    """Pays `notional * (realized volatility - volatility_strike)` at `expiry`.

    Realized volatility is `√V_T` with `V_T` the **annualised** realized variance, matching
    `VarianceSwap`'s convention so the convexity adjustment between them is a difference of two
    comparable numbers rather than of two conventions."""

    asset: EquityAsset
    expiry: date
    volatility_strike: float
    volatility_notional: Money

    def __post_init__(self) -> None:
        if self.volatility_strike < 0.0:
            raise ValueError(
                f"a volatility strike may not be negative, got {self.volatility_strike}"
            )
