"""The two-asset commodity Black model (L3). `CLARK-2014` ch 2.9.

Two volatilities and **a correlation** - the one genuinely-new parameter of the spread tranche.
Carries the `MarketSnapshot` it belongs to, per §3, so the engine reaches market data only through
`model.market`.

**A distinct type from `CommodityBlackModel`, and here §6 is not even close.** Two assets plus a
correlation is a different idea from one asset's flat vol, the field lists differ, and the dispatch
must tell a spread engine from a vanilla one. (The borderline §6 question was tranche 60's
`CommodityBlackModel` versus `EquityBlackModel`, which were field-identical; this is not that case.)

**The correlation is a MODEL parameter, not a product field**, which is the same line §3 draws
everywhere: a spread *contract* says which legs and what strike; how the two legs co-move is a
market view, and two desks pricing one contract may hold different correlations. Putting it on the
product would make `rho` part of the trade's identity and a re-marked correlation a different trade.

`|rho| <= 1` is refused rather than clamped: a correlation outside the unit interval makes the
combined variance negative, so the method is undefined rather than inaccurate, and clamping would
price a plausible wrong number from an impossible input.

Provenance:
  stock:  none
  source: CLARK-2014 (docs/references.md) ch 2.9 - eqs (2.136a/b), the two correlated SDEs
  oracle: tests/L4/test_commodity_spread.py - the unit-interval and non-finite correlation
          refusals, the non-negative volatility refusals; the model's content is exercised by
          every price in that file, and the correlation specifically by the realised-covariance pin
  crosscheck: none - a frozen four-field carrier, no numerics of its own
  slice:  tranche 63 - spread options: Margrabe and Kirk
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class CommoditySpreadModel:
    """Flat lognormal volatilities for the two legs and the correlation between them."""

    market: MarketSnapshot
    vol_long: float
    vol_short: float
    correlation: float

    def __post_init__(self) -> None:
        for name, value in (("vol_long", self.vol_long), ("vol_short", self.vol_short)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"the volatility {name} must be finite and non-negative, got {value}"
                )
        if not math.isfinite(self.correlation) or not -1.0 <= self.correlation <= 1.0:
            raise ValueError(
                f"the correlation must be finite and within [-1, 1], got {self.correlation}"
            )
