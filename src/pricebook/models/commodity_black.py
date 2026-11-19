"""The flat lognormal commodity volatility (L3). `CLARK-2014` ch 2.2.4.

Carries the `MarketSnapshot` it was calibrated to, per §3: the engine reaches market data only
through `model.market`, never as a second argument, so a model/market mismatch cannot be expressed.

**Its own type rather than reusing `EquityBlackModel`, and the reason is divergence, not vocabulary
alone.** The two are field-for-field identical today - `(MarketSnapshot, float)` - so §6's
rule-of-two has a case to answer, and the answer is the charter's own DRY rule: *two similar-looking
blocks that represent different ideas must not be merged; merging them couples things that will
diverge*. A commodity's volatility is on a path to diverge from an equity's within this very arc,
where a mean-reverting (Schwartz) spot model gives it a term structure an equity flat vol does not
have. Merging now to save six lines would have to be undone there, and the dispatch would meanwhile
read "a commodity option is priced with an equity model", which is false.

*(Raised to Cowork at the tranche-60 handoff as a design question rather than settled here - §10. If
it rules the other way, the change is a rename and a second `@register`.)*

Provenance:
  stock:  none
  source: CLARK-2014 (docs/references.md) ch 2.2.4 - the Black-76 model on a commodity forward
  oracle: tests/L4/test_commodity_vanilla.py - the non-negative-volatility refusal; the model's
          content is exercised through every price in that file
  crosscheck: none - a frozen two-field carrier, no numerics of its own
  slice:  tranche 60 - the commodity forward and option; the commodity arc opens
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class CommodityBlackModel:
    """A flat lognormal volatility, carrying the snapshot it belongs to."""

    market: MarketSnapshot
    volatility: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValueError(
                f"the volatility must be finite and non-negative, got {self.volatility}"
            )
