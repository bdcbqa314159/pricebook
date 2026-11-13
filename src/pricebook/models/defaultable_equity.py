"""Defaultable equity under independent default (L3) - Brockhaus 13.1.1.

Three fields: the snapshot (§3), the volatility of the **surviving** process, and the
`SurvivalCurve` - **reused** from the XVA arc rather than re-declared, so the library has one
definition of what a hazard rate means. This is the tranche where the equity vertical meets the
credit vertical, and reusing the curve is what makes that a seam rather than a second convention.

**`volatility` is `sigma~`, the vol of `S~`, not of the observed stock.** The observed `S` carries
the jump to default on top of the diffusion, so its implied skew is steeper; `sigma~` is what the
modified-Black formulas take, and the book's own remark is that these formulas are how one *implies*
`sigma~` back out of a quoted surface. Naming the field for the quantity the formula consumes keeps
that distinction where a reader meets it rather than two layers away.

**Independence of `tau` and `S~` is the model's premise, not an approximation.** It is what collapses
the price to a Black formula at all; correlated default (a stock that falls *because* credit
worsened - the chapter's own motivation) is a different model and is not this one.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 13.1.1 - `S_t = S~_t 1{tau>t}` with `tau`
          independent of `S~`; eq (13.3) and the two modified Black-Scholes formulas, read in
          `-raw` and `-layout` and reconciled
  oracle: tests/L4/test_defaultable_equity.py - exercised through the engine; the guards here are
          reached directly, including the shared-clock refusal
  crosscheck: quantlib (defaultable equity / jump-to-default) - deferred to the optional QuantLib tier
  slice:  tranche 35 - defaultable equity, the equity-credit seam
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.market.snapshot import MarketSnapshot
from pricebook.market.survival_curve import SurvivalCurve


@dataclass(frozen=True)
class DefaultableEquityModel:
    """The snapshot, the surviving process's volatility, and the survival curve."""

    market: MarketSnapshot
    volatility: float
    survival: SurvivalCurve

    def __post_init__(self) -> None:
        if not math.isfinite(self.volatility) or self.volatility < 0.0:
            raise ValueError(
                f"the volatility must be finite and non-negative, got {self.volatility}"
            )
        # C12's family: the option's `t` is read on the discount curve AND the survival curve, so
        # two anchors would make one `t` denote two different dates. Refused at construction rather
        # than at pricing, because a mismatched pair is never valid for anything.
        anchor = self.market.valuation_date
        if self.survival.time_measure.anchor != anchor:
            raise ValueError(
                "the survival curve and the snapshot must share one time measure anchor: "
                f"{self.survival.time_measure.anchor} vs {anchor}"
            )
