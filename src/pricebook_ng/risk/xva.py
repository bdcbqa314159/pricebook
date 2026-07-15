"""XVA — valuation adjustments on top of the risk-free mark (L5 risk & capital).

The first XVA: unilateral **CVA**, the market value of counterparty default risk.
Given a trade's *expected positive exposure* profile `EE(t) = E[(V(t))^+]` on a
time grid, CVA discounts the expected loss over each interval's default probability:

    CVA = (1 - R) * sum_i  EE(t_i) * DF(t_i) * (Q(t_{i-1}) - Q(t_i))

This is exactly a CDS **protection leg** (`_protection_pv`/`cds_pv` at zero spread)
with the unit notional replaced by the exposure profile — CVA is "buying protection
on your own counterparty exposure." Unilateral form: exposure and default are taken
independent (no wrong-way risk), own default ignored (that is DVA).

Exposure generation is deliberately upstream and out of scope here: `EE(t)` is an
input. For a deterministic trade it is analytic; for an optional/path-dependent one
it comes from a Monte-Carlo exposure engine — a later slice. This module is the XVA
*integrator*, keyed to the counterparty's survival curve in the snapshot (A5), so a
credit bump (`bump_curve`/`credit01`) already gives CVA sensitivity for free.

Provenance:
  quarry: python/pricebook/risk/ (xva)
  source: Gregory, The xVA Challenge; Brigo & Mercurio (unilateral CVA)
  oracle: unit-exposure CVA == CDS protection leg (cds_pv @ 0 spread); linearity; Q≡1 -> 0
  slice:  cva (unilateral)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.market.keys import MarketKey
from pricebook_ng.market.snapshot import MarketSnapshot


@dataclass(frozen=True)
class ExposureProfile:
    """Expected positive exposure `EE(t) = E[(V(t))^+]` on a time grid. `grid[0]` is
    the valuation date (exposure today); the CVA sum runs over the intervals
    `(grid[i-1], grid[i]]`. `grid` and `ee` are aligned and the same length."""

    grid: tuple[date, ...]
    ee: tuple[float, ...]


def cva(
    profile: ExposureProfile, snapshot: MarketSnapshot, key: MarketKey, recovery: float
) -> float:
    """Unilateral CVA against the counterparty whose survival curve is at `key`.

    The survival curve is reached through the `CurveHandle` capability (`df` = Q, the
    A5 survival-as-discount-factor alias), so no concrete curve type is assumed."""
    survival = snapshot.curves[key]
    discount = snapshot.discount_curve
    grid, ee = profile.grid, profile.ee
    return (1.0 - recovery) * sum(
        ee[i] * discount.df(grid[i]) * (survival.df(grid[i - 1]) - survival.df(grid[i]))
        for i in range(1, len(grid))
    )
