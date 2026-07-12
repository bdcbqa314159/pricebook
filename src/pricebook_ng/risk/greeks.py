"""Greeks by bump-and-reprice on the Pricable protocol (L5).

Generic over the product/model: `dv01` central-differences a `Pricable` under a
parallel rate shift of the snapshot's curve. The same function prices rate delta
for a cashflow, a bond, a swap, or an HW swaption — the swaption case rebuilds
the model under the bumped snapshot (Amendment A1).

Provenance:
  quarry: python/pricebook/risk/ (bump-and-reprice greeks; ex-L3)
  source: redesign/02_spine.md (risk at L5, bump the snapshot re-run the engine)
  oracle: dv01 matches analytic (cashflow, bond) < 1e-6; rebuilds the model (swaption)
  slice:  S00 (dv01 introduced); L5-risk (generalised onto Pricable)
"""

from __future__ import annotations

from dataclasses import replace

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.risk.pricable import Pricable

_ONE_BP = 1e-4


def bump_rate(snapshot: MarketSnapshot, dr: float) -> MarketSnapshot:
    """Parallel shift of the (flat) discount curve by `dr`, as a new snapshot.

    ponytail: single-rate parallel shift for the flat curve. A pillar-wise shift
    of a bootstrapped curve is a later greek slice, behind the same protocol."""
    curve = snapshot.discount_curve
    return replace(snapshot, discount_curve=replace(curve, rate=curve.rate + dr))


def dv01(pricable: Pricable, snapshot: MarketSnapshot, numerics: NumericalConfig) -> float:
    """PV change per 1bp parallel rate rise, by central finite difference. The
    bump rebuilds the model inside the `pricable` (risk flows through the model)."""
    h = numerics.fd_bump
    up = pricable(bump_rate(snapshot, h))
    down = pricable(bump_rate(snapshot, -h))
    return (up - down) / (2.0 * h) * _ONE_BP
