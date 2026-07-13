"""Greeks by bump-and-reprice on the Pricable protocol (L5).

Generic over the product/model: a greek central-differences a `Pricable` under a
bump of the snapshot. `dv01` bumps the discount rate; `credit01` bumps the
survival (hazard) curve — both market data on the snapshot (ruling §5.1), so the
same `Pricable` and the same finite-difference core serve rate risk and credit
risk. The bump rebuilds the model inside the pricable (Amendment A1).

Provenance:
  quarry: python/pricebook/risk/ (bump-and-reprice greeks; ex-L3)
  source: redesign/02_spine.md (risk at L5, bump the snapshot re-run the engine)
  oracle: dv01 matches analytic (cashflow, bond); credit01 buyer>0, seller=-buyer
  slice:  S00 (dv01); L5-risk (onto Pricable); survival-in-snapshot (credit01 unified)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace

from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.market.survival_curve import SurvivalCurve
from pricebook_ng.risk.pricable import Pricable

_ONE_BP = 1e-4
_CURVE_DC = DayCountConvention.ACT_365_FIXED
_Bump = Callable[[MarketSnapshot, float], MarketSnapshot]


def bump_rate(snapshot: MarketSnapshot, dr: float) -> MarketSnapshot:
    """Parallel shift of the (flat) discount curve by `dr`, as a new snapshot.

    ponytail: single-rate parallel shift for the flat curve. A pillar-wise shift
    of a bootstrapped curve is a later greek slice, behind the same protocol."""
    curve = snapshot.discount_curve
    return replace(snapshot, discount_curve=replace(curve, rate=curve.rate + dr))


def bump_hazard(snapshot: MarketSnapshot, dh: float) -> MarketSnapshot:
    """Parallel hazard shift by `dh`: scale each survival pillar by exp(-dh*t),
    as a new snapshot (only the credit curve moves)."""
    survival = snapshot.survival_curve
    assert isinstance(survival, SurvivalCurve)
    v = survival.valuation_date
    pillars = tuple(
        (d, q * math.exp(-dh * year_fraction(v, d, _CURVE_DC))) for d, q in survival.pillars
    )
    return replace(snapshot, survival_curve=replace(survival, pillars=pillars))


def _bp_sensitivity(
    pricable: Pricable, snapshot: MarketSnapshot, bump: _Bump, numerics: NumericalConfig
) -> float:
    """PV change per 1bp, by central finite difference under `bump`."""
    h = numerics.fd_bump
    up = pricable(bump(snapshot, h))
    down = pricable(bump(snapshot, -h))
    return (up - down) / (2.0 * h) * _ONE_BP


def dv01(pricable: Pricable, snapshot: MarketSnapshot, numerics: NumericalConfig) -> float:
    """PV change per 1bp parallel rate rise."""
    return _bp_sensitivity(pricable, snapshot, bump_rate, numerics)


def credit01(pricable: Pricable, snapshot: MarketSnapshot, numerics: NumericalConfig) -> float:
    """PV change per 1bp parallel credit-spread (hazard) widening (CS01)."""
    return _bp_sensitivity(pricable, snapshot, bump_hazard, numerics)
