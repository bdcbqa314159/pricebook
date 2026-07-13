"""Credit greeks — CDS credit01 / CS01 by hazard bump-and-reprice (L5).

The credit analogue of `greeks.dv01`: where dv01 bumps the discount rate through
the snapshot, credit01 bumps the survival curve's hazard, rebuilds the
`CreditModel`, and reprices. A parallel hazard shift `dh` scales each survival
pillar by `exp(-dh * t)` (piecewise-constant hazard -> `Q(t) -> Q(t) e^{-dh t}`).

ponytail: CDS is the only credit product today, so credit01 calls `CDSEngine`
directly. Route it through a `survival -> PV` closure (like the `Pricable`
factories) when a second hazard-sensitive product arrives.

Provenance:
  quarry: python/pricebook/risk/ (credit greeks)
  source: redesign/02_spine.md (risk at L5, bump-and-reprice)
  oracle: buyer credit01 > 0; seller = -buyer; matches an independent hazard FD
  slice:  cds-credit01
"""

from __future__ import annotations

import math
from dataclasses import replace

from pricebook_ng.engine.cds import CDSEngine
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.survival_curve import SurvivalCurve
from pricebook_ng.models.credit_model import CreditModel
from pricebook_ng.products.cds import CDS

_ONE_BP = 1e-4
_CURVE_DC = DayCountConvention.ACT_365_FIXED


def bump_hazard(survival: SurvivalCurve, dh: float) -> SurvivalCurve:
    """Parallel hazard shift by `dh`: scale each survival pillar by exp(-dh*t)."""
    v = survival.valuation_date
    pillars = tuple(
        (d, q * math.exp(-dh * year_fraction(v, d, _CURVE_DC))) for d, q in survival.pillars
    )
    return SurvivalCurve(v, pillars)


def credit01(cds: CDS, model: CreditModel, numerics: NumericalConfig) -> float:
    """PV change per 1bp parallel credit-spread (hazard) widening, central FD."""
    h = numerics.fd_bump
    up = CDSEngine().price(cds, replace(model, survival=bump_hazard(model.survival, h)), numerics)
    down = CDSEngine().price(cds, replace(model, survival=bump_hazard(model.survival, -h)), numerics)
    return (up.pv.amount - down.pv.amount) / (2.0 * h) * _ONE_BP
