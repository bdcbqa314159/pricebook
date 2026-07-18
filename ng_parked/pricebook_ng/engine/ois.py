"""OISEngine — single-curve overnight index swap pricer (L4).

Stateless. Reuses the shared `float_leg_pv` for the compounded-overnight float leg (single-
curve: the compounded rate equals the curve forward) and the `DiscountingEngine` for the
fixed leg — so OIS reprices identically to the vanilla IRS in single-curve. NPV to the
holder: payer (pay fixed, receive float) is `float - fixed`.

Provenance:
  quarry: python/pricebook/fixed_income/ois.py
  source: standard single-curve OIS valuation
  oracle: par OIS -> 0; OIS == vanilla IRS (single-curve)
  slice:  ois-spine (CP-2c #4)
"""

from __future__ import annotations

from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.engine.swap import float_leg_pv
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.ois import OvernightIndexSwap


class OISEngine:
    """Prices a single-curve overnight index swap (fixed vs compounded overnight)."""

    def price(
        self, ois: OvernightIndexSwap, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        fixed = DiscountingEngine().price(ois.fixed_leg, model, numerics)
        if isinstance(fixed, PricingFailure):
            return fixed
        if ois.float_leg.face.currency is not fixed.pv.currency:
            return PricingFailure(
                f"OIS legs differ in currency: {ois.float_leg.face.currency} vs {fixed.pv.currency}"
            )
        float_pv = float_leg_pv(ois.float_leg, model.market)
        if isinstance(float_pv, PricingFailure):
            return float_pv

        npv = (float_pv - fixed.pv.amount) if ois.pay_fixed else (fixed.pv.amount - float_pv)
        return PricingResult(pv=Money(npv, fixed.pv.currency))
