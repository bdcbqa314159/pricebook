"""Linear pricing engine — bind a swap + a model to a mark (L4).

`price(swap, model)` composes the SAME `rpv01`/`float_leg_pv` building blocks the
calibrator uses (§3d) — no second annuity or forward loop — reaching the curves only
through `model.market` (A1). It resolves the discount curve from the swap's currency and
the projection curve from the float leg's index, both via the `CurveSet`; single-curve is
the degenerate case where those are the same curve. PV is the fixed-rate PAYER's:
`N·(float − rate·annuity)`, zero exactly when the swap is at par. Failure is a value
(invariant 4): a cashflow beyond a curve's pillars, or an unresolved key, returns a
`PricingFailure`, never a raise. No `numerics` argument — linear pricing reads no
reproducibility knob; a `NumericalConfig` lands with the first engine that needs one.

Provenance:
  quarry: python/pricebook/pricing/ (swap engine)
  source: CLAUDE.md §2 (stateless engine, A1) · §3d (shared atoms); redesign/22 Q3
  oracle: EURIBOR swap → zero NPV dual-curve to 1e-9; DV01 analytic vs finite-diff to 1e-6
  slice:  dual-curve-euribor-estr (T1 slice 2)
"""

from __future__ import annotations

from pricebook_ng.foundation import Money, PricingFailure, PricingResult
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import VanillaSwap


def price(swap: VanillaSwap, model: DiscountingModel) -> PricingResult | PricingFailure:
    """Mark a vanilla swap off the model's curves. Payer PV = N·(float − rate·annuity)."""
    curves = model.market.curves
    try:
        discount = curves.discount(swap.currency)
        projection = curves.projection(swap.float_leg.index)
        annuity = rpv01(swap.fixed_leg.schedule, swap.fixed_leg.day_count, discount)
        floating = float_leg_pv(swap.float_leg.schedule, swap.float_leg.day_count, discount, projection)
    except (ValueError, KeyError) as exc:  # cashflow beyond a curve, or an unresolved curve key
        return PricingFailure(str(exc))
    pv = swap.notional * (floating - swap.fixed_leg.rate * annuity)
    return PricingResult(pv=Money(pv, swap.currency))
