"""Linear pricing engine — bind a swap + a model to a mark (L4).

`price(swap, model)` composes the SAME `rpv01`/`float_leg_pv` building blocks the
calibrator uses (§3d) — no second annuity loop — reaching the curve only through
`model.market` (A1). PV is the fixed-rate PAYER's: `N·(float − rate·annuity)`; it is
zero exactly when the swap is at par. Failure is a value (CLAUDE.md §2, invariant 4):
a cashflow beyond the curve's pillars returns a `PricingFailure`, never a raise. No
`numerics` argument — linear pricing reads no reproducibility knob; a `NumericalConfig`
lands with the first engine that needs one (MC/PDE), decomposed by method family.

Provenance:
  quarry: python/pricebook/pricing/ (swap engine)
  source: CLAUDE.md §2 (stateless engine, A1) · §3d (shared atoms); redesign/22 Q3
  oracle: par swap → zero NPV to 1e-9; DV01 analytic vs finite-diff to 1e-6
  slice:  swap-to-zero-npv (T1 slice 1)
"""

from __future__ import annotations

from pricebook_ng.foundation import Money, PricingFailure, PricingResult
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import VanillaSwap


def price(swap: VanillaSwap, model: DiscountingModel) -> PricingResult | PricingFailure:
    """Mark a vanilla swap off the model's curve. Payer PV = N·(float − rate·annuity)."""
    curve = model.market.discount_curve
    try:
        annuity = rpv01(swap.fixed_leg.schedule, swap.fixed_leg.day_count, curve)
        floating = float_leg_pv(swap.float_leg.schedule, curve)
    except ValueError as exc:  # a cashflow beyond the curve's pillars — failure is a value
        return PricingFailure(str(exc))
    pv = swap.notional * (floating - swap.fixed_leg.rate * annuity)
    return PricingResult(pv=Money(pv, swap.currency))
