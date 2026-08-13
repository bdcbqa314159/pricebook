"""Linear (swap) pricer — registered behind the L4 engine registry.

`price_swap` composes the SAME `rpv01`/`float_leg_pv` building blocks the calibrator uses
(§3d), reaching the curves only through `model.market` (A1). It is registered for
`VanillaSwap` and reached via the registry's `dispatch` (re-exported here as `price`, the
public entry). PV is the fixed-rate PAYER's: `N·(float − rate·annuity)`, zero at par.
Failure is a value (invariant 4).

Provenance:
  quarry: python/pricebook/pricing/ (swap engine)
  source: CLAUDE.md §2 (stateless engine, A1) · §3d (shared atoms); §1 (registry dispatch)
  oracle: EURIBOR swap → zero NPV dual-curve to 1e-9; DV01 analytic vs finite-diff to 1e-6
  slice:  cash-instruments (T1 slice 3, Step A — swap pricer moved behind the registry, body unchanged)
"""

from __future__ import annotations

from pricebook_ng.engine.registry import dispatch as price
from pricebook_ng.engine.registry import register
from pricebook_ng.foundation import Money, PricingFailure, PricingResult
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.products.swap import VanillaSwap

__all__ = ["price", "price_swap"]


@register(VanillaSwap)
def price_swap(swap: VanillaSwap, model: CalibratedModel) -> PricingResult | PricingFailure:
    """Mark a vanilla swap off the model's curves. Payer PV = N·(float − rate·annuity)."""
    curves = model.market.curves
    try:
        discount = curves.discount(swap.currency, swap.collateral)  # CSA-keyed (A1: through the model)
        projection = curves.projection(swap.float_leg.index)
        annuity = rpv01(swap.fixed_leg.schedule, swap.fixed_leg.day_count, discount)
        floating = float_leg_pv(swap.float_leg.schedule, swap.float_leg.day_count, discount, projection)
    except (ValueError, KeyError) as exc:  # cashflow beyond a curve, or an unresolved curve key
        return PricingFailure(str(exc))
    pv = swap.notional * (floating - swap.fixed_leg.rate * annuity)
    basis = None if swap.collateral in (None, swap.currency) else swap.collateral
    return PricingResult(pv=Money(pv, swap.currency), basis=basis)
