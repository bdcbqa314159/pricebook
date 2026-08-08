"""Cross-currency basis swap pricer — registered behind the L4 registry.

`price_xccy` composes the SAME `float_leg_pv`/`rpv01` atoms `XccyBasisInstrument` composes (§3d),
so the L3 residual and the L4 mark cannot drift: the foreign leg discounted on the foreign-collateral
curve (`discount(foreign_ccy, collateral)`, 6b keying), projected on the foreign OIS, plus the basis
annuity, plus the foreign notional exchange. The domestic leg and FX spot cancel (see `XccyBasisSwap`),
so neither enters the mark. Market is reached only through `model.market` (A1); failure is a value.

Provenance:
  quarry: python/pricebook/fx/ (xccy basis — reassigned to the FX topic; ng builds the minimal curve)
  source: CLAUDE.md §2 (A1) · §3d (shared atoms); doc 18 §1/§8 (xccy → foreign-collateral curve)
  oracle: each calibrating xccy basis swap reprices to zero; CIP F=S·df_f/df_d at zero basis (1e-10)
  slice:  xccy-basis (T1 slice 6c, C1 close)
"""

from __future__ import annotations

from pricebook_ng.engine.registry import register
from pricebook_ng.foundation import Money, PricingFailure, PricingResult
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.models.discounting_model import DiscountingModel
from pricebook_ng.products.swap import XccyBasisSwap

__all__ = ["price_xccy"]


@register(XccyBasisSwap)
def price_xccy(swap: XccyBasisSwap, model: DiscountingModel) -> PricingResult | PricingFailure:
    """Mark a constant-notional xccy basis swap; zero at par. PV (in the foreign currency) =
    N·(foreign_leg + basis·annuity + df_coll(T) − 1), discounted on the foreign-collateral curve."""
    leg = swap.foreign_leg
    foreign_ccy = leg.index.id.currency
    curves = model.market.curves
    try:
        collateral = curves.discount(foreign_ccy, swap.collateral)  # df_foreign^collateral (A1)
        foreign_ois = curves.projection(leg.index)  # projects the foreign floating
        floating = float_leg_pv(leg.schedule, leg.day_count, collateral, foreign_ois)
        annuity = rpv01(leg.schedule, leg.day_count, collateral)
        maturity = leg.schedule.periods[-1].payment_date
        notional_exchange = collateral.df(maturity) - 1.0
    except (ValueError, KeyError) as exc:
        return PricingFailure(str(exc))
    pv = swap.notional * (floating + swap.basis * annuity + notional_exchange)
    return PricingResult(pv=Money(pv, foreign_ccy), basis=swap.collateral)
