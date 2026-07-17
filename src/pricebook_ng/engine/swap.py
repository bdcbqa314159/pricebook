"""SwapEngine — prices a vanilla single-curve IRS (L4).

Stateless (spine invariants 1-5). The fixed leg is discounted by reusing the
`DiscountingEngine` (it is a `CashflowProduct`). The float leg's coupons are
the curve's forwards: over period (a, b) the forward accrual is
`DF(a)/DF(b) - 1`, and discounting that coupon by `DF(b)` gives `DF(a) - DF(b)`
per unit notional — so the float leg telescopes to `notional*(DF(t0) - DF(tn))`.
Written per-period so the forward structure is explicit (single-curve: the
projection curve is the discount curve).

NPV is to the swap holder: a payer swap (pays fixed, receives float) is
`float_pv - fixed_pv`; a receiver swap is the negative.

Provenance:
  quarry: python/pricebook/pricing/ (swap pricing)
  source: standard single-curve vanilla IRS valuation
  oracle: par swap reprices to zero NPV; float leg telescopes (S06)
  slice:  S06
"""

from __future__ import annotations

from pricebook_ng.engine.discounting import DiscountingEngine
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.market.snapshot import MarketSnapshot
from pricebook_ng.products.swap import FloatLeg, VanillaSwap
from pricebook_ng.models.discounting_model import CalibratedModel


def float_leg_pv(leg: FloatLeg, market: MarketSnapshot) -> float | PricingFailure:
    """PV of a single-curve float leg. Each future period accrues the curve's
    simply-compounded forward `L(a,b)·tau = DF(a)/DF(b)-1` (via `curve.forward_rate`); a
    seasoned reset (a < valuation) uses the realized fixing; a settled period (b <= valuation)
    contributes nothing (A2). Shared by the vanilla-IRS and OIS engines — single-curve, the OIS
    compounded overnight rate equals the same forward."""
    valuation = market.valuation_date
    curve = market.discount_curve
    pv = 0.0
    for a, b in zip(leg.schedule[:-1], leg.schedule[1:]):
        if b <= valuation:
            continue
        tau = year_fraction(a, b, leg.day_count)
        if a < valuation:
            rate = market.fixings.get(a)
            if rate is None:
                return PricingFailure(f"missing float fixing for reset {a}")
            accrual = rate * tau
        else:
            accrual = curve.forward_rate(a, b, leg.day_count) * tau  # = DF(a)/DF(b) - 1
        pv += leg.face.amount * accrual * curve.df(b)
    return pv


class SwapEngine:
    """Prices a vanilla single-curve interest-rate swap."""

    def price(
        self,
        swap: VanillaSwap,
        model: CalibratedModel,
        numerics: NumericalConfig,
    ) -> PricingResult | PricingFailure:
        fixed = DiscountingEngine().price(swap.fixed_leg, model, numerics)
        if isinstance(fixed, PricingFailure):
            return fixed
        if swap.float_leg.face.currency is not fixed.pv.currency:
            return PricingFailure(
                f"swap legs differ in currency: {swap.float_leg.face.currency} vs {fixed.pv.currency}"
            )
        float_pv = float_leg_pv(swap.float_leg, model.market)
        if isinstance(float_pv, PricingFailure):
            return float_pv

        npv = (float_pv - fixed.pv.amount) if swap.pay_fixed else (fixed.pv.amount - float_pv)
        return PricingResult(pv=Money(npv, fixed.pv.currency))
