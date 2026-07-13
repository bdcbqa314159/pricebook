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
from pricebook_ng.products.swap import VanillaSwap
from pricebook_ng.models.discounting_model import CalibratedModel


class SwapEngine:
    """Prices a vanilla single-curve interest-rate swap."""

    def price(
        self,
        swap: VanillaSwap,
        model: CalibratedModel,
        numerics: NumericalConfig,
    ) -> PricingResult | PricingFailure:
        market = model.market
        fixed = DiscountingEngine().price(swap.fixed_leg, model, numerics)
        if isinstance(fixed, PricingFailure):
            return fixed

        leg = swap.float_leg
        if leg.face.currency is not fixed.pv.currency:
            return PricingFailure(
                f"swap legs differ in currency: {leg.face.currency} vs {fixed.pv.currency}"
            )

        # Temporality (A2 + fixings): a period that already paid (b <= valuation) is
        # settled; a period whose reset is strictly past (a < valuation) uses the
        # realized fixing; a future period projects the curve forward.
        valuation = market.valuation_date
        df = market.discount_curve.df
        float_pv = 0.0
        for a, b in zip(leg.schedule[:-1], leg.schedule[1:]):
            if b <= valuation:
                continue
            if a < valuation:
                rate = market.fixings.get(a)
                if rate is None:
                    return PricingFailure(f"missing float fixing for reset {a}")
                accrual = rate * year_fraction(a, b, leg.day_count)
            else:
                accrual = df(a) / df(b) - 1.0            # forward * tau, single-curve
            float_pv += leg.face.amount * accrual * df(b)

        fixed_pv = fixed.pv.amount
        npv = (float_pv - fixed_pv) if swap.pay_fixed else (fixed_pv - float_pv)
        return PricingResult(pv=Money(npv, fixed.pv.currency))
