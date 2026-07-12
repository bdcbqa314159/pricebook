"""SwapEngine — prices a vanilla single-curve IRS (L4).

Stateless (spine invariants 1-5). The fixed leg is discounted by reusing the
`DiscountingEngine` (it is a `CashflowInstrument`). The float leg's coupons are
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
from pricebook_ng.instruments.swap import VanillaSwap
from pricebook_ng.market.snapshot import MarketSnapshot


class SwapEngine:
    """Prices a vanilla single-curve interest-rate swap."""

    def price(
        self,
        swap: VanillaSwap,
        model: None,
        market: MarketSnapshot,
        numerics: NumericalConfig,
    ) -> PricingResult | PricingFailure:
        fixed = DiscountingEngine().price(swap.fixed_leg, model, market, numerics)
        if isinstance(fixed, PricingFailure):
            return fixed

        leg = swap.float_leg
        if leg.face.currency is not fixed.pv.currency:
            return PricingFailure(
                f"swap legs differ in currency: {leg.face.currency} vs {fixed.pv.currency}"
            )
        dates = leg.schedule
        if dates[0] < market.valuation_date:
            return PricingFailure(
                f"float leg starts {dates[0]} before valuation {market.valuation_date}"
            )

        df = market.discount_curve.df
        float_pv = 0.0
        for a, b in zip(dates[:-1], dates[1:]):
            forward_accrual = df(a) / df(b) - 1.0        # forward * tau, single-curve
            float_pv += leg.face.amount * forward_accrual * df(b)

        fixed_pv = fixed.pv.amount
        npv = (float_pv - fixed_pv) if swap.pay_fixed else (fixed_pv - float_pv)
        return PricingResult(pv=Money(npv, fixed.pv.currency))
