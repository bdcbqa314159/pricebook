"""SwaptionMCEngine — HW European swaption by Monte Carlo (L4).

The independent cross-check on the S08 Jamshidian analytic. Prices under the
T0-forward measure, where the swaption value is
`P(0,T0) * E^{T0}[ payoff(T0) ]` and the short rate at expiry needs only ONE
exact Gaussian draw:

    x(T0) ~ Normal(M, V),   V = sigma^2 (1 - e^{-2 a T0}) / (2a)
    M = -(sigma^2/a^2) [ (1 - e^{-a T0}) - (1 - e^{-2 a T0})/2 ]   (T0-forward mean)
    r(T0) = x(T0) + alpha(T0),  alpha(T0) = r0 + (sigma^2/2a^2)(1 - e^{-a T0})^2

The coupon-bond value at expiry is reconstituted with the model's `zero_bond`;
the payer/receiver payoff is `max(±(notional - couponbond), 0)`. Reproducible via
`NumericalConfig.mc_seed`; stdlib `random` (no numpy dependency).

Provenance:
  quarry: python/pricebook/numerical/_mc.py; pricing/ (swaption MC)
  source: Brigo & Mercurio s.3.3; T-forward measure simulation
  oracle: MC converges to the Jamshidian analytic within a few standard errors (S09)
  slice:  swaption-mc
"""

from __future__ import annotations

import random

from pricebook_ng.engine.swaption import coupon_bond_cashflows
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.models.hull_white import HullWhite
from pricebook_ng.products.swaption import Swaption


class SwaptionMCEngine:
    """Prices a European swaption under Hull-White by T0-forward Monte Carlo."""

    def price(
        self, swaption: Swaption, model: HullWhite, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        expiry = swaption.expiry
        market = model.market
        if expiry < market.valuation_date:
            return PricingFailure(f"swaption expiry {expiry} precedes valuation")

        dates, amounts, notional = coupon_bond_cashflows(swaption.swap)
        is_payer = swaption.swap.pay_fixed
        currency = swaption.swap.float_leg.face.currency

        p0_t0 = market.discount_curve.df(expiry)

        rng = random.Random(numerics.mc_seed)
        total = 0.0
        for _ in range(numerics.mc_paths):
            r = model.forward_short_rate(expiry, rng.gauss(0.0, 1.0))
            coupon_bond = sum(amt * model.zero_bond(expiry, d, r) for amt, d in zip(amounts, dates))
            intrinsic = notional - coupon_bond
            total += max(intrinsic, 0.0) if is_payer else max(-intrinsic, 0.0)

        price = p0_t0 * total / numerics.mc_paths
        return PricingResult(pv=Money(price, currency))
