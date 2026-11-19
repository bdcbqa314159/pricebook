"""The European commodity option engine (L4) - the carry forward, then verified Black.

Deliberately thin, and the thinness is the design. The commodity content of a vanilla is **the
forward**: once `F(T)` carries the cost of carry and the convenience yield, the price is Black-76's,
and `models/black.py` has been Black-76-on-a-forward since tranche 5. An engine that re-derived the
option formula with a convenience yield folded inside would be a second transcription of something
already proved, with its own chance of being wrong - so reducing to the existing function is both
the simpler design and the stronger oracle. `engine/equity_vanilla.py` made the same call at
tranche 30 and is the precedent this follows.

Provenance:
  stock:  none - the archived predecessor had no commodity option engine
  source: CLARK-2014 (docs/references.md) ch 2.5 (European options on spot and futures) and 2.2.4
          (Black-76); the forward is ch 2.2.6 eq (2.56), built in market/commodity.py
  oracle: tests/L4/test_commodity_vanilla.py - the price is EXACTLY `df(T) * N * black_price` on the
          same forward (the reduction proving this tranche does not perturb Black), with a companion
          that the total variance carries its time factor; put-call parity swept OFF the money as the
          model-free requirement oracle; the Clark-vs-black.py agreement kept as a CONSISTENCY GUARD
          and labelled as one, since both render the same Black-76 and it is not cross-source
          independence (the T17 lesson); invariant 6 with an IN-THE-MONEY expired fixture (F-4.1's
          sub-shape); missing commodity data and a missing curve are PricingFailures
  crosscheck: quantlib AnalyticEuropeanEngine / BlackCalculator on a commodity forward - deferred to
          the optional QuantLib tier
  slice:  tranche 60 - the commodity forward and option; the commodity arc opens
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.commodity import commodity_forward
from pricebook.models.black import black_price
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.products.commodity_option import CommodityOption


@register(CommodityOption, CommodityBlackModel)
def price_commodity_option(
    product: CommodityOption, model: CommodityBlackModel
) -> PricingResult | PricingFailure:
    """`df(T) * N * Black(F(T), K, sigma^2 T, omega)`."""
    market = model.market
    currency = product.notional.currency

    # Invariant 6: an expired option is the shell's business, PV 0, and checked before any market
    # lookup so a settled trade needs no commodity data at all.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    underlying = market.commodities.get(product.asset)
    if underlying is None:
        return PricingFailure(
            f"no commodity data for {product.asset.symbol} in the snapshot"
        )
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the commodity option: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    forward = commodity_forward(underlying, product.expiry, discount)
    total_variance = model.volatility * model.volatility * years
    undiscounted = black_price(forward, product.strike, total_variance, product.kind)
    return PricingResult(
        pv=Money(discount.df(product.expiry) * product.notional.amount * undiscounted, currency)
    )
