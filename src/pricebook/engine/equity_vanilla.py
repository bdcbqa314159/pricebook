"""The European equity option engine (L4) - the dividend forward, then verified Black.

Deliberately thin. The equity content of a vanilla is **the forward**, not the option formula: once
`F(T)` carries the dividends and the repo, the price is Black's, and Black has been verified since
tranche 5. An engine that re-derived the option formula with dividends folded inside would be a
second transcription of something already proved, with its own chance of being wrong - so the
reduction to the existing function is both the simpler design and the stronger oracle.

Provenance:
  stock:  none - the archived predecessor's equity engine was never purified into this tree
  source: BROCKHAUS-2016 (docs/references.md) ch 5 - the European equity option priced off the
          dividend forward
  oracle: tests/L4/test_equity_vanilla.py - with no dividends and no repo the price is
          **bit-identically** `black_price` on `S0/df(T)` (so this tranche provably does not perturb
          Black), with the companion that a material dividend stream moves it; put-call parity
          (model-free, so it survives a transcription error in the forward); monotone in vol and
          inside the no-arbitrage bounds
  crosscheck: quantlib AnalyticEuropeanEngine on a BlackScholesMertonProcess - deferred to the
          optional QuantLib tier
  slice:  tranche 30 - the equity underlying: the equity & hybrids arc opens
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.market.equity import dividend_forward
from pricebook.models.black import black_price
from pricebook.models.equity_black import EquityBlackModel
from pricebook.products.equity_option import EuropeanEquityOption


@register(EuropeanEquityOption)
def price_european_equity_option(
    product: EuropeanEquityOption, model: EquityBlackModel
) -> PricingResult | PricingFailure:
    """`df(T) * N * Black(F(T), K, sigma^2 T, omega)`."""
    market = model.market
    currency = product.notional.currency

    # Invariant 6: an expired option is the shell's business, PV 0, and checked before any market
    # lookup so a settled trade needs no equity data at all.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    underlying = market.equities.get(product.asset)
    if underlying is None:
        return PricingFailure(
            f"no equity data for {product.asset.ticker} in the snapshot"
        )
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the equity option: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    forward = dividend_forward(underlying, product.expiry, discount)
    total_variance = model.volatility * model.volatility * years
    undiscounted = black_price(forward, product.strike, total_variance, product.kind)
    return PricingResult(pv=Money(discount.df(product.expiry) * product.notional.amount * undiscounted, currency))
