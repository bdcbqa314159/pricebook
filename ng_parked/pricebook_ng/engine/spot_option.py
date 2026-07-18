"""Black-Scholes on a spot underlying with a carry curve — shared engine (L4).

The one pricer behind the equity option (carry = dividend/repo curve) and the
commodity option (carry = convenience yield net of storage). Both are `black_76`
on the forward `F = spot * DF_carry(T) / DF_r(T)`, discounted by the home
`DF_r`. Spot, carry curve, and vol are read from the snapshot's keyed registry
(A5) under `MarketKey(asset, option.ticker)`.

The `option` is duck-typed: it exposes `ticker`, `quantity`, `strike` (a `Money`
strike price = amount + currency), `maturity`, `is_call`. The underlying trades in home currency
(`discount_curve`); a foreign-listed underlying is a later refinement.

Provenance:
  quarry: python/pricebook/pricing/ (equity/commodity option)
  source: Black-Scholes-Merton (1973); Black (1976)
  oracle: put-call parity vs the forward; Black recompute (equity + commodity slices)
  slice:  commodity-option (factored from equity-option, rule of two)
"""

from __future__ import annotations

import math

from pricebook_ng.foundation.black import black_76
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.models.discounting_model import CalibratedModel

_CURVE_DC = DayCountConvention.ACT_365_FIXED


def price_spot_option(
    option, asset: AssetClass, model: CalibratedModel, numerics: NumericalConfig
) -> PricingResult | PricingFailure:
    """Value a European option on a spot underlying (carry curve) via Black-Scholes."""
    market = model.market
    currency = option.strike.currency
    if option.maturity <= market.valuation_date:
        return PricingResult(pv=Money(0.0, currency))  # expired

    key = MarketKey(asset, option.ticker)
    spot = market.spots.get(key)
    carry_curve = market.curves.get(key)
    vol = market.vols.get(key)
    if spot is None or carry_curve is None or vol is None:
        return PricingFailure(f"no {asset.value} market (spot/carry/vol) for {option.ticker!r}")

    df_r = market.discount_curve.df(option.maturity)
    fwd = spot * carry_curve.df(option.maturity) / df_r
    t = year_fraction(market.valuation_date, option.maturity, _CURVE_DC)
    per_unit = black_76(fwd, option.strike.amount, df_r, vol * math.sqrt(t), option.is_call)
    return PricingResult(pv=Money(option.quantity * per_unit, currency))
