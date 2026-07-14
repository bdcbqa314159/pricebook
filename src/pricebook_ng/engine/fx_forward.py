"""FXForwardEngine — prices an FX forward by covered interest parity (L4).

Stateless. The FX forward is linear, priced with a `DiscountingModel` over a
snapshot that carries the FX market data (§5.1): the home/quote curve is
`market.discount_curve`; the base-currency curve and spot are looked up by the
product's base currency in `market.fx_curves` / `market.fx_spots`.

Both legs are valued in the quote currency: `base` is worth
`base_notional * spot * DF_base(T)`, `quote` is worth `quote_notional * DF_quote(T)`;
the buyer's PV is base leg minus quote leg. A settled (past) exchange prices to 0.

Provenance:
  quarry: python/pricebook/pricing/ (fx forward); core/currency.py
  source: covered interest parity
  oracle: par FX forward reprices to zero; PV matches CIP (fx-forward slice)
  slice:  fx-forward; fx-in-snapshot (§5.1 promotion)
"""

from __future__ import annotations

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.fx_forward import FXForward


class FXForwardEngine:
    """Values an FX forward in the quote currency via covered interest parity."""

    def price(
        self, fx: FXForward, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        market = model.market
        quote_ccy = fx.quote.currency
        base_ccy = fx.base.currency
        if fx.maturity <= market.valuation_date:
            return PricingResult(pv=Money(0.0, quote_ccy))  # exchange settled (A2)

        key = MarketKey(AssetClass.FX, base_ccy.value)
        base_curve = market.curves.get(key)
        spot = market.spots.get(key)
        if base_curve is None or spot is None:
            return PricingFailure(f"no FX market (curve/spot) for {base_ccy}")

        base_leg = fx.base.amount * spot * base_curve.df(fx.maturity)
        quote_leg = fx.quote.amount * market.discount_curve.df(fx.maturity)
        pv = (base_leg - quote_leg) if fx.buy_base else (quote_leg - base_leg)
        return PricingResult(pv=Money(pv, quote_ccy))
