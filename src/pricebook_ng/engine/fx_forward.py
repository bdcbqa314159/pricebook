"""FXForwardEngine — prices an FX forward by covered interest parity (L4).

Stateless. Values both legs in the quote currency off the two curves and spot:
`base` is worth `base_notional * spot * DF_base(T)`; `quote` is worth
`quote_notional * DF_quote(T)`. The buyer's PV is base leg minus quote leg. A
forward whose exchange date has passed is settled (PV 0, A2).

Provenance:
  quarry: python/pricebook/pricing/ (fx forward)
  source: covered interest parity
  oracle: par FX forward reprices to zero; PV matches CIP (fx-forward slice)
  slice:  fx-forward
"""

from __future__ import annotations

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingResult
from pricebook_ng.models.fx_model import FXForwardModel
from pricebook_ng.products.fx_forward import FXForward


class FXForwardEngine:
    """Values an FX forward in the quote currency via covered interest parity."""

    def price(
        self, fx: FXForward, model: FXForwardModel, numerics: NumericalConfig
    ) -> PricingResult:
        quote_ccy = fx.quote.currency
        if fx.maturity <= model.market.valuation_date:
            return PricingResult(pv=Money(0.0, quote_ccy))  # exchange settled (A2)

        base_leg = fx.base.amount * model.spot * model.base_curve.df(fx.maturity)
        quote_leg = fx.quote.amount * model.market.discount_curve.df(fx.maturity)
        pv = (base_leg - quote_leg) if fx.buy_base else (quote_leg - base_leg)
        return PricingResult(pv=Money(pv, quote_ccy))
