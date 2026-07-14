"""FXOptionEngine — European FX option by Garman-Kohlhagen (L4).

Stateless. GK is Black-76 on the FX forward `F = spot*DF_base/DF_quote`,
discounted by the quote curve. The two curves, spot, and flat vol are all read
from the snapshot (§5.1); the FX forward rate reuses covered interest parity.

    d1 = [ln(F/K) + sigma^2 T / 2] / (sigma sqrt(T)),   d2 = d1 - sigma sqrt(T)
    call = DF_quote (F N(d1) - K N(d2)),  put = DF_quote (K N(-d2) - F N(-d1))
    PV = base_notional * (call or put),  in the quote currency.

Provenance:
  quarry: python/pricebook/pricing/ (fx option)
  source: Garman & Kohlhagen (1983); Black (1976)
  oracle: put-call parity vs the FX forward; Black recompute (fx-option slice)
  slice:  fx-option
"""

from __future__ import annotations

import math

from pricebook_ng.foundation.black import black_76
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.fx_option import FXOption

_CURVE_DC = DayCountConvention.ACT_365_FIXED


class FXOptionEngine:
    """Values a European FX option in the quote currency via Garman-Kohlhagen."""

    def price(
        self, option: FXOption, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        market = model.market
        quote_ccy = option.quote.currency
        base_ccy = option.base.currency
        notional = option.base.amount
        strike = option.quote.amount / notional

        base_curve = market.fx_curves.get(base_ccy)
        spot = market.fx_spots.get(base_ccy)
        vol = market.fx_vols.get(base_ccy)
        if base_curve is None or spot is None or vol is None:
            return PricingFailure(f"no FX market (curve/spot/vol) for {base_ccy}")

        df_quote = market.discount_curve.df(option.maturity)
        fwd = spot * base_curve.df(option.maturity) / df_quote
        t = year_fraction(market.valuation_date, option.maturity, _CURVE_DC)
        std = vol * math.sqrt(t)

        per_base = black_76(fwd, strike, df_quote, std, option.is_call)
        return PricingResult(pv=Money(notional * per_base, quote_ccy))
