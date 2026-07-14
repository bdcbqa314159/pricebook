"""EquityOptionEngine — European equity option by Black-Scholes (L4).

Stateless. Black-Scholes is `black_76` on the equity forward
`F = spot * DF_div(T) / DF_r(T)` (continuous dividend/repo curve `DF_div`,
discount curve `DF_r`), discounted by `DF_r`. Spot, dividend curve, and vol are
read from the snapshot by ticker (§5.1).

ponytail: the equity is assumed to trade in the snapshot's home currency
(`discount_curve`); a foreign-listed equity (its own currency curve) is a later
refinement.

Provenance:
  quarry: python/pricebook/pricing/ (equity option)
  source: Black-Scholes-Merton (1973); Black (1976)
  oracle: put-call parity vs the equity forward; Black recompute (equity-option slice)
  slice:  equity-option
"""

from __future__ import annotations

import math

from pricebook_ng.foundation.black import black_76
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.equity_option import EquityOption

_CURVE_DC = DayCountConvention.ACT_365_FIXED


class EquityOptionEngine:
    """Values a European equity option via Black-Scholes (dividends)."""

    def price(
        self, option: EquityOption, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        market = model.market
        ticker = option.ticker
        if option.maturity <= market.valuation_date:
            return PricingResult(pv=Money(0.0, option.currency))  # expired

        spot = market.equity_spots.get(ticker)
        div_curve = market.equity_div_curves.get(ticker)
        vol = market.equity_vols.get(ticker)
        if spot is None or div_curve is None or vol is None:
            return PricingFailure(f"no equity market (spot/div/vol) for {ticker!r}")

        df_r = market.discount_curve.df(option.maturity)
        fwd = spot * div_curve.df(option.maturity) / df_r
        t = year_fraction(market.valuation_date, option.maturity, _CURVE_DC)
        per_share = black_76(fwd, option.strike, df_r, vol * math.sqrt(t), option.is_call)
        return PricingResult(pv=Money(option.quantity * per_share, option.currency))
