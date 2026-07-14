"""ZCISEngine — zero-coupon inflation swap by no-arbitrage (L4).

Stateless. The inflation forward index ratio is `I(T) = DF_real(T) / DF_r(T)`
(Fisher relation), with the real curve read from the snapshot at
`MarketKey(INFLATION, index)` and the nominal home curve as `discount_curve`. The
receiver (of inflation) value per unit notional is
`DF_r(T) * (I(T) - (1+K)^T)`; the fixed payer is the negative.

Provenance:
  quarry: python/pricebook/pricing/ (inflation)
  source: zero-coupon inflation swap; Fisher relation I = DF_real / DF_nominal
  oracle: par ZCIS reprices to zero (inflation-zcis slice)
  slice:  inflation-zcis
"""

from __future__ import annotations

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import DayCountConvention, year_fraction
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.inflation import ZeroCouponInflationSwap

_CURVE_DC = DayCountConvention.ACT_365_FIXED


class ZCISEngine:
    """Values a zero-coupon inflation swap in its currency."""

    def price(
        self, zcis: ZeroCouponInflationSwap, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        market = model.market
        if zcis.maturity <= market.valuation_date:
            return PricingResult(pv=Money(0.0, zcis.currency))  # settled

        real_curve = market.curves.get(MarketKey(AssetClass.INFLATION, zcis.index))
        if real_curve is None:
            return PricingFailure(f"no inflation (real) curve for index {zcis.index!r}")

        df_r = market.discount_curve.df(zcis.maturity)
        index_ratio = real_curve.df(zcis.maturity) / df_r        # I(T) = DF_real / DF_r
        t = year_fraction(market.valuation_date, zcis.maturity, _CURVE_DC)
        fixed_growth = (1.0 + zcis.fixed_rate) ** t

        receiver = df_r * (index_ratio - fixed_growth)           # per unit notional
        pv = receiver if zcis.receive_inflation else -receiver
        return PricingResult(pv=Money(zcis.notional * pv, zcis.currency))
