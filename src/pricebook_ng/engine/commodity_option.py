"""CommodityOptionEngine — European commodity option by Black-Scholes (L4).

A thin binding of the shared `price_spot_option` engine to `AssetClass.CMDTY`
(carry = convenience yield net of storage). See `engine/spot_option.py`.

Provenance:
  quarry: python/pricebook/pricing/ (commodity option)
  source: Black-Scholes on the commodity forward (carry)
  oracle: put-call parity vs the commodity forward; Black recompute (commodity-option slice)
  slice:  commodity-option
"""

from __future__ import annotations

from pricebook_ng.engine.spot_option import price_spot_option
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.market.keys import AssetClass
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.commodity_option import CommodityOption


class CommodityOptionEngine:
    """Values a European commodity option via Black-Scholes (carry)."""

    def price(
        self, option: CommodityOption, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        return price_spot_option(option, AssetClass.CMDTY, model, numerics)
