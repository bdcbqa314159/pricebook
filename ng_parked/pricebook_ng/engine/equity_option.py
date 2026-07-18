"""EquityOptionEngine — European equity option by Black-Scholes (L4).

A thin binding of the shared `price_spot_option` engine to `AssetClass.EQUITY`
(carry = dividend/repo curve). See `engine/spot_option.py` for the math.

Provenance:
  quarry: python/pricebook/pricing/ (equity option)
  source: Black-Scholes-Merton (1973); Black (1976)
  oracle: put-call parity vs the equity forward; Black recompute (equity-option slice)
  slice:  equity-option; commodity-option (onto shared spot_option)
"""

from __future__ import annotations

from pricebook_ng.engine.spot_option import price_spot_option
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.market.keys import AssetClass
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.equity_option import EquityOption


class EquityOptionEngine:
    """Values a European equity option via Black-Scholes (dividends)."""

    def price(
        self, option: EquityOption, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        return price_spot_option(option, AssetClass.EQUITY, model, numerics)
