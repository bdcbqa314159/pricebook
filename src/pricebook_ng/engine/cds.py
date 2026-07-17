"""CDSEngine — prices a single-name CDS under a CreditModel (L4).

Stateless. Reads the discount curve, survival curve, and recovery through the
model (Amendment A1), and values the protection buyer with the L1 CDS leg math
(`cds_pv` = protection PV - spread * RPV01). The seller is the negative.

Provenance:
  quarry: python/pricebook/pricing/ (CDS)
  source: standard single-name CDS valuation
  oracle: par CDS reprices to zero; matches L1 cds_pv (cds-product slice)
  slice:  cds-product
"""

from __future__ import annotations

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.market.keys import AssetClass, MarketKey
from pricebook_ng.market.survival_curve import cds_pv
from pricebook_ng.models.credit_model import CreditModel
from pricebook_ng.products.cds import CDS


class CDSEngine:
    """Values a CDS via the survival-curve leg math."""

    def price(
        self, cds: CDS, model: CreditModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        survival = model.market.curves.get(MarketKey(AssetClass.CREDIT, cds.issuer))
        if survival is None:
            return PricingFailure(f"no survival curve for issuer {cds.issuer!r}")
        buyer_pv = cds_pv(
            model.market.discount_curve,
            survival,
            list(cds.premium_schedule),
            cds.spread,
            model.recovery,
        )
        signed = buyer_pv if cds.buy_protection else -buyer_pv
        return PricingResult(pv=Money(signed * cds.face.amount, cds.face.currency))
