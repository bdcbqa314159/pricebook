"""DiscountingEngine — the Slice 0 stateless pricer.

L4, the stateless heart: `price(instrument, model, market, numerics)` binds a
trade to a market and returns a value. It holds no state, mutates nothing, reads
"today" from the snapshot, and returns failure as a value (spine invariants 1-5).
The model is None — discounting needs no dynamics, which proves the engine works
with a null model.

Provenance:
  quarry: python/pricebook/pricing/ (discounting logic lifted out of instruments)
  source: redesign/02_spine.md (stateless-engine contract)
  oracle: PV = notional * df(T), closed form < 1e-12 (Slice 0)
  slice:  S00
"""

from __future__ import annotations

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.instruments.fixed_cashflow import FixedCashflowTrade
from pricebook_ng.market.snapshot import MarketSnapshot


class DiscountingEngine:
    """Prices a FixedCashflowTrade by discounting its cashflow on the curve."""

    def price(
        self,
        trade: FixedCashflowTrade,
        model: None,
        market: MarketSnapshot,
        numerics: NumericalConfig,
    ) -> PricingResult | PricingFailure:
        cf = trade.cashflow
        if cf.date < market.valuation_date:
            return PricingFailure(
                f"cashflow date {cf.date} precedes valuation {market.valuation_date}"
            )
        df = market.discount_curve.df(cf.date)
        pv = cf.amount.amount * df
        return PricingResult(pv=Money(pv, cf.amount.currency))
