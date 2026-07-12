"""DiscountingEngine — the stateless discounting pricer.

L4, the stateless heart: `price(instrument, model, market, numerics)` binds a
trade to a market and returns a value. It holds no state, mutates nothing, reads
"today" from the snapshot, and returns failure as a value (spine invariants 1-5).
The model is None — discounting needs no dynamics, which proves the engine works
with a null model.

The engine consumes any instrument that presents a `cashflows` leg (structural
`CashflowInstrument` protocol) — it never imports or `isinstance`-checks concrete
instrument classes. It discounts each cashflow on the curve and sums.

Provenance:
  quarry: python/pricebook/pricing/ (discounting logic lifted out of instruments)
  source: redesign/02_spine.md (stateless-engine contract)
  oracle: PV = sum(cf * df) closed form < 1e-12 (S00 single cashflow; S04 bond)
  slice:  S00; S04 (cashflow leg); A1 (engine reads the curve through model.market)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.models.discounting_model import CalibratedModel


@runtime_checkable
class CashflowInstrument(Protocol):
    """Anything the discounting engine can price: it presents a cashflow leg."""

    @property
    def cashflows(self) -> tuple[Cashflow, ...]: ...


class DiscountingEngine:
    """Prices a cashflow leg by discounting each cashflow on the model's curve."""

    def price(
        self,
        instrument: CashflowInstrument,
        model: CalibratedModel,
        numerics: NumericalConfig,
    ) -> PricingResult | PricingFailure:
        market = model.market
        cashflows = instrument.cashflows
        if not cashflows:
            return PricingFailure("instrument has no cashflows")
        currencies = {cf.amount.currency for cf in cashflows}
        if len(currencies) != 1:
            return PricingFailure(f"discounting needs one currency, got {currencies}")

        pv = 0.0
        for cf in cashflows:
            if cf.date < market.valuation_date:
                return PricingFailure(
                    f"cashflow date {cf.date} precedes valuation {market.valuation_date}"
                )
            pv += cf.amount.amount * market.discount_curve.df(cf.date)
        return PricingResult(pv=Money(pv, currencies.pop()))
