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
  slice:  S00; S04 (cashflow leg); A1 (curve via model.market);
          A2 (segment-and-settle + accrued/clean/dirty)
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
        valuation = market.valuation_date
        cashflows = instrument.cashflows
        if not cashflows:
            return PricingFailure("instrument has no cashflows")
        currencies = {cf.amount.currency for cf in cashflows}
        if len(currencies) != 1:
            return PricingFailure(f"discounting needs one currency, got {currencies}")
        currency = next(iter(currencies))

        # Segment-and-settle (A2): cashflows on/before valuation are historical
        # (excluded from PV — the shell settles them), never discounted. The
        # current coupon period contributes accrued interest.
        pv = 0.0
        accrued = 0.0
        for cf in cashflows:
            if cf.date > valuation:
                pv += cf.amount.amount * market.discount_curve.df(cf.date)
                if cf.accrual is not None:
                    accrued += cf.amount.amount * cf.accrual.earned_fraction(valuation)
        return PricingResult(pv=Money(pv, currency), accrued=Money(accrued, currency))
