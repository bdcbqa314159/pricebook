"""The fixed-cashflow pricer (L4) — the whole engine contract, on the simplest product.

`PV = df(payment_date) · amount`, and the only subtlety is the one every product shares:

**A flow on or before the valuation date is historical and excluded.** It is the shell's business
(it already paid), not the mark's, and discounting it would mean a non-positive `t` — a discount
factor above 1 inflating a PV with money that has already moved. The boundary case
(`date == valuation_date`) is where off-by-one errors live, so it is excluded explicitly rather
than falling out of a comparison.

Provenance:
  stock:  _stock/src/pricebook/engine/cash.py (shape mined: discount-and-exclude; the stock's
          pricers are Deposit/FRA/Future, each with its own convention handling)
  source: standard discounting; CLAUDE.md-era §2 invariant 6 (valuation-date partition)
  oracle: `PV = N·e^{−rt}` to 1e-12; analytic vs finite-difference DV01 to 1e-6; the
          on-the-date and one-day-after boundary cases
  crosscheck: none — PV = N·df; the df convention is cross-checked at market/curve.py
  slice:  tranche 1 — minimal spine
"""

from __future__ import annotations

from pricebook.engine.registry import register
from pricebook.foundation import Money, PricingFailure, PricingResult
from pricebook.models.discounting_model import DiscountingModel
from pricebook.products.cash import FixedCashflow


@register(FixedCashflow)
def price_fixed_cashflow(
    product: FixedCashflow, model: DiscountingModel
) -> PricingResult | PricingFailure:
    """Mark a single fixed cashflow off the discount curve its currency names."""
    market = model.market
    flow = product.cashflow

    # Invariant: on or before the valuation date is historical. Excluded BEFORE any curve is
    # touched, so a past flow cannot fail for a missing curve it never needed.
    if flow.date <= market.valuation_date:
        return PricingResult(pv=Money(0.0, product.currency))

    try:
        curve = market.curves.discount(product.currency)
    except KeyError as exc:  # no curve for this currency → a value, not a raise
        return PricingFailure(f"cannot price {product.currency}: {exc}")

    return PricingResult(pv=Money(curve.df(flow.date) * flow.amount.amount, product.currency))
