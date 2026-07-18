"""FRAEngine — single-curve forward rate agreement pricer (L4).

Stateless. Resolves the simply-compounded forward `L(T1,T2) = (P(0,T1)/P(0,T2) - 1)/tau`
from the model's discount curve (single-curve: discount = projection) and values the
single period:
    PV(pay-fixed) = notional · tau · (L - fixed_rate) · P(0,T2)
Composes the curve's `df`, so it prices on any curve (flat or bootstrapped).

Temporal (A2): a forward-starting/spot period (`accrual.start >= valuation`) uses the curve
forward; a **seasoned** period (`start < valuation`) uses the realized reset from the snapshot's
`FixingHistory`; a fully-paid period (`end <= valuation`) is settled (PV 0 — the shell remembers
the realized cash).

Provenance:
  quarry: python/pricebook/fixed_income/fra.py
  source: standard single-curve FRA valuation
  oracle: par (K = forward) -> 0; closed-form off-par; seasoned uses the fixing; settled -> 0
  slice:  fra-spine (CP-2b #3); fra-seasoned-fixings (CP-2c #2 — FixingHistory)
"""

from __future__ import annotations

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.numerical_config import NumericalConfig
from pricebook_ng.foundation.results import PricingFailure, PricingResult
from pricebook_ng.foundation.time import year_fraction
from pricebook_ng.models.discounting_model import CalibratedModel
from pricebook_ng.products.fra import ForwardRateAgreement


class FRAEngine:
    """Prices a FRA by discounting the single forward-vs-fixed period on the curve."""

    def price(
        self, fra: ForwardRateAgreement, model: CalibratedModel, numerics: NumericalConfig
    ) -> PricingResult | PricingFailure:
        market = model.market
        currency = fra.face.currency
        start, end, day_count = fra.accrual.start, fra.accrual.end, fra.accrual.day_count
        if end <= market.valuation_date:
            return PricingResult(pv=Money(0.0, currency))  # settled — the shell remembers the realized cash
        curve = market.discount_curve
        tau = year_fraction(start, end, day_count)
        if start < market.valuation_date:
            rate = market.fixings.get(start)  # seasoned: the reset already fixed (A2)
            if rate is None:
                return PricingFailure(f"seasoned FRA needs the fixing at {start}")
        else:
            rate = curve.forward_rate(start, end, day_count)  # forward-implied building block
        value = fra.face.amount * tau * (rate - fra.fixed_rate) * curve.df(end)
        if not fra.pay_fixed:
            value = -value
        return PricingResult(pv=Money(value, currency))
