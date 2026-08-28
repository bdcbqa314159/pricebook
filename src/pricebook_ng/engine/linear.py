"""Linear (swap) pricer — registered behind the L4 engine registry.

`price_swap` composes the SAME `rpv01`/`float_leg_pv` building blocks the calibrator uses
(§3d), reaching the curves only through `model.market` (A1). It is registered for
`VanillaSwap` and reached via the registry's `dispatch` (re-exported here as `price`, the
public entry). PV is the fixed-rate PAYER's: `N·(float − rate·annuity)`, zero at par.
Failure is a value (invariant 4).

Provenance:
  quarry: python/pricebook/pricing/ (swap engine)
  source: CLAUDE.md §2 (stateless engine, A1) · §3d (shared atoms); §1 (registry dispatch)
  oracle: EURIBOR swap → zero NPV dual-curve to 1e-9; DV01 analytic vs finite-diff to 1e-6
  slice:  cash-instruments (T1 slice 3, Step A — swap pricer moved behind the registry, body unchanged)
"""

from __future__ import annotations

from pricebook_ng.engine.registry import dispatch as price
from pricebook_ng.engine.registry import register
from pricebook_ng.engine.seasoned import (
    current_period_failure,
    current_period_float_rate,
    split_current_period,
)
from pricebook_ng.foundation import Accrual, Money, PricingFailure, PricingResult, future_periods
from pricebook_ng.market.building_blocks import float_leg_pv, rpv01
from pricebook_ng.models.protocols import CalibratedModel
from pricebook_ng.products.swap import VanillaSwap

__all__ = ["price", "price_swap"]


@register(VanillaSwap)
def price_swap(swap: VanillaSwap, model: CalibratedModel) -> PricingResult | PricingFailure:
    """Mark a vanilla swap off the model's curves. Payer PV = N·(float − rate·annuity)."""
    curves = model.market.curves
    vd = model.market.valuation_date
    dc = swap.float_leg.day_count
    try:
        discount = curves.discount(swap.currency, swap.collateral)  # CSA-keyed (A1: through the model)
        projection = curves.projection(swap.float_leg.index)
        # invariant 6: mark = FUTURE PV only; historical periods excluded ABOVE the atoms (unchanged).
        fixed_sched = future_periods(swap.fixed_leg.schedule, vd)
        float_sched = future_periods(swap.float_leg.schedule, vd)
        annuity = rpv01(fixed_sched, swap.fixed_leg.day_count, discount)  # incl. the current fixed coupon
        # the current in-progress float period is spliced from the past fixing (#3b); the strictly-future
        # periods price through the UNCHANGED float_leg_pv atom (spot/boundary → no split → byte-identical).
        current, future_float = split_current_period(float_sched, vd)
        floating = float_leg_pv(future_float, dc, discount, projection)
        accrued: Money | None = None
        if current is not None:
            cur = Accrual(current.accrual_start, current.accrual_end, dc)
            try:
                rate = current_period_float_rate(swap.float_leg.index, cur, projection, model.market.fixings, vd)
            except (ValueError, KeyError):  # no fixing for the current period → honest failure (invariant 4)
                return current_period_failure(current.accrual_start, vd) or PricingFailure(
                    f"missing fixing for the current period starting {current.accrual_start.isoformat()}"
                )
            floating += discount.df(current.payment_date) * cur.year_fraction() * rate
            elapsed = Accrual(current.accrual_start, vd, dc).year_fraction()  # earned-but-unpaid slice
            accrued = Money(swap.notional * elapsed * (rate - swap.fixed_leg.rate), swap.currency)
    except (ValueError, KeyError) as exc:  # cashflow beyond a curve, or an unresolved curve key
        return PricingFailure(str(exc))
    pv = swap.notional * (floating - swap.fixed_leg.rate * annuity)
    basis = None if swap.collateral in (None, swap.currency) else swap.collateral
    return PricingResult(pv=Money(pv, swap.currency), accrued=accrued, basis=basis)
