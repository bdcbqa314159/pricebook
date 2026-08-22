"""The benefit table — a product's already-paid cash, by structural dispatch (L6).

The realized cash of a product is product-specific (a swap nets past fixed vs float coupons), so the
shell dispatches by `type(product)` through a registry — the SAME structural pattern as the L4 engine
registry, never an `isinstance`-on-product ladder (CLAUDE.md §1). Each product registers its benefit
with its slice; a product with no benefit table is a value (`PricingFailure`), never a raise.

Realized cash is UNDISCOUNTED actual cash (the benefit table, §2) — the shell REMEMBERS it; it is never
a PV. Float coupons are resolved from the `FixingSource` via the shared `accrued_rate` atom (the same
atom the engine's current-period fixing composes, §3d) — a missing fixing is a `PricingFailure`.

Provenance:
  quarry: python/pricebook/core/fixings.py; core/trade.py (benefit / realized-cash — realigned)
  source: CLAUDE.md §1 (structural dispatch, no isinstance) · §2 (realized undiscounted; shell remembers)
  oracle: a seasoned swap's realized == Σ past coupons (undiscounted), float via fixings, fixed via rate
  slice:  l6-realized (L6 stateful; closes C3/T1)
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import cast

from pricebook_ng.foundation import Accrual, FixingSource, Money, PricingFailure, accrued_rate
from pricebook_ng.products.swap import VanillaSwap

_Benefit = Callable[[object, date, FixingSource], "Money | PricingFailure"]
_BENEFIT: dict[type, _Benefit] = {}


def register_benefit(product_type: type) -> Callable[[_Benefit], _Benefit]:
    """Register a benefit-table function for `product_type` (structural dispatch, no isinstance)."""

    def _register(fn: _Benefit) -> _Benefit:
        _BENEFIT[product_type] = fn
        return fn

    return _register


def benefit(product: object, valuation_date: date, fixings: FixingSource) -> Money | PricingFailure:
    """The product's realized (already-paid, undiscounted) cash at `valuation_date`, dispatched by
    type. An unregistered product returns a `PricingFailure` (invariant 4), never raises."""
    fn = _BENEFIT.get(type(product))
    if fn is None:
        return PricingFailure(f"no benefit table for {type(product).__name__}")
    return fn(product, valuation_date, fixings)


@register_benefit(VanillaSwap)
def _swap_benefit(
    product: object, valuation_date: date, fixings: FixingSource
) -> Money | PricingFailure:
    """A payer swap's realized cash: Σ over PAST periods (payment ≤ valuation) of undiscounted
    `notional·τ·(realized_float − fixed)` — float from `accrued_rate`, fixed from the coupon."""
    swap = cast(VanillaSwap, product)  # the registry dispatched by type — no isinstance ladder
    net = 0.0
    for period in swap.fixed_leg.schedule.periods:
        if period.payment_date <= valuation_date:
            accrual = Accrual(period.accrual_start, period.accrual_end, swap.fixed_leg.day_count)
            net -= accrual.year_fraction() * swap.fixed_leg.rate
    for period in swap.float_leg.schedule.periods:
        if period.payment_date <= valuation_date:
            accrual = Accrual(period.accrual_start, period.accrual_end, swap.float_leg.day_count)
            try:
                rate = accrued_rate(swap.float_leg.index, accrual, fixings)
            except ValueError as exc:  # no fixing for a required historical coupon
                return PricingFailure(str(exc))
            net += accrual.year_fraction() * rate
    return Money(swap.notional * net, swap.currency)
