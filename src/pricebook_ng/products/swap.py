"""Vanilla interest-rate swap — pure product data (L2).

A frozen description with no behaviour (pricing lives in L4). A `VanillaSwap` is a
fixed leg and a float leg on one notional/currency. The legs are bundled value objects
(≤5 fields each): a swap of 8 loose primitives is the same un-bundled smell PLR0913
can't see (CLAUDE.md §3b). The float leg's projection index arrives with multicurve —
its second (projection) curve is the consumer that earns it (rule of two).

Provenance:
  quarry: python/pricebook/fixed_income/interest_rate_swap.py
  source: redesign/18 (T1 linear products); CLAUDE.md §2 (products are pure data)
  oracle: EURIBOR par swap reprices to zero NPV dual-curve through the L4 engine
  slice:  dual-curve-euribor-estr (T1 slice 2); float leg gained its index here
"""

from __future__ import annotations

from dataclasses import dataclass

from pricebook_ng.foundation import Currency, DayCountConvention, RateIndex, Schedule


@dataclass(frozen=True)
class FixedLeg:
    """A fixed leg: its accrual `schedule`, `day_count`, and fixed `rate`."""

    schedule: Schedule
    day_count: DayCountConvention
    rate: float


@dataclass(frozen=True)
class FloatLeg:
    """A floating leg: its accrual `schedule`, `day_count`, and the `index` it fixes on.
    The `index` selects the projection curve (`CurveSet.projection(index)`); for an OIS
    index that curve is the discount curve itself (single-curve degenerate)."""

    schedule: Schedule
    day_count: DayCountConvention
    index: RateIndex


@dataclass(frozen=True)
class VanillaSwap:
    """A fixed-for-float swap on one `notional`/`currency`. PV is taken from the
    fixed-rate PAYER's side (pay fixed, receive float); the receiver is its negative."""

    notional: float
    currency: Currency
    fixed_leg: FixedLeg
    float_leg: FloatLeg
