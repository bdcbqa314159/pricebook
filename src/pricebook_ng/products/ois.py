"""OvernightIndexSwap — fixed vs compounded-overnight float, as pure data (L2).

An OIS pays a fixed rate against the compounded overnight rate (SOFR/SONIA/ESTR).
Single-curve, the compounded overnight rate over a period equals the curve's forward,
so structurally it is the vanilla IRS (it reuses the swap's `FixedLeg`/`FloatLeg`). The
distinct type carries the RFR semantics; the multi-curve OIS/IBOR basis and daily-fixing
compounding are a later slice.

Provenance:
  quarry: python/pricebook/fixed_income/ois.py
  source: standard single-curve OIS (RFR compounded in arrears)
  oracle: par OIS -> 0; OIS == vanilla IRS (single-curve)
  slice:  ois-spine (CP-2c #4)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.schedule import generate_schedule
from pricebook_ng.products.leg import fixed_coupon_cashflows
from pricebook_ng.products.swap import FixedLeg, FloatLeg, SwapTerms


@dataclass(frozen=True)
class OvernightIndexSwap:
    """A fixed-vs-compounded-overnight swap: same leg shape as a vanilla IRS."""

    fixed_leg: FixedLeg
    float_leg: FloatLeg
    pay_fixed: bool


def overnight_index_swap(
    face: Money, fixed_rate: float, start: date, maturity: date, terms: SwapTerms
) -> OvernightIndexSwap:
    """Build an OIS: fixed coupons + a structural compounded-overnight float leg."""
    fixed_leg = FixedLeg(
        fixed_coupon_cashflows(face, fixed_rate, start, maturity, terms.fixed_schedule)
    )
    float_dates = generate_schedule(
        start, maturity, terms.float_schedule.frequency, terms.float_schedule.roll
    )
    float_leg = FloatLeg(
        face=face, schedule=tuple(float_dates), day_count=terms.float_schedule.day_count
    )
    return OvernightIndexSwap(fixed_leg=fixed_leg, float_leg=float_leg, pay_fixed=terms.pay_fixed)
