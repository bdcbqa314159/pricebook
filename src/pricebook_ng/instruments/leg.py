"""Shared leg construction (L2).

`fixed_coupon_cashflows` is the one definition of a fixed leg's coupons —
`notional * rate * year_fraction(period)` paid at each period end. Consumed by
both the fixed-rate bond and the swap's fixed leg (rule of two, CLAUDE.md 6b).

Provenance:
  quarry: python/pricebook/fixed_income/fixed_leg.py
  source: ISDA 2006 accrual; standard fixed leg
  oracle: exercised by the bond (S04) and swap (S06) closed-form oracles
  slice:  S06
"""

from __future__ import annotations

from datetime import date

from pricebook_ng.foundation.cashflow import Accrual, Cashflow
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.schedule import ScheduleTerms, generate_schedule
from pricebook_ng.foundation.time import year_fraction


def fixed_coupon_cashflows(
    face: Money,
    rate: float,
    start: date,
    maturity: date,
    terms: ScheduleTerms,
) -> tuple[Cashflow, ...]:
    """Coupons of a fixed leg: `notional * rate * tau_i` paid at each period end,
    each carrying its `Accrual` period (for accrued interest at valuation)."""
    notional, currency = face.amount, face.currency
    schedule = generate_schedule(start, maturity, terms.frequency, terms.roll)
    return tuple(
        Cashflow(
            date=schedule[i],
            amount=Money(
                notional * rate * year_fraction(schedule[i - 1], schedule[i], terms.day_count),
                currency,
            ),
            accrual=Accrual(schedule[i - 1], schedule[i], terms.day_count),
        )
        for i in range(1, len(schedule))
    )
