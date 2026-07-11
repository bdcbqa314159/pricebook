"""FixedRateBond — a fixed-coupon bond as pure data (L2).

The bond describes a leg of coupon cashflows plus the redemption; it holds NO
`pv` method (CLAUDE.md 2 — pricing lives in the L4 engine). The builder
`fixed_rate_bond(...)` expands schedule + day-count into explicit `Cashflow`s at
construction, so the frozen instrument stays pure data.

Coupons are dropped into a plain `tuple[Cashflow, ...]`; a shared `Leg` type is
not introduced until a second consumer needs it (rule of two, CLAUDE.md 6b).

Provenance:
  quarry: python/pricebook/fixed_income/ (fixed-rate bond / fixed leg)
  source: standard fixed-coupon bond; ISDA 2006 accrual
  oracle: closed-form discounted-cashflow PV < 1e-12 (S04)
  slice:  S04
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.calendar import BusinessDayConvention, Calendar
from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import Frequency, generate_schedule
from pricebook_ng.foundation.time import DayCountConvention, year_fraction


@dataclass(frozen=True)
class FixedRateBond:
    """A fixed-rate bond: its coupon+redemption cashflows plus identifying data."""

    notional: float
    coupon_rate: float
    currency: Currency
    cashflows: tuple[Cashflow, ...]


def fixed_rate_bond(
    notional: float,
    coupon_rate: float,
    start: date,
    maturity: date,
    frequency: Frequency,
    day_count: DayCountConvention,
    currency: Currency,
    calendar: Calendar | None = None,
    convention: BusinessDayConvention = BusinessDayConvention.MODIFIED_FOLLOWING,
) -> FixedRateBond:
    """Build a fixed-rate bond, expanding its schedule into cashflows.

    Coupon i pays `notional * coupon_rate * year_fraction(period)` at the period
    end; the notional redeems at maturity (as a separate cashflow on that date).
    """
    schedule = generate_schedule(start, maturity, frequency, calendar, convention)
    cashflows = [
        Cashflow(
            date=schedule[i],
            amount=Money(
                notional * coupon_rate * year_fraction(schedule[i - 1], schedule[i], day_count),
                currency,
            ),
        )
        for i in range(1, len(schedule))
    ]
    cashflows.append(Cashflow(date=schedule[-1], amount=Money(notional, currency)))
    return FixedRateBond(
        notional=notional,
        coupon_rate=coupon_rate,
        currency=currency,
        cashflows=tuple(cashflows),
    )
