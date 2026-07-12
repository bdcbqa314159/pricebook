"""Vanilla interest-rate swap as pure data (L2).

A `VanillaSwap` is a fixed leg vs a float leg. The fixed leg is known cashflows
(a `CashflowProduct`). The float leg is *structural* — only its schedule and
notional; its coupons are the curve's forwards, resolved by the L4 SwapEngine at
pricing time (they are not knowable without a curve). No `pv` method here
(CLAUDE.md 2).

Single-curve (discount = projection), no basis spread — the float leg carries no
day-count because, under single-curve, forward*tau = DF(start)/DF(end) - 1 and
the accrual factor cancels. A projection curve and spread land with a later slice.

Provenance:
  quarry: python/pricebook/fixed_income/ (swap / fixed + float legs)
  source: standard single-curve vanilla IRS
  oracle: par swap reprices to zero NPV (S06)
  slice:  S06
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.schedule import ScheduleTerms, generate_schedule
from pricebook_ng.products.leg import fixed_coupon_cashflows


@dataclass(frozen=True)
class FixedLeg:
    """The fixed leg: known coupon cashflows (a `CashflowProduct`)."""

    cashflows: tuple[Cashflow, ...]


@dataclass(frozen=True)
class FloatLeg:
    """The float leg: structural only — notional face and the period dates."""

    face: Money
    schedule: tuple[date, ...]


@dataclass(frozen=True)
class SwapTerms:
    """Swap specification: each leg's schedule conventions and the direction.

    `pay_fixed=True` is a payer swap (pays the fixed leg, receives the float leg).
    """

    fixed_schedule: ScheduleTerms
    float_schedule: ScheduleTerms
    pay_fixed: bool = True


@dataclass(frozen=True)
class VanillaSwap:
    """A vanilla fixed-vs-float interest-rate swap."""

    fixed_leg: FixedLeg
    float_leg: FloatLeg
    pay_fixed: bool


def vanilla_swap(
    face: Money,
    fixed_rate: float,
    start: date,
    maturity: date,
    terms: SwapTerms,
) -> VanillaSwap:
    """Build a vanilla swap: fixed coupons + a structural float leg."""
    fixed_leg = FixedLeg(
        fixed_coupon_cashflows(face, fixed_rate, start, maturity, terms.fixed_schedule)
    )
    float_dates = generate_schedule(
        start, maturity, terms.float_schedule.frequency, terms.float_schedule.roll
    )
    float_leg = FloatLeg(face=face, schedule=tuple(float_dates))
    return VanillaSwap(fixed_leg=fixed_leg, float_leg=float_leg, pay_fixed=terms.pay_fixed)
