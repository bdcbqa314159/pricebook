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
  slice:  S04; S05 (Money face + ScheduleTerms — 5-arg ceiling);
          S06 (coupons via shared fixed_coupon_cashflows)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.cashflow import Cashflow
from pricebook_ng.foundation.money import Currency, Money
from pricebook_ng.foundation.schedule import ScheduleTerms
from pricebook_ng.products.leg import fixed_coupon_cashflows


@dataclass(frozen=True)
class FixedRateBond:
    """A fixed-rate bond: its coupon+redemption cashflows plus identifying data."""

    notional: float
    coupon_rate: float
    currency: Currency
    cashflows: tuple[Cashflow, ...]


def fixed_rate_bond(
    face: Money,
    coupon_rate: float,
    start: date,
    maturity: date,
    terms: ScheduleTerms,
) -> FixedRateBond:
    """Build a fixed-rate bond: fixed coupons plus the notional redemption.

    `face` is the notional as `Money(amount, currency)`. The notional redeems at
    maturity as a separate cashflow on that date.
    """
    coupons = fixed_coupon_cashflows(face, coupon_rate, start, maturity, terms)
    redemption = Cashflow(date=maturity, amount=face)
    return FixedRateBond(
        notional=face.amount,
        coupon_rate=coupon_rate,
        currency=face.currency,
        cashflows=(*coupons, redemption),
    )
