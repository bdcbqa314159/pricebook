"""Instrument atoms — Accrual, Cashflow, Leg (L0).

Finance-free pure data: an `Accrual` is a day-count period, a `Cashflow` a dated
payment, a `Leg` an ordered run of cashflows on one convention. `Accrual.year_fraction`
is the ergonomic entry point over the Slice-2 `year_fraction` primitive — it bundles
start + end + day_count, so callers pass one object, not three loose args.

Provenance:
  quarry: python/pricebook/fixed_income/fixed_leg.py (Cashflow atom)
  source: promoted to L0 per redesign/16 §2.3 (instrument atoms)
  oracle: Accrual.year_fraction matches the S2 primitive; leg construction
  slice:  money-quantity (Topic 0 S4)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from pricebook_ng.foundation.day_count import (
    CouponPeriod,
    DayCountConvention,
    year_fraction,
)
from pricebook_ng.foundation.money import Money
from pricebook_ng.foundation.settlement import Delivery

# calendar is same-layer (L0), so a direct import is fine here.
from pricebook_ng.foundation.calendar import Calendar

# A leg's flow is cash OR physical: a `Cashflow(Money)` or a `Delivery(Quantity)`
# (gate audit S2). Pay/receive is the SIGN of the amount — no direction field (S13).


@dataclass(frozen=True)
class Accrual:
    """A day-count period: its span and convention. `year_fraction` is the ergonomic
    wrapper over the L0 primitive (start + end + day_count bundled). An accrual is ordered
    by construction — a reversed or zero-length span raises (S14)."""

    start: date
    end: date
    day_count: DayCountConvention

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(f"accrual must be ordered (start < end); got {self.start}..{self.end}")

    def year_fraction(
        self, *, coupon_period: CouponPeriod | None = None, calendar: Calendar | None = None
    ) -> float:
        return year_fraction(
            self.start, self.end, self.day_count,
            coupon_period=coupon_period, calendar=calendar,
        )


@dataclass(frozen=True)
class Cashflow:
    """A single payment of `amount` on `date`; a coupon additionally carries its
    `accrual` period (for accrued interest at valuation), a bullet leaves it `None`."""

    date: date
    amount: Money
    accrual: Accrual | None = None


@dataclass(frozen=True)
class Leg:
    """An ordered run of flows (cash `Cashflow` or physical `Delivery`) sharing one
    day-count convention. Direction is carried by the sign of each amount, not a flag."""

    flows: tuple[Cashflow | Delivery, ...]
    day_count: DayCountConvention
