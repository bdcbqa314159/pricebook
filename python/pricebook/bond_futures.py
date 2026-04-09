"""Bond futures: conversion factors, cheapest-to-deliver, implied repo, basis.

A bond futures contract allows delivery of any bond from a basket.
Each deliverable bond has a conversion factor that normalises its
price relative to a standard coupon. The cheapest-to-deliver (CTD)
bond minimises the net cost of delivery.

    from pricebook.bond_futures import (
        conversion_factor, cheapest_to_deliver, bond_futures_basis,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve
from pricebook.bond import FixedRateBond


# ---- Conversion factor ----

def conversion_factor(
    coupon_rate: float,
    years_to_maturity: float,
    yield_standard: float = 0.06,
    frequency: int = 2,
) -> float:
    """Bond conversion factor for futures delivery.

    Approximation: price of the bond at the standard yield (typically 6%),
    per $1 face value. Rounds maturity to nearest quarter.

    CF = PV(coupons + principal at standard yield) / 100

    Args:
        coupon_rate: annual coupon rate.
        years_to_maturity: time to maturity in years (rounded to nearest quarter).
        yield_standard: standard yield for CF calculation (default 6%).
        frequency: coupon frequency (2 = semi-annual).
    """
    # Round to nearest quarter
    n_quarters = max(1, round(years_to_maturity * 4))
    n_periods = n_quarters * frequency // 4
    if n_periods < 1:
        n_periods = 1

    c = coupon_rate / frequency
    y = yield_standard / frequency

    if abs(y) < 1e-12:
        # Zero yield: bond price = sum of coupons + principal
        return c * n_periods + 1.0

    # PV of coupons + principal
    annuity = (1 - (1 + y) ** (-n_periods)) / y
    pv = c * annuity + (1 + y) ** (-n_periods)

    return pv


# ---- Deliverable bond ----

@dataclass
class DeliverableBond:
    """A bond in the futures delivery basket."""
    bond: FixedRateBond
    market_price: float  # clean price per 100 face
    conversion_factor: float
    accrued: float = 0.0


# ---- Cheapest-to-deliver ----

@dataclass
class CTDResult:
    """Cheapest-to-deliver analysis."""
    ctd_index: int
    ctd_bond: DeliverableBond
    gross_basis: float
    net_basis: float
    implied_repo: float
    all_bases: list[dict]


def cheapest_to_deliver(
    deliverables: list[DeliverableBond],
    futures_price: float,
) -> CTDResult:
    """Find the cheapest-to-deliver bond.

    CTD minimises: bond_price - CF × futures_price (gross basis).
    Equivalently, CTD has the highest implied repo rate.

    Args:
        deliverables: list of deliverable bonds with prices and CFs.
        futures_price: current futures price.
    """
    if not deliverables:
        raise ValueError("need at least 1 deliverable bond")

    bases = []
    for i, d in enumerate(deliverables):
        gross = d.market_price - d.conversion_factor * futures_price
        bases.append({
            "index": i,
            "gross_basis": gross,
            "price": d.market_price,
            "cf": d.conversion_factor,
        })

    # CTD = minimum gross basis
    ctd_idx = min(range(len(bases)), key=lambda i: bases[i]["gross_basis"])
    ctd = deliverables[ctd_idx]
    gross = bases[ctd_idx]["gross_basis"]

    return CTDResult(
        ctd_index=ctd_idx,
        ctd_bond=ctd,
        gross_basis=gross,
        net_basis=gross,  # simplified: net basis assumes no carry adjustment
        implied_repo=float("nan"),  # use implied_repo_rate() to compute explicitly
        all_bases=bases,
    )


# ---- Implied repo rate ----

def implied_repo_rate(
    bond_price: float,
    futures_price: float,
    cf: float,
    accrued_at_delivery: float,
    coupon_income: float,
    days_to_delivery: int,
) -> float:
    """Implied repo rate from bond futures delivery.

    repo = (futures_invoice - purchase_cost + coupon_income) / purchase_cost × (365/days).

    Args:
        bond_price: clean purchase price.
        futures_price: futures settlement price.
        cf: conversion factor.
        accrued_at_delivery: accrued interest at delivery.
        coupon_income: coupon received between now and delivery.
        days_to_delivery: days until delivery.
    """
    if days_to_delivery <= 0 or bond_price <= 0:
        return 0.0

    invoice = futures_price * cf + accrued_at_delivery
    cost = bond_price  # simplified: ignore accrued at purchase
    profit = invoice - cost + coupon_income

    return profit / cost * (365.0 / days_to_delivery)


# ---- Bond futures basis ----

@dataclass
class BondFuturesBasis:
    """Basis analysis for a single bond vs futures."""
    bond_price: float
    futures_price: float
    cf: float
    gross_basis: float
    net_basis: float
    carry: float


def bond_futures_basis(
    bond_price: float,
    futures_price: float,
    cf: float,
    coupon_rate: float,
    repo_rate: float,
    days_to_delivery: int,
    face_value: float = 100.0,
) -> BondFuturesBasis:
    """Compute gross and net basis for a bond vs futures.

    Gross basis = bond_price - CF × futures_price.
    Carry = coupon_income - financing_cost (to delivery).
    Net basis = gross_basis - carry ≈ delivery option value.
    """
    gross = bond_price - cf * futures_price

    # Carry to delivery
    dt = days_to_delivery / 365.0
    coupon_income = coupon_rate * face_value * dt
    financing = bond_price * repo_rate * dt
    carry = coupon_income - financing

    net = gross - carry

    return BondFuturesBasis(
        bond_price=bond_price,
        futures_price=futures_price,
        cf=cf,
        gross_basis=gross,
        net_basis=net,
        carry=carry,
    )
