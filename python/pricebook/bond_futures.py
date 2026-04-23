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
    accrued_at_purchase: float = 0.0,
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
        accrued_at_purchase: accrued interest at purchase (for dirty cost).
    """
    if days_to_delivery <= 0 or bond_price <= 0:
        return 0.0

    invoice = futures_price * cf + accrued_at_delivery
    cost = bond_price + accrued_at_purchase
    profit = invoice - cost + coupon_income

    return profit / cost * (360.0 / days_to_delivery)


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
    repo_rate: float,
    days_to_delivery: int,
    coupon_income: float = 0.0,
    accrued_at_purchase: float = 0.0,
) -> BondFuturesBasis:
    """Compute gross and net basis for a bond vs futures.

    Gross basis = bond_price - CF × futures_price.
    Carry = coupon_income - financing_cost (to delivery).
    Net basis = gross_basis - carry ≈ delivery option value.

    Args:
        bond_price: clean price.
        futures_price: futures settlement price.
        cf: conversion factor.
        repo_rate: financing rate.
        days_to_delivery: days to delivery.
        coupon_income: actual coupon received between now and delivery.
        accrued_at_purchase: accrued at trade date (for financing cost on dirty).
    """
    gross = bond_price - cf * futures_price

    # Carry to delivery
    dt = days_to_delivery / 360.0
    dirty = bond_price + accrued_at_purchase
    financing = dirty * repo_rate * dt
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


# ---- Invoice price ----

def invoice_price(
    futures_price: float,
    cf: float,
    accrued_at_delivery: float,
) -> float:
    """Futures settlement invoice price.

    The short delivers the bond and receives:
        invoice = futures_price × CF + accrued_at_delivery.
    """
    return futures_price * cf + accrued_at_delivery


# ---- Forward bond price ----

@dataclass
class ForwardBondPrice:
    """Forward bond pricing result."""
    forward_dirty: float
    forward_clean: float
    carry: float
    repo_cost: float
    coupon_income: float


def forward_bond_price(
    dirty_price: float,
    repo_rate: float,
    days_to_forward: int,
    coupon_income: float = 0.0,
    accrued_at_forward: float = 0.0,
) -> ForwardBondPrice:
    """Forward price of a bond for delivery on a future date.

    Forward dirty = dirty_today × (1 + repo × T) - coupon_income.
    Forward clean = forward_dirty - accrued_at_forward.

    This is the cash-and-carry arbitrage relationship.

    Args:
        dirty_price: current dirty (full) price.
        repo_rate: financing rate (simple, ACT/365 or ACT/360).
        days_to_forward: days until forward settlement.
        coupon_income: coupons received between now and forward date.
        accrued_at_forward: accrued interest at forward delivery date.
    """
    dt = days_to_forward / 360.0
    repo_cost = dirty_price * repo_rate * dt
    fwd_dirty = dirty_price + repo_cost - coupon_income
    fwd_clean = fwd_dirty - accrued_at_forward
    carry = coupon_income - repo_cost

    return ForwardBondPrice(
        forward_dirty=fwd_dirty,
        forward_clean=fwd_clean,
        carry=carry,
        repo_cost=repo_cost,
        coupon_income=coupon_income,
    )


# ---- Delivery basket analytics ----

@dataclass
class BasketBondAnalytics:
    """Analytics for a single bond in the delivery basket."""
    index: int
    bond: DeliverableBond
    gross_basis: float
    implied_repo: float
    switch_yield: float | None   # yield at which this bond becomes CTD


@dataclass
class DeliveryBasketResult:
    """Full delivery basket analysis."""
    ctd_index: int
    bonds: list[BasketBondAnalytics]


def delivery_basket(
    deliverables: list[DeliverableBond],
    futures_price: float,
    accrued_at_delivery: list[float],
    coupon_incomes: list[float],
    days_to_delivery: int,
    accrued_at_purchase: list[float] | None = None,
) -> DeliveryBasketResult:
    """Analyse full delivery basket: implied repo ranking, CTD identification.

    Args:
        deliverables: bonds in the basket.
        futures_price: current futures price.
        accrued_at_delivery: accrued at delivery for each bond.
        coupon_incomes: coupon income to delivery for each bond.
        days_to_delivery: days until delivery.
        accrued_at_purchase: accrued at purchase for each bond (default 0).
    """
    n = len(deliverables)
    if accrued_at_purchase is None:
        accrued_at_purchase = [0.0] * n

    bonds_analytics = []
    for i, d in enumerate(deliverables):
        gross = d.market_price - d.conversion_factor * futures_price
        repo = implied_repo_rate(
            d.market_price, futures_price, d.conversion_factor,
            accrued_at_delivery[i], coupon_incomes[i],
            days_to_delivery, accrued_at_purchase[i],
        )
        bonds_analytics.append(BasketBondAnalytics(
            index=i, bond=d, gross_basis=gross,
            implied_repo=repo, switch_yield=None,
        ))

    # CTD = highest implied repo
    ctd_idx = max(range(n), key=lambda i: bonds_analytics[i].implied_repo)

    return DeliveryBasketResult(ctd_index=ctd_idx, bonds=bonds_analytics)


# ---- Futures hedge ratio ----

def futures_hedge_ratio(
    bond_dv01: float,
    ctd_dv01: float,
    ctd_cf: float,
) -> float:
    """Number of futures contracts to hedge a bond position.

    HR = (bond_DV01 / CTD_DV01) × CTD_CF.

    The CTD's DV01 determines how the futures price moves.
    The CF converts between bond-space and futures-space.
    """
    if ctd_dv01 <= 0:
        return 0.0
    return (bond_dv01 / ctd_dv01) * ctd_cf


def tail_adjusted_hedge_ratio(
    bond_dv01: float,
    ctd_dv01: float,
    ctd_cf: float,
    repo_rate: float,
    days_to_delivery: int,
) -> float:
    """Hedge ratio with tail adjustment for daily margin on futures.

    Futures P&L is received/paid daily (margin), so the present value
    of the futures hedge is slightly different from the bond position.

    Tail HR = HR / (1 + repo × T_delivery).
    """
    hr = futures_hedge_ratio(bond_dv01, ctd_dv01, ctd_cf)
    dt = days_to_delivery / 360.0
    return hr / (1.0 + repo_rate * dt)


# ---- Calendar spread ----

@dataclass
class CalendarSpreadResult:
    """Calendar spread analysis."""
    spread: float           # front - back price
    roll_cost: float        # carry difference
    front_carry: float
    back_carry: float


def calendar_spread(
    front_price: float,
    back_price: float,
    front_carry: float,
    back_carry: float,
) -> CalendarSpreadResult:
    """Calendar spread: front month vs back month futures.

    Roll cost = back carry - front carry.
    Positive roll cost means it's expensive to roll from front to back.

    Args:
        front_price: front month futures price.
        back_price: back month futures price.
        front_carry: carry to front delivery.
        back_carry: carry to back delivery.
    """
    spread = front_price - back_price
    roll_cost = back_carry - front_carry

    return CalendarSpreadResult(
        spread=spread,
        roll_cost=roll_cost,
        front_carry=front_carry,
        back_carry=back_carry,
    )
