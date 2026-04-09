"""Equity index and commodity futures: pricing, margining, roll, and spreads.

Equity futures: F = S × exp((r - q) × T), daily settlement.
Commodity futures: observed forward curve, contango/backwardation, roll yield.

    from pricebook.futures import (
        EquityFuture, CommodityFuture, calendar_spread,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


# ---- Equity index futures ----

class EquityFuture:
    """Equity index future with daily margining.

    Args:
        spot: current index level.
        expiry: futures expiry date.
        rate: risk-free rate (continuous compounding).
        div_yield: continuous dividend yield.
        notional_per_point: dollar value per index point (e.g. 50 for ES).
    """

    def __init__(
        self,
        spot: float,
        expiry: date,
        rate: float,
        div_yield: float = 0.0,
        notional_per_point: float = 1.0,
    ):
        self.spot = spot
        self.expiry = expiry
        self.rate = rate
        self.div_yield = div_yield
        self.notional_per_point = notional_per_point

    def fair_price(self, valuation_date: date) -> float:
        """Theoretical futures price: S × exp((r - q) × T)."""
        T = year_fraction(valuation_date, self.expiry, DayCountConvention.ACT_365_FIXED)
        if T <= 0:
            return self.spot
        return self.spot * math.exp((self.rate - self.div_yield) * T)

    def basis(self, valuation_date: date) -> float:
        """Basis = futures price - spot price."""
        return self.fair_price(valuation_date) - self.spot

    def daily_settlement_pnl(
        self,
        prev_settlement: float,
        curr_settlement: float,
        contracts: int = 1,
    ) -> float:
        """Daily variation margin P&L.

        P&L = (curr - prev) × notional_per_point × contracts.
        """
        return (curr_settlement - prev_settlement) * self.notional_per_point * contracts

    def convergence(self, valuation_date: date) -> float:
        """Distance to convergence: fair_price - spot → 0 at expiry."""
        return self.fair_price(valuation_date) - self.spot


# ---- Commodity futures ----

class CommodityFuture:
    """Commodity future with observed price and term structure analysis.

    Args:
        commodity: commodity name.
        expiry: delivery month/date.
        price: observed futures price.
    """

    def __init__(
        self,
        commodity: str,
        expiry: date,
        price: float,
    ):
        self.commodity = commodity
        self.expiry = expiry
        self.price = price


def contango_or_backwardation(
    near: CommodityFuture,
    far: CommodityFuture,
) -> str:
    """Determine if the curve is in contango or backwardation.

    Contango: far > near (storage costs dominate).
    Backwardation: far < near (convenience yield dominates).
    """
    if near.expiry >= far.expiry:
        raise ValueError("near must expire before far")
    if far.price > near.price:
        return "contango"
    elif far.price < near.price:
        return "backwardation"
    return "flat"


def roll_yield(
    near: CommodityFuture,
    far: CommodityFuture,
) -> float:
    """Roll yield: annualised return from rolling near to far.

    roll_yield = (near - far) / near × (365 / days_between) annualised.
    Positive in backwardation (sell high near, buy low far).
    """
    if near.expiry >= far.expiry:
        raise ValueError("near must expire before far")
    days = (far.expiry - near.expiry).days
    if days <= 0 or near.price <= 0:
        return 0.0
    return (near.price - far.price) / near.price * (365.0 / days)


# ---- Calendar spread ----

@dataclass
class CalendarSpreadResult:
    """Calendar spread analysis."""
    near_price: float
    far_price: float
    spread: float
    roll_yield: float
    structure: str  # "contango" or "backwardation" or "flat"


def calendar_spread(
    near: CommodityFuture,
    far: CommodityFuture,
) -> CalendarSpreadResult:
    """Calendar spread: long near, short far.

    Spread = near - far.
    Positive spread = backwardation (near > far).
    """
    if near.expiry >= far.expiry:
        raise ValueError("near must expire before far")

    spread = near.price - far.price
    ry = roll_yield(near, far)
    structure = contango_or_backwardation(near, far)

    return CalendarSpreadResult(
        near_price=near.price,
        far_price=far.price,
        spread=spread,
        roll_yield=ry,
        structure=structure,
    )


# ---- Futures strip ----

def futures_strip_curve(
    futures: list[CommodityFuture],
) -> list[tuple[date, float]]:
    """Extract a forward curve from a strip of futures, sorted by expiry."""
    return sorted([(f.expiry, f.price) for f in futures], key=lambda x: x[0])


def implied_convenience_yield(
    spot: float,
    futures_price: float,
    rate: float,
    storage_cost: float,
    T: float,
) -> float:
    """Implied convenience yield from futures price.

    F = S × exp((r + c - y) × T)  →  y = r + c - ln(F/S) / T
    """
    if T <= 0 or spot <= 0 or futures_price <= 0:
        return 0.0
    if futures_price / spot <= 0:
        return 0.0
    return rate + storage_cost - math.log(futures_price / spot) / T
