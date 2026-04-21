"""Cross-currency bond pricing.

A bond issued in a foreign currency, priced from the perspective of a
domestic investor. The FX-hedged yield accounts for the cost of hedging
the currency exposure via FX forwards (CIP basis).

    from pricebook.xccy_bond import fx_hedged_yield, unhedged_yield

References:
    Tuckman & Serrat, *Fixed Income Securities*, Wiley, 2012, Ch. 16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.bond import FixedRateBond
from pricebook.day_count import DayCountConvention, year_fraction
from pricebook.discount_curve import DiscountCurve


@dataclass
class XccyBondResult:
    """Cross-currency bond analysis."""
    foreign_ytm: float
    domestic_ytm_unhedged: float
    fx_hedging_cost: float
    fx_hedged_yield: float
    carry_pickup: float    # hedged yield - domestic benchmark


def fx_hedged_yield(
    foreign_ytm: float,
    domestic_rate: float,
    foreign_rate: float,
) -> float:
    """FX-hedged yield of a foreign bond.

    FX hedge cost ≈ domestic_rate - foreign_rate (CIP).
    Hedged yield ≈ foreign_ytm + (domestic_rate - foreign_rate).

    This is the yield a domestic investor earns after hedging the FX risk
    back to domestic currency via FX forwards.
    """
    hedge_cost = domestic_rate - foreign_rate
    return foreign_ytm + hedge_cost


def cross_currency_pickup(
    foreign_ytm: float,
    domestic_benchmark_ytm: float,
    domestic_rate: float,
    foreign_rate: float,
) -> XccyBondResult:
    """Analyse a cross-currency bond trade.

    Computes the FX-hedged yield and the carry pickup over the domestic
    benchmark at the same maturity.

    Args:
        foreign_ytm: yield of the foreign bond.
        domestic_benchmark_ytm: yield of the domestic benchmark at same tenor.
        domestic_rate: domestic short-term interest rate (for FX hedge cost).
        foreign_rate: foreign short-term interest rate.
    """
    hedge_cost = domestic_rate - foreign_rate
    hedged = foreign_ytm + hedge_cost
    pickup = hedged - domestic_benchmark_ytm

    return XccyBondResult(
        foreign_ytm=foreign_ytm,
        domestic_ytm_unhedged=foreign_ytm,
        fx_hedging_cost=hedge_cost,
        fx_hedged_yield=hedged,
        carry_pickup=pickup,
    )


def breakeven_fx_move(
    foreign_ytm: float,
    domestic_benchmark_ytm: float,
    years: float,
) -> float:
    """FX move that eliminates the yield advantage of a foreign bond.

    If the foreign bond yields more than domestic, the FX can depreciate
    by this amount before the trade becomes unprofitable (unhedged).

    Returns annualised FX depreciation breakeven (positive = foreign depreciation).
    """
    if years <= 0:
        return 0.0
    yield_advantage = foreign_ytm - domestic_benchmark_ytm
    # Total return = yield × years. FX move that wipes it out:
    # (1 + yield_adv)^years = (1 + fx_move)^years
    # Approximate: breakeven ≈ yield_advantage per year
    return yield_advantage
