"""Cost-of-carry decomposition and arbitrage detection.

Explicit breakdown of the forward premium into risk-free rate,
storage costs, and convenience yield. Cash-and-carry / reverse
cash-and-carry arbitrage detection.

* :class:`CostOfCarryResult` — decomposition result.
* :func:`cost_of_carry` — decompose forward premium.
* :func:`cash_and_carry_arb` — detect cash-and-carry arbitrage.
* :func:`reverse_cash_and_carry_arb` — detect reverse arb.

References:
    Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 5.
    Working, *The Theory of the Price of Storage*, AER, 1949.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CostOfCarryResult:
    """Cost-of-carry decomposition."""
    total_carry: float          # r + c − y (annualised)
    risk_free_rate: float       # r
    storage_cost: float         # c (annual, as fraction of spot)
    convenience_yield: float    # y (implied)
    forward_premium_pct: float  # (F − S) / S as %
    contango: bool              # F > S

    def to_dict(self) -> dict:
        return vars(self)


def cost_of_carry(
    spot: float,
    futures_price: float,
    maturity_years: float,
    rate: float = 0.04,
    storage_cost: float = 0.0,
) -> CostOfCarryResult:
    """Decompose forward premium into cost-of-carry components.

    F = S × exp((r + c − y) × T)

    Solving for convenience yield:
    y = r + c − (1/T) × ln(F/S)

    Args:
        spot: current spot price.
        futures_price: observed futures/forward price.
        maturity_years: time to delivery.
        rate: risk-free rate (continuous).
        storage_cost: annual storage cost as fraction of spot.
    """
    if maturity_years <= 0 or spot <= 0 or futures_price <= 0:
        return CostOfCarryResult(0, rate, storage_cost, 0, 0, False)

    forward_premium = (futures_price - spot) / spot
    implied_carry = math.log(futures_price / spot) / maturity_years
    convenience_yield = rate + storage_cost - implied_carry
    contango = futures_price > spot

    return CostOfCarryResult(
        total_carry=implied_carry,
        risk_free_rate=rate,
        storage_cost=storage_cost,
        convenience_yield=convenience_yield,
        forward_premium_pct=forward_premium * 100,
        contango=contango,
    )


@dataclass
class ArbitrageResult:
    """Arbitrage opportunity detection."""
    arb_type: str               # "cash_and_carry" or "reverse"
    profit_per_unit: float      # arbitrage profit per unit spot
    profit_pct: float           # as % of spot
    breakeven_cost: float       # max transaction cost before arb disappears
    feasible: bool              # profit > transaction_cost

    def to_dict(self) -> dict:
        return vars(self)


def cash_and_carry_arb(
    spot: float,
    futures_price: float,
    maturity_years: float,
    rate: float = 0.04,
    storage_cost: float = 0.0,
    transaction_cost: float = 0.001,
    repo_rate: float | None = None,
) -> ArbitrageResult:
    """Detect cash-and-carry arbitrage.

    Strategy: buy spot, sell futures, finance at repo rate.
    Profit if F > S × exp((repo + storage) × T).

    For bonds: buy bond, sell bond futures, finance via repo.

    Args:
        spot: current spot/cash price.
        futures_price: observed futures price.
        maturity_years: time to delivery.
        rate: risk-free rate.
        storage_cost: annual storage cost (fraction of spot).
        transaction_cost: round-trip transaction cost (fraction).
        repo_rate: repo financing rate (if None, uses risk-free rate).
    """
    financing_rate = repo_rate if repo_rate is not None else rate
    carry_cost = financing_rate + storage_cost

    # Fair futures = S × exp(carry × T)
    fair_futures = spot * math.exp(carry_cost * maturity_years)

    # Profit: sell futures above fair value, buy spot
    profit = futures_price - fair_futures
    profit_pct = profit / spot * 100 if spot > 0 else 0

    # Breakeven: max transaction cost before arb disappears
    breakeven = abs(profit) / spot if spot > 0 else 0

    feasible = profit > spot * transaction_cost

    return ArbitrageResult(
        arb_type="cash_and_carry",
        profit_per_unit=profit,
        profit_pct=profit_pct,
        breakeven_cost=breakeven,
        feasible=feasible,
    )


def reverse_cash_and_carry_arb(
    spot: float,
    futures_price: float,
    maturity_years: float,
    rate: float = 0.04,
    storage_cost: float = 0.0,
    transaction_cost: float = 0.001,
    borrow_cost: float = 0.0,
) -> ArbitrageResult:
    """Detect reverse cash-and-carry arbitrage.

    Strategy: short sell spot, buy futures, invest proceeds.
    Profit if F < S × exp((r − borrow_cost − convenience_yield) × T).

    Only feasible when the asset can be borrowed/shorted.

    Args:
        borrow_cost: cost of borrowing the physical commodity (annual).
    """
    carry_cost = rate - borrow_cost + storage_cost
    fair_futures = spot * math.exp(carry_cost * maturity_years)

    # Profit: buy futures below fair value, short sell spot
    profit = fair_futures - futures_price
    profit_pct = profit / spot * 100 if spot > 0 else 0

    breakeven = abs(profit) / spot if spot > 0 else 0
    feasible = profit > spot * transaction_cost

    return ArbitrageResult(
        arb_type="reverse_cash_and_carry",
        profit_per_unit=profit,
        profit_pct=profit_pct,
        breakeven_cost=breakeven,
        feasible=feasible,
    )


def carry_roll_decomposition(
    spot: float,
    front_futures: float,
    back_futures: float,
    front_maturity: float,
    back_maturity: float,
    rate: float = 0.04,
) -> dict:
    """Decompose total return into carry and roll components.

    Carry: income from holding the asset (coupon, div, convenience yield).
    Roll: cost/benefit from rolling futures contracts.

    carry_return = (F_front − S) / S × (1/T_front)  (annualised)
    roll_return = (F_front − F_back) / F_front × (1/(T_back − T_front))

    Args:
        front_futures: near-month futures price.
        back_futures: far-month futures price.
        front_maturity: near-month time to delivery (years).
        back_maturity: far-month time to delivery (years).
    """
    if spot <= 0 or front_maturity <= 0:
        return {"carry": 0, "roll": 0, "total": 0}

    # Carry: annualised basis
    carry = (front_futures - spot) / spot / front_maturity

    # Roll: annualised calendar spread return
    dt = back_maturity - front_maturity
    if dt > 0 and front_futures > 0:
        roll = (front_futures - back_futures) / front_futures / dt
    else:
        roll = 0.0

    return {
        "carry": carry,
        "roll": roll,
        "total": carry + roll,
        "contango": back_futures > front_futures,
    }
