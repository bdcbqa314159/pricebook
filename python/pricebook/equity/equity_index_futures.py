"""Equity index futures pricing and analytics.

* :class:`IndexFuturesResult` — fair value, basis, carry result.
* :func:`index_futures_fair_value` — cost-of-carry fair value.
* :func:`index_futures_basis` — observed basis and annualised basis.
* :class:`IndexFuturesRollResult` — roll analytics result.
* :func:`index_futures_roll` — calendar spread, roll cost, implied repo.
* :func:`implied_dividend_yield` — back out q from observed futures price.
* :func:`implied_repo_rate` — back out r from observed futures price.
* :func:`fair_value_table` — fair value term structure across multiple expiries.

References:
    Hull, J.C., *Options, Futures and Other Derivatives*, Ch. 5, 10th ed.
    Chance, D.M. & Brooks, R., *Introduction to Derivatives and Risk Management*, 9th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IndexFuturesResult:
    """Result of index futures fair value calculation."""
    fair_value: float
    basis: float
    basis_bps: float
    carry: float
    cost_of_carry_rate: float
    days_to_expiry: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def index_futures_fair_value(
    spot: float,
    rate: float,
    div_yield: float,
    T: float,
    borrow_cost: float = 0.0,
) -> IndexFuturesResult:
    """Cost-of-carry fair value of an equity index future.

    F = S × exp((r - q + b) × T)

    Args:
        spot: current index spot level.
        rate: continuously compounded risk-free rate.
        div_yield: continuous dividend yield.
        T: time to expiry in years.
        borrow_cost: annualised stock borrow cost (default 0).

    Returns:
        :class:`IndexFuturesResult` with fair value, basis, and carry metrics.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    net_carry = rate - div_yield + borrow_cost
    fair_value = spot * math.exp(net_carry * T)
    basis = fair_value - spot
    basis_bps = (basis / spot) * 10_000
    carry = fair_value - spot
    days_to_expiry = T * 365.0

    return IndexFuturesResult(
        fair_value=fair_value,
        basis=basis,
        basis_bps=basis_bps,
        carry=carry,
        cost_of_carry_rate=net_carry,
        days_to_expiry=days_to_expiry,
    )


def index_futures_basis(
    futures_price: float,
    spot: float,
    T: float,
) -> dict[str, float]:
    """Observed basis between futures price and spot.

    Args:
        futures_price: observed market futures price.
        spot: current index spot level.
        T: time to expiry in years.

    Returns:
        Dict with ``basis``, ``basis_bps``, and ``annualised_basis``.
    """
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    basis = futures_price - spot
    basis_bps = (basis / spot) * 10_000
    annualised_basis = (basis / spot) / T

    return {
        "basis": basis,
        "basis_bps": basis_bps,
        "annualised_basis": annualised_basis,
    }


@dataclass
class IndexFuturesRollResult:
    """Result of index futures roll analytics."""
    roll_cost: float
    roll_cost_bps: float
    implied_repo: float
    calendar_spread: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def index_futures_roll(
    front_price: float,
    back_price: float,
    spot: float,
    T_front: float,
    T_back: float,
) -> IndexFuturesRollResult:
    """Roll analytics between front and back index futures contracts.

    The implied repo rate between two contracts is derived from:
    back / front = exp(r_implied × (T_back - T_front))

    Args:
        front_price: observed price of the front (nearby) contract.
        back_price: observed price of the back (deferred) contract.
        spot: current index spot level.
        T_front: time to expiry of front contract in years.
        T_back: time to expiry of back contract in years.

    Returns:
        :class:`IndexFuturesRollResult` with calendar spread, roll cost, and implied repo.
    """
    if T_back <= T_front:
        raise ValueError("T_back must be greater than T_front")
    if front_price <= 0 or back_price <= 0 or spot <= 0:
        raise ValueError("prices and spot must be positive")

    calendar_spread = back_price - front_price
    roll_cost = front_price - back_price
    roll_cost_bps = (roll_cost / spot) * 10_000
    dT = T_back - T_front
    implied_repo = math.log(back_price / front_price) / dT

    return IndexFuturesRollResult(
        roll_cost=roll_cost,
        roll_cost_bps=roll_cost_bps,
        implied_repo=implied_repo,
        calendar_spread=calendar_spread,
    )


def implied_dividend_yield(
    futures_price: float,
    spot: float,
    rate: float,
    T: float,
) -> float:
    """Back out the implied dividend yield from an observed futures price.

    From F = S × exp((r - q) × T):
    q = r - ln(F / S) / T

    Args:
        futures_price: observed market futures price.
        spot: current index spot level.
        rate: continuously compounded risk-free rate.
        T: time to expiry in years.

    Returns:
        Implied continuous dividend yield.
    """
    if futures_price <= 0 or spot <= 0:
        raise ValueError("futures_price and spot must be positive")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    return rate - math.log(futures_price / spot) / T


def implied_repo_rate(
    futures_price: float,
    spot: float,
    div_yield: float,
    T: float,
) -> float:
    """Back out the implied repo rate from an observed futures price.

    From F = S × exp((r - q) × T):
    r = ln(F / S) / T + q

    Args:
        futures_price: observed market futures price.
        spot: current index spot level.
        div_yield: continuous dividend yield.
        T: time to expiry in years.

    Returns:
        Implied continuously compounded repo/risk-free rate.
    """
    if futures_price <= 0 or spot <= 0:
        raise ValueError("futures_price and spot must be positive")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    return math.log(futures_price / spot) / T + div_yield


def fair_value_table(
    spot: float,
    rate: float,
    div_yield: float,
    expiries_years: list[float],
    borrow_cost: float = 0.0,
) -> list[dict]:
    """Fair value term structure across multiple expiries.

    Args:
        spot: current index spot level.
        rate: continuously compounded risk-free rate.
        div_yield: continuous dividend yield.
        expiries_years: list of expiry tenors in years.
        borrow_cost: annualised stock borrow cost (default 0).

    Returns:
        List of dicts, each containing ``T`` and the fields of
        :class:`IndexFuturesResult`.
    """
    results = []
    for T in expiries_years:
        row = index_futures_fair_value(spot, rate, div_yield, T, borrow_cost).to_dict()
        row["T"] = T
        results.append(row)
    return results
