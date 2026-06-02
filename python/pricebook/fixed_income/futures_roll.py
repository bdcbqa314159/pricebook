"""Futures roll mechanics: auto-roll calendar, slippage, liquidity.

* :class:`RollSchedule` — roll calendar with dates and contracts.
* :func:`generate_roll_schedule` — create roll schedule for a product.
* :func:`roll_adjusted_returns` — compute roll-adjusted return series.
* :func:`roll_slippage` — estimate slippage from bid-ask.
* :func:`liquidity_curve` — volume by contract month.

References:
    Gorton & Rouwenhorst, *Facts and Fantasies about Commodity Futures*,
    Financial Analysts Journal, 2006.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np


@dataclass
class RollEvent:
    """Single roll event."""
    roll_date: date
    from_contract: str
    to_contract: str
    from_price: float
    to_price: float
    roll_cost: float        # to_price - from_price
    slippage: float

    def to_dict(self) -> dict:
        return {
            "roll_date": self.roll_date.isoformat(),
            "from": self.from_contract,
            "to": self.to_contract,
            "roll_cost": self.roll_cost,
            "slippage": self.slippage,
        }


@dataclass
class RollSchedule:
    """Complete roll schedule."""
    events: list[RollEvent]
    total_roll_cost: float
    total_slippage: float
    n_rolls: int

    def to_dict(self) -> dict:
        return {
            "n_rolls": self.n_rolls,
            "total_roll_cost": self.total_roll_cost,
            "total_slippage": self.total_slippage,
            "events": [e.to_dict() for e in self.events],
        }


def generate_roll_schedule(
    start_date: date,
    end_date: date,
    contract_months: list[int],
    front_prices: list[float],
    back_prices: list[float],
    ticker: str = "CL",
    roll_days_before_expiry: int = 5,
    bid_ask_spread: float = 0.02,
) -> RollSchedule:
    """Generate a roll schedule for a futures product.

    Args:
        start_date: start of period.
        end_date: end of period.
        contract_months: delivery months (e.g., [3, 6, 9, 12] for quarterly).
        front_prices: front-month prices at each roll.
        back_prices: back-month prices at each roll.
        ticker: product ticker.
        roll_days_before_expiry: days before expiry to roll.
        bid_ask_spread: typical bid-ask spread.
    """
    events = []
    current = start_date
    idx = 0

    while current < end_date and idx < len(front_prices):
        # Next expiry month
        month_idx = idx % len(contract_months)
        year_offset = idx // len(contract_months)
        exp_month = contract_months[month_idx]
        exp_year = start_date.year + year_offset

        # Roll date: roll_days_before_expiry before month end
        try:
            roll_date = date(exp_year, exp_month, 15) - timedelta(days=roll_days_before_expiry)
        except ValueError:
            idx += 1
            continue

        if roll_date < start_date or roll_date > end_date:
            idx += 1
            continue

        fp = front_prices[min(idx, len(front_prices) - 1)]
        bp = back_prices[min(idx, len(back_prices) - 1)]
        slippage = bid_ask_spread / 2  # half spread on each leg

        events.append(RollEvent(
            roll_date=roll_date,
            from_contract=f"{ticker}{exp_month:02d}",
            to_contract=f"{ticker}{contract_months[(month_idx + 1) % len(contract_months)]:02d}",
            from_price=fp,
            to_price=bp,
            roll_cost=bp - fp,
            slippage=slippage,
        ))

        current = roll_date + timedelta(days=1)
        idx += 1

    total_cost = sum(e.roll_cost for e in events)
    total_slip = sum(e.slippage for e in events)

    return RollSchedule(
        events=events,
        total_roll_cost=total_cost,
        total_slippage=total_slip,
        n_rolls=len(events),
    )


def roll_adjusted_returns(
    prices: list[float],
    roll_adjustments: list[float],
    roll_indices: list[int],
) -> np.ndarray:
    """Compute roll-adjusted return series.

    Subtracts roll cost from raw returns at each roll date,
    creating a continuous return series.

    Args:
        prices: daily settlement prices.
        roll_adjustments: roll cost at each roll.
        roll_indices: index in prices where each roll occurs.

    Returns:
        Roll-adjusted cumulative return series.
    """
    n = len(prices)
    returns = np.zeros(n)

    for i in range(1, n):
        raw_return = (prices[i] - prices[i - 1]) / prices[i - 1]
        returns[i] = raw_return

    # Subtract roll cost at roll dates
    for adj, idx in zip(roll_adjustments, roll_indices):
        if 0 < idx < n and prices[idx - 1] > 0:
            returns[idx] -= adj / prices[idx - 1]

    # Cumulative
    cum = np.cumprod(1 + returns)
    return cum


def roll_slippage(
    bid_ask_spread: float,
    n_contracts: int,
    market_depth: float = 100.0,
) -> float:
    """Estimate roll slippage.

    Slippage = half-spread + market impact.
    Market impact ∝ sqrt(n_contracts / market_depth).

    Args:
        bid_ask_spread: typical bid-ask spread in price.
        n_contracts: number of contracts to roll.
        market_depth: average depth (contracts at best bid/ask).
    """
    half_spread = bid_ask_spread / 2
    impact = half_spread * math.sqrt(n_contracts / max(market_depth, 1))
    return half_spread + impact


def liquidity_curve(
    volumes: list[float],
    contract_labels: list[str],
) -> list[dict]:
    """Analyse volume distribution across contract months.

    Args:
        volumes: daily volume per contract month.
        contract_labels: contract identifiers (e.g., "CLZ24").

    Returns:
        List of dicts with volume and % of total.
    """
    total = sum(volumes)
    result = []
    for label, vol in zip(contract_labels, volumes):
        result.append({
            "contract": label,
            "volume": vol,
            "pct_of_total": vol / total * 100 if total > 0 else 0,
        })
    return sorted(result, key=lambda x: -x["volume"])
