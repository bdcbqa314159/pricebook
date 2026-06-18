"""Power/electricity derivatives: swing, tolling, capacity options.

Electricity-specific products that differ from standard commodity
derivatives due to non-storability and hourly price variation.

* :func:`swing_option_price` — swing option with volume flexibility.
* :func:`tolling_agreement` — virtual power plant tolling economics.
* :func:`capacity_option` — option on generation capacity.
* :func:`block_forward` — peak/off-peak block forward pricing.

References:
    Eydeland & Wolyniec, *Energy and Power Risk Management*, Wiley, 2003.
    Geman, *Commodities and Commodity Derivatives*, Ch. 10, 2005.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SwingOptionResult:
    """Swing option pricing result."""
    price: float
    expected_exercises: float
    min_take: int
    max_take: int
    total_periods: int
    avg_exercise_price: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def swing_option_price(
    forward_prices: list[float],
    strike: float,
    min_take: int,
    max_take: int,
    vol: float = 0.40,
    rate: float = 0.04,
    n_sims: int = 20_000,
    seed: int = 42,
) -> SwingOptionResult:
    """Price a power swing option with volume flexibility.

    The holder can exercise on any subset of periods, subject to
    minimum and maximum total take constraints.

    At each period: payoff = max(S_i − K, 0) if exercised.
    Optimal strategy: exercise on the max_take periods with
    highest positive intrinsic value.

    Args:
        forward_prices: forward price per delivery period.
        strike: exercise price.
        min_take: minimum number of periods to exercise.
        max_take: maximum number of periods to exercise.
        vol: price volatility.
    """
    n = len(forward_prices)
    rng = np.random.default_rng(seed)

    total_pv = 0.0
    total_exercises = 0
    total_ex_price = 0.0

    for _ in range(n_sims):
        # Simulate prices
        sim_prices = np.array([
            F * math.exp(-0.5 * vol**2 + vol * rng.standard_normal())
            for F in forward_prices
        ])

        # Intrinsic values
        intrinsics = np.maximum(sim_prices - strike, 0)

        # Optimal exercise: pick top max_take periods by intrinsic
        sorted_idx = np.argsort(-intrinsics)
        n_positive = int(np.sum(intrinsics > 0))

        # Must exercise at least min_take, at most max_take
        n_exercise = max(min_take, min(max_take, n_positive))
        exercise_idx = sorted_idx[:n_exercise]

        payoff = sum(intrinsics[i] for i in exercise_idx)

        # Discount (assume evenly spaced periods over 1 year)
        avg_df = math.exp(-rate * 0.5)
        total_pv += payoff * avg_df
        total_exercises += n_exercise
        total_ex_price += sum(sim_prices[i] for i in exercise_idx)

    price = total_pv / n_sims
    avg_ex = total_exercises / n_sims
    avg_price = total_ex_price / max(total_exercises, 1)

    return SwingOptionResult(
        price=price,
        expected_exercises=avg_ex,
        min_take=min_take,
        max_take=max_take,
        total_periods=n,
        avg_exercise_price=avg_price,
    )


@dataclass
class TollingResult:
    """Tolling agreement valuation."""
    value: float                # NPV of tolling agreement
    expected_generation_hours: float
    heat_rate: float
    spark_spread_avg: float
    capacity_mw: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def tolling_agreement(
    power_forwards: list[float],
    gas_forwards: list[float],
    heat_rate: float = 7.0,
    capacity_mw: float = 100.0,
    vom_cost: float = 3.0,
    rate: float = 0.04,
    hours_per_period: float = 730.0,
) -> TollingResult:
    """Value a virtual power plant tolling agreement.

    The toll holder pays gas + VOM and receives power output.
    Profit per period = max(0, power_price − heat_rate × gas_price − VOM) × MW × hours.

    Args:
        power_forwards: power price per period ($/MWh).
        gas_forwards: natural gas price per period ($/MMBtu).
        heat_rate: MMBtu per MWh (efficiency).
        capacity_mw: plant capacity in MW.
        vom_cost: variable O&M cost ($/MWh).
        hours_per_period: hours per delivery period.
    """
    n = len(power_forwards)
    total_value = 0.0
    gen_hours = 0.0
    spreads = []

    for i, (pw, gas) in enumerate(zip(power_forwards, gas_forwards)):
        spark = pw - heat_rate * gas - vom_cost
        spreads.append(spark)

        if spark > 0:
            # Generate
            period_value = spark * capacity_mw * hours_per_period
            gen_hours += hours_per_period
        else:
            period_value = 0.0

        t = (i + 0.5) / max(n, 1)
        df = math.exp(-rate * t)
        total_value += period_value * df

    return TollingResult(
        value=total_value,
        expected_generation_hours=gen_hours,
        heat_rate=heat_rate,
        spark_spread_avg=sum(spreads) / len(spreads) if spreads else 0,
        capacity_mw=capacity_mw,
    )


@dataclass
class CapacityOptionResult:
    """Capacity option pricing result."""
    price: float
    expected_dispatch_hours: float
    capacity_mw: float
    strike: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def capacity_option(
    forward_prices: list[float],
    strike: float,
    capacity_mw: float = 100.0,
    vol: float = 0.50,
    rate: float = 0.04,
    hours_per_period: float = 730.0,
    n_sims: int = 20_000,
    seed: int = 42,
) -> CapacityOptionResult:
    """Option on generation capacity (call on power price).

    The holder can dispatch the plant when profitable.
    payoff = Σ max(S_i − K, 0) × MW × hours × df.

    Similar to a strip of call options on power.

    Args:
        forward_prices: power forwards per period.
        strike: dispatch cost (fuel + VOM, $/MWh).
        capacity_mw: plant capacity.
    """
    n = len(forward_prices)
    rng = np.random.default_rng(seed)

    total_pv = 0.0
    total_hours = 0.0

    for _ in range(n_sims):
        path_pv = 0.0
        dispatch = 0.0
        for i, F in enumerate(forward_prices):
            S = F * math.exp(-0.5 * vol**2 + vol * rng.standard_normal())
            if S > strike:
                payoff = (S - strike) * capacity_mw * hours_per_period
                dispatch += hours_per_period
            else:
                payoff = 0.0
            t = (i + 0.5) / max(n, 1)
            path_pv += payoff * math.exp(-rate * t)

        total_pv += path_pv
        total_hours += dispatch

    return CapacityOptionResult(
        price=total_pv / n_sims,
        expected_dispatch_hours=total_hours / n_sims,
        capacity_mw=capacity_mw,
        strike=strike,
    )


@dataclass
class BlockForwardResult:
    """Block forward pricing result."""
    peak_price: float
    off_peak_price: float
    base_price: float           # flat 24h price
    peak_premium: float         # peak vs off-peak

    def to_dict(self) -> dict:
        return dict(vars(self))


def block_forward(
    hourly_forwards: list[float],
    peak_hours: tuple[int, int] = (7, 22),
) -> BlockForwardResult:
    """Peak/off-peak block forward pricing.

    Splits 24-hour forward curve into peak and off-peak blocks.

    Args:
        hourly_forwards: 24 hourly forward prices.
        peak_hours: (start_hour, end_hour) for peak definition.
    """
    if len(hourly_forwards) != 24:
        # Pad or truncate
        hourly_forwards = (hourly_forwards * 2)[:24]

    peak_start, peak_end = peak_hours
    peak_prices = [hourly_forwards[h] for h in range(24) if peak_start <= h < peak_end]
    off_peak_prices = [hourly_forwards[h] for h in range(24) if not (peak_start <= h < peak_end)]

    peak = sum(peak_prices) / len(peak_prices) if peak_prices else 0
    off_peak = sum(off_peak_prices) / len(off_peak_prices) if off_peak_prices else 0
    base = sum(hourly_forwards) / 24

    return BlockForwardResult(
        peak_price=peak,
        off_peak_price=off_peak,
        base_price=base,
        peak_premium=peak - off_peak,
    )
