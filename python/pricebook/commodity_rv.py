"""Commodity rich/cheap analysis and roll strategies.

* :func:`spread_zscore` — z-score a calendar spread level vs history.
* :func:`ratio_monitor` — inter-commodity ratio (gold/silver, Brent/WTI, …) vs history.
* :func:`seasonality_monitor` — compare current level to seasonal norm.
* :func:`optimal_roll_date` — pick the roll date that minimises cost.
* :func:`roll_pnl` — P&L from rolling a position from one delivery to the next.
* :func:`roll_cost_or_gain` — classify roll as cost (contango) or gain (backwardation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.zscore import zscore as _zscore_impl, ZScoreSignal


def _zscore_and_signal(
    current: float,
    history: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    return _zscore_impl(current, history, threshold)


# ---- Step 1: RV analysis ----

def spread_zscore(
    current_spread: float,
    history: list[float],
    threshold: float = 2.0,
) -> ZScoreSignal:
    """Z-score a calendar spread level against its history."""
    return _zscore_and_signal(current_spread, history, threshold)


@dataclass
class RatioLevel:
    """Inter-commodity ratio monitor."""
    numerator: str
    denominator: str
    ratio: float
    mean: float
    std: float
    z_score: float | None
    percentile: float | None
    signal: str


def ratio_monitor(
    numerator: str,
    denominator: str,
    numerator_price: float,
    denominator_price: float,
    history: list[float],
    threshold: float = 2.0,
) -> RatioLevel:
    """Monitor an inter-commodity price ratio vs history.

    E.g. gold/silver ratio, Brent/WTI spread ratio, gas/coal ratio.
    """
    ratio = numerator_price / denominator_price if denominator_price != 0 else 0.0
    z = _zscore_and_signal(ratio, history, threshold)
    return RatioLevel(
        numerator=numerator,
        denominator=denominator,
        ratio=ratio,
        mean=z.mean,
        std=z.std,
        z_score=z.z_score,
        percentile=z.percentile,
        signal=z.signal,
    )


@dataclass
class SeasonalLevel:
    """Seasonality monitor result."""
    commodity: str
    month: int
    current_price: float
    seasonal_norm: float
    deviation: float
    z_score: float | None
    signal: str


def seasonality_monitor(
    commodity: str,
    month: int,
    current_price: float,
    seasonal_norms: dict[int, float],
    history_deviations: list[float] | None = None,
    threshold: float = 2.0,
) -> SeasonalLevel:
    """Compare the current price to its seasonal norm for the given month.

    ``seasonal_norms`` maps month (1–12) to the historical average price
    for that month. ``deviation = current - norm``.
    """
    norm = seasonal_norms.get(month, current_price)
    deviation = current_price - norm
    z = _zscore_and_signal(deviation, history_deviations or [], threshold)
    return SeasonalLevel(
        commodity=commodity,
        month=month,
        current_price=current_price,
        seasonal_norm=norm,
        deviation=deviation,
        z_score=z.z_score,
        signal=z.signal,
    )


# ---- Step 2: roll strategies ----

@dataclass
class RollCandidate:
    """One possible roll date with associated cost/gain."""
    roll_date: date
    old_delivery: date
    new_delivery: date
    old_price: float
    new_price: float
    roll_spread: float    # old − new (positive = backwardation gain)
    roll_cost: float      # −roll_spread × quantity (positive = cost)


def roll_pnl(
    old_price: float,
    new_price: float,
    quantity: float,
    direction: int = 1,
) -> float:
    """P&L from rolling a position from one delivery to the next.

    For a long position (direction=+1): sell old at old_price, buy new
    at new_price. P&L = direction × quantity × (old_price − new_price).
    """
    return direction * quantity * (old_price - new_price)


def roll_cost_or_gain(old_price: float, new_price: float) -> str:
    """Classify the roll as contango (cost) or backwardation (gain).

    For a long position rolling forward:
    - Contango (new > old): sell low, buy high → cost.
    - Backwardation (new < old): sell high, buy low → gain.
    """
    if new_price > old_price:
        return "contango_cost"
    elif new_price < old_price:
        return "backwardation_gain"
    return "flat"


def optimal_roll_date(
    candidates: list[RollCandidate],
) -> RollCandidate | None:
    """Pick the roll date that minimises roll cost (maximises gain).

    Among all candidates, selects the one with the highest roll_spread
    (most negative cost or most positive gain for a long position).
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.roll_spread)


def track_roll_pnl(
    rolls: list[tuple[float, float, float]],
) -> float:
    """Cumulative roll P&L from a sequence of rolls.

    Args:
        rolls: list of (old_price, new_price, quantity) tuples.

    Returns:
        Total roll P&L = Σ quantity × (old − new).
    """
    return sum(qty * (old - new) for old, new, qty in rolls)
