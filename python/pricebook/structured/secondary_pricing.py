"""Secondary market structured product pricing.

Spread aging for CLNs, mark-to-bid for illiquid structures,
stale pricing detection, and liquidity premium modelling.

* :func:`spread_aging` — adjust CLN spread for time since issuance.
* :func:`mark_to_bid` — haircut mid-market for illiquidity.
* :func:`stale_price_detector` — flag stale/unchanged prices.
* :func:`liquidity_premium` — model-based illiquidity premium.

References:
    Longstaff, *How Much Can Marketability Affect Security Values?*,
    JF, 1995.
    Koziol & Sauerbier, *Valuation of Bond Illiquidity*, JBF, 2007.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class SpreadAgingResult:
    """Spread aging result for a structured note."""
    original_spread_bp: float
    aged_spread_bp: float
    aging_adjustment_bp: float
    age_years: float
    remaining_years: float
    pull_to_par_effect: float

    def to_dict(self) -> dict:
        return vars(self)


def spread_aging(
    original_spread_bp: float,
    original_maturity: float,
    age_years: float,
    current_market_spread_bp: float | None = None,
    credit_migration_bp: float = 0.0,
) -> SpreadAgingResult:
    """Adjust CLN/structured note spread for time since issuance.

    As a note ages:
    1. Pull-to-par reduces the spread impact on price.
    2. Credit migration may have changed the reference spread.
    3. Market conditions (current new-issue spreads) may differ.

    aged_spread = original_spread + credit_migration + market_adjustment

    The market_adjustment captures the difference between the
    note's original spread and current new-issue levels.

    Args:
        original_spread_bp: spread at issuance.
        original_maturity: original maturity in years.
        age_years: years since issuance.
        current_market_spread_bp: current new-issue spread for comparable.
        credit_migration_bp: spread change from credit migration (positive = wider).
    """
    remaining = max(original_maturity - age_years, 0.01)

    # Pull-to-par: spread impact on price decreases as maturity shortens
    pull_to_par = remaining / original_maturity if original_maturity > 0 else 1.0

    # Market adjustment
    if current_market_spread_bp is not None:
        market_adj = (current_market_spread_bp - original_spread_bp) * 0.5
    else:
        market_adj = 0.0

    aged = original_spread_bp + credit_migration_bp + market_adj
    adjustment = aged - original_spread_bp

    return SpreadAgingResult(
        original_spread_bp=original_spread_bp,
        aged_spread_bp=aged,
        aging_adjustment_bp=adjustment,
        age_years=age_years,
        remaining_years=remaining,
        pull_to_par_effect=pull_to_par,
    )


@dataclass
class MarkToBidResult:
    """Mark-to-bid result."""
    mid_price: float
    bid_price: float
    haircut_pct: float
    liquidity_score: float      # 0 (illiquid) to 1 (liquid)

    def to_dict(self) -> dict:
        return vars(self)


def mark_to_bid(
    mid_price: float,
    bid_ask_spread_pct: float = 2.0,
    liquidity_score: float = 0.5,
    stress_multiplier: float = 1.0,
) -> MarkToBidResult:
    """Apply bid-side haircut to mid-market price.

    bid = mid × (1 − haircut)
    haircut = bid_ask/2 × (2 − liquidity_score) × stress

    More illiquid structures get larger haircuts.

    Args:
        mid_price: mid-market price.
        bid_ask_spread_pct: typical bid-ask as % of price.
        liquidity_score: 0 (illiquid) to 1 (liquid).
        stress_multiplier: widen haircut in stress (>1).
    """
    half_spread = bid_ask_spread_pct / 2.0
    illiquidity_factor = 2.0 - liquidity_score  # 1 (liquid) to 2 (illiquid)
    haircut = half_spread * illiquidity_factor * stress_multiplier / 100.0
    bid = mid_price * (1 - haircut)

    return MarkToBidResult(
        mid_price=mid_price,
        bid_price=bid,
        haircut_pct=haircut * 100,
        liquidity_score=liquidity_score,
    )


@dataclass
class StalePriceResult:
    """Stale price detection result."""
    is_stale: bool
    days_unchanged: int
    total_days: int
    staleness_pct: float
    recommendation: str

    def to_dict(self) -> dict:
        return vars(self)


def stale_price_detector(
    prices: list[float],
    threshold_days: int = 5,
    tolerance: float = 0.001,
) -> StalePriceResult:
    """Detect stale/unchanged prices in a time series.

    A price is stale if it hasn't changed for threshold_days
    or more consecutive observations.

    Args:
        prices: daily price observations.
        threshold_days: consecutive unchanged days to flag.
        tolerance: relative tolerance for "unchanged".
    """
    if len(prices) < 2:
        return StalePriceResult(False, 0, len(prices), 0, "insufficient data")

    consecutive = 0
    max_consecutive = 0
    total_unchanged = 0

    for i in range(1, len(prices)):
        if abs(prices[i] - prices[i - 1]) / max(abs(prices[i - 1]), 1e-10) < tolerance:
            consecutive += 1
            total_unchanged += 1
        else:
            max_consecutive = max(max_consecutive, consecutive)
            consecutive = 0

    max_consecutive = max(max_consecutive, consecutive)
    staleness = total_unchanged / (len(prices) - 1) * 100

    is_stale = max_consecutive >= threshold_days

    if is_stale:
        rec = f"price unchanged for {max_consecutive} days — request fresh quote"
    elif staleness > 50:
        rec = "high staleness — consider alternative pricing source"
    else:
        rec = "pricing appears active"

    return StalePriceResult(
        is_stale=is_stale,
        days_unchanged=max_consecutive,
        total_days=len(prices),
        staleness_pct=staleness,
        recommendation=rec,
    )


@dataclass
class LiquidityPremiumResult:
    """Liquidity premium model result."""
    premium_bp: float
    bid_ask_component: float
    holding_period_component: float
    credit_component: float

    def to_dict(self) -> dict:
        return vars(self)


def liquidity_premium(
    bid_ask_bp: float = 50.0,
    holding_period_years: float = 1.0,
    turnover_ratio: float = 0.2,
    credit_spread_bp: float = 100.0,
    rating_notch: int = 5,
) -> LiquidityPremiumResult:
    """Model-based illiquidity premium estimation.

    premium = bid_ask_component + holding_period + credit_adjustment

    bid_ask: direct cost of round-trip.
    holding: opportunity cost of being locked in.
    credit: illiquid credit trades wider than liquid.

    Args:
        bid_ask_bp: typical bid-ask spread in bp.
        holding_period_years: expected holding period.
        turnover_ratio: annual turnover (0-1).
        credit_spread_bp: credit spread of the instrument.
        rating_notch: 1 (AAA) to 10 (CCC), higher = more illiquid.
    """
    # Bid-ask component: amortised over holding period
    ba_component = bid_ask_bp / max(holding_period_years, 0.1) * 0.5

    # Holding period: longer hold = more compensation needed
    hp_component = 10 * math.sqrt(holding_period_years) * (1 - turnover_ratio)

    # Credit: illiquid credit instruments trade wider
    credit_factor = 1.0 + 0.05 * rating_notch
    credit_component = credit_spread_bp * 0.10 * credit_factor

    total = ba_component + hp_component + credit_component

    return LiquidityPremiumResult(
        premium_bp=total,
        bid_ask_component=ba_component,
        holding_period_component=hp_component,
        credit_component=credit_component,
    )
