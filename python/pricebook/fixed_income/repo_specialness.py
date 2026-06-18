"""Specialness analytics — multi-market forecasting and supply/demand signals.

    from pricebook.fixed_income.repo_specialness import (
        forecast_specialness, specialness_term_structure,
        SpecialnessConventions, get_specialness_conventions,
    )

References:
    Duffie (1996). Special Repo Rates. Journal of Finance.
    Krishnamurthy (2002). The Bond/Old-Bond Spread. JFE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pricebook.core.serialisable import serialisable_convention
from datetime import date

import numpy as np


@serialisable_convention("specialness_conventions")
@dataclass(frozen=True)
class SpecialnessConventions:
    """Per-market specialness conventions."""
    market: str                  # UST, Bund, Gilt, JGB, OAT, BTP
    on_the_run_tenors: list[str] # e.g. ["2Y", "5Y", "10Y", "30Y"]
    auction_frequency: str       # "monthly", "quarterly"
    typical_special_range_bp: tuple[float, float]  # (min, max) typical special spread
    settlement_days: int


_CONVENTIONS: dict[str, SpecialnessConventions] = {
    "UST": SpecialnessConventions("UST", ["2Y", "3Y", "5Y", "7Y", "10Y", "20Y", "30Y"],
                                   "monthly", (5, 200), 1),
    "BUND": SpecialnessConventions("BUND", ["2Y", "5Y", "10Y", "30Y"],
                                    "monthly", (5, 100), 2),
    "GILT": SpecialnessConventions("GILT", ["2Y", "5Y", "10Y", "30Y", "50Y"],
                                    "monthly", (5, 150), 1),
    "JGB": SpecialnessConventions("JGB", ["2Y", "5Y", "10Y", "20Y", "30Y", "40Y"],
                                   "monthly", (2, 50), 2),
    "OAT": SpecialnessConventions("OAT", ["2Y", "5Y", "10Y", "30Y"],
                                   "monthly", (5, 80), 2),
    "BTP": SpecialnessConventions("BTP", ["3Y", "5Y", "7Y", "10Y", "15Y", "30Y"],
                                   "monthly", (10, 150), 2),
}

from pricebook.core.data_registry import load_registry as _load_reg
_CONVENTIONS = _load_reg("repo_specialness.json", SpecialnessConventions, lambda c: c.market, _CONVENTIONS)


def get_specialness_conventions(market: str) -> SpecialnessConventions:
    key = market.upper()
    conv = _CONVENTIONS.get(key)
    if conv is None:
        available = sorted(_CONVENTIONS.keys())
        raise ValueError(f"No specialness conventions for {key!r}. Available: {available}")
    return conv


def list_specialness_markets() -> list[str]:
    return sorted(_CONVENTIONS.keys())


@dataclass
class SpecialnessForecast:
    """Result of specialness forecast."""
    bond_id: str
    market: str
    current_special_bp: float
    forecast_special_bp: float
    mean_reversion_target_bp: float
    half_life_days: float
    signal: str                  # "RICH_SPECIAL", "CHEAP_SPECIAL", "FAIR"
    z_score: float

    def to_dict(self) -> dict:
        return dict(vars(self))


def forecast_specialness(
    bond_id: str,
    market: str,
    current_special_bp: float,
    historical_mean_bp: float = 20.0,
    historical_std_bp: float = 15.0,
    mean_reversion_speed: float = 0.05,
    days_to_auction: int | None = None,
    horizon_days: int = 5,
) -> SpecialnessForecast:
    """Forecast specialness using mean-reversion + auction-cycle.

    Model: ds = κ(μ - s)dt + auction_effect
    Forecast: s(t+h) ≈ s(t) + κ(μ - s(t))×h + auction_bump

    Args:
        current_special_bp: current GC-special spread (bp, positive = on special).
        historical_mean_bp: long-run average special spread.
        historical_std_bp: historical std of special spread.
        mean_reversion_speed: κ (per day).
        days_to_auction: days until next auction (drives squeeze).
        horizon_days: forecast horizon.
    """
    # Mean-reversion forecast
    reversion = mean_reversion_speed * (historical_mean_bp - current_special_bp) * horizon_days
    forecast = current_special_bp + reversion

    # Auction effect: specialness tends to increase before auction, normalise after
    if days_to_auction is not None and days_to_auction > 0:
        if days_to_auction <= 5:
            # Pre-auction squeeze: specialness increases
            auction_bump = 10.0 * (1 - days_to_auction / 5)
        elif days_to_auction <= 10:
            auction_bump = 0.0
        else:
            auction_bump = 0.0
        forecast += auction_bump

    forecast = max(forecast, 0.0)

    # Half-life of mean reversion
    half_life = math.log(2) / max(mean_reversion_speed, 1e-10)

    # Z-score
    z = (current_special_bp - historical_mean_bp) / max(historical_std_bp, 1.0)

    # Signal
    if z > 1.5:
        signal = "RICH_SPECIAL"  # very special, may normalise
    elif z < -0.5:
        signal = "CHEAP_SPECIAL"  # unusually low, may increase
    else:
        signal = "FAIR"

    return SpecialnessForecast(
        bond_id=bond_id, market=market,
        current_special_bp=current_special_bp,
        forecast_special_bp=forecast,
        mean_reversion_target_bp=historical_mean_bp,
        half_life_days=half_life,
        signal=signal, z_score=z,
    )


def specialness_term_structure(
    gc_rates: dict[int, float],
    special_rates: dict[int, float],
) -> list[dict]:
    """GC-special spread term structure.

    Args:
        gc_rates: {tenor_days: gc_rate}.
        special_rates: {tenor_days: special_rate}.

    Returns list of {tenor_days, gc_rate, special_rate, spread_bp}.
    """
    tenors = sorted(set(gc_rates.keys()) & set(special_rates.keys()))
    return [
        {
            "tenor_days": t,
            "gc_rate": gc_rates[t],
            "special_rate": special_rates[t],
            "spread_bp": (gc_rates[t] - special_rates[t]) * 10_000,
        }
        for t in tenors
    ]


def supply_demand_indicator(
    fail_rate_pct: float,
    on_the_run: bool,
    days_since_auction: int,
    outstanding_bn: float,
    short_interest_pct: float = 0.0,
) -> dict:
    """Supply/demand signal from market microstructure.

    Higher fail rate + on-the-run + recent auction + high short interest
    → higher likelihood of going on special.
    """
    score = 0.0
    if fail_rate_pct > 3.0:
        score += 2.0
    elif fail_rate_pct > 1.0:
        score += 1.0

    if on_the_run:
        score += 1.5

    if days_since_auction < 7:
        score += 1.0

    if outstanding_bn < 30:
        score += 0.5  # small issue → easier to squeeze

    score += min(short_interest_pct / 10, 2.0)

    if score >= 4:
        signal = "HIGH_SPECIAL_RISK"
    elif score >= 2:
        signal = "MODERATE"
    else:
        signal = "LOW"

    return {
        "score": score,
        "signal": signal,
        "fail_rate_pct": fail_rate_pct,
        "on_the_run": on_the_run,
        "days_since_auction": days_since_auction,
    }
