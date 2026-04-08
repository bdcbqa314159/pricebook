"""Rich/cheap analysis: relative value, z-scores, and spread monitors.

Compare market swap rates to model-implied rates, compute z-scores,
and monitor spread/butterfly levels for trading signals.

    from pricebook.rich_cheap import (
        relative_value, spread_monitor, butterfly_monitor,
    )

    rv = relative_value(market_rate=0.051, model_rate=0.050,
                        history=[0.0005, -0.0003, 0.0008])
    print(rv.spread, rv.z_score, rv.percentile)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.discount_curve import DiscountCurve
from pricebook.swap import InterestRateSwap, SwapDirection


# ---- Relative value ----

@dataclass
class RelativeValueResult:
    """Result of a relative value analysis."""
    market_rate: float
    model_rate: float
    spread: float
    z_score: float | None
    percentile: float | None
    signal: str  # "rich", "cheap", or "fair"


def relative_value(
    market_rate: float,
    model_rate: float,
    history: list[float] | None = None,
    threshold: float = 2.0,
) -> RelativeValueResult:
    """Compare a market swap rate to a model-implied rate.

    Args:
        market_rate: observed market rate.
        model_rate: fair rate from curve model.
        history: historical spreads for z-score and percentile.
        threshold: z-score threshold for rich/cheap signal.
    """
    spread = market_rate - model_rate

    z_score = None
    percentile = None
    if history and len(history) >= 2:
        mean = sum(history) / len(history)
        var = sum((h - mean) ** 2 for h in history) / len(history)
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 1e-12:
            z_score = (spread - mean) / std
        sorted_hist = sorted(history)
        rank = sum(1 for h in sorted_hist if h <= spread)
        percentile = rank / len(sorted_hist) * 100.0

    if z_score is not None and abs(z_score) >= threshold:
        signal = "rich" if z_score > 0 else "cheap"
    else:
        signal = "fair"

    return RelativeValueResult(
        market_rate=market_rate,
        model_rate=model_rate,
        spread=spread,
        z_score=z_score,
        percentile=percentile,
        signal=signal,
    )


def rv_from_curve(
    curve: DiscountCurve,
    start: date,
    tenor_years: int,
    market_rate: float,
    history: list[float] | None = None,
    threshold: float = 2.0,
) -> RelativeValueResult:
    """Relative value using model par rate from a curve."""
    end = date(start.year + tenor_years, start.month, start.day)
    swap = InterestRateSwap(
        start, end, fixed_rate=0.05,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    model_rate = swap.par_rate(curve)
    return relative_value(market_rate, model_rate, history, threshold)


# ---- Spread monitor ----

@dataclass
class SpreadLevel:
    """Current spread level with signal."""
    name: str
    short_tenor: int
    long_tenor: int
    short_rate: float
    long_rate: float
    spread: float
    z_score: float | None
    signal: str


def spread_monitor(
    curve: DiscountCurve,
    start: date,
    short_tenor: int,
    long_tenor: int,
    history: list[float] | None = None,
    threshold: float = 2.0,
) -> SpreadLevel:
    """Monitor a spread (e.g. 2s10s).

    Spread = long_rate - short_rate.
    """
    short_rate = _par_rate(curve, start, short_tenor)
    long_rate = _par_rate(curve, start, long_tenor)
    spread = long_rate - short_rate

    z_score = None
    signal = "fair"
    if history and len(history) >= 2:
        mean = sum(history) / len(history)
        var = sum((h - mean) ** 2 for h in history) / len(history)
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 1e-12:
            z_score = (spread - mean) / std
            if abs(z_score) >= threshold:
                signal = "wide" if z_score > 0 else "tight"

    return SpreadLevel(
        name=f"{short_tenor}s{long_tenor}s",
        short_tenor=short_tenor,
        long_tenor=long_tenor,
        short_rate=short_rate,
        long_rate=long_rate,
        spread=spread,
        z_score=z_score,
        signal=signal,
    )


# ---- Butterfly monitor ----

@dataclass
class ButterflyLevel:
    """Current butterfly level with signal."""
    name: str
    short_tenor: int
    belly_tenor: int
    long_tenor: int
    short_rate: float
    belly_rate: float
    long_rate: float
    butterfly: float
    z_score: float | None
    signal: str


def butterfly_monitor(
    curve: DiscountCurve,
    start: date,
    short_tenor: int,
    belly_tenor: int,
    long_tenor: int,
    history: list[float] | None = None,
    threshold: float = 2.0,
) -> ButterflyLevel:
    """Monitor a butterfly (e.g. 2s5s10s).

    Butterfly = (short_rate + long_rate) / 2 - belly_rate.
    Positive = belly cheap, negative = belly rich.
    """
    short_rate = _par_rate(curve, start, short_tenor)
    belly_rate = _par_rate(curve, start, belly_tenor)
    long_rate = _par_rate(curve, start, long_tenor)
    butterfly = (short_rate + long_rate) / 2.0 - belly_rate

    z_score = None
    signal = "fair"
    if history and len(history) >= 2:
        mean = sum(history) / len(history)
        var = sum((h - mean) ** 2 for h in history) / len(history)
        std = math.sqrt(var) if var > 0 else 0.0
        if std > 1e-12:
            z_score = (butterfly - mean) / std
            if abs(z_score) >= threshold:
                signal = "belly_cheap" if z_score > 0 else "belly_rich"

    return ButterflyLevel(
        name=f"{short_tenor}s{belly_tenor}s{long_tenor}s",
        short_tenor=short_tenor,
        belly_tenor=belly_tenor,
        long_tenor=long_tenor,
        short_rate=short_rate,
        belly_rate=belly_rate,
        long_rate=long_rate,
        butterfly=butterfly,
        z_score=z_score,
        signal=signal,
    )


# ---- Helpers ----

def _par_rate(curve: DiscountCurve, start: date, tenor_years: int) -> float:
    end = date(start.year + tenor_years, start.month, start.day)
    swap = InterestRateSwap(
        start, end, fixed_rate=0.05,
        direction=SwapDirection.PAYER, notional=1_000_000.0,
    )
    return swap.par_rate(curve)
