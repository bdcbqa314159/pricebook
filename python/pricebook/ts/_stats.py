"""Statistics: mean, vol, sharpe, sortino, drawdown — delegates to backtest.py."""

from __future__ import annotations

import math

import numpy as np

from pricebook.ts._core import TimeSeries


def mean(ts: TimeSeries) -> float:
    """Arithmetic mean of values."""
    if len(ts) == 0:
        return 0.0
    return float(np.nanmean(ts.values))


def vol(ts: TimeSeries, annualise: int = 252) -> float:
    """Annualised volatility."""
    if len(ts) < 2:
        return 0.0
    return float(np.nanstd(ts.values)) * math.sqrt(annualise)


def sharpe(ts: TimeSeries, annualise: int = 252) -> float:
    """Annualised Sharpe ratio (assumes zero risk-free rate)."""
    if len(ts) < 2:
        return 0.0
    mu = float(np.nanmean(ts.values))
    sigma = float(np.nanstd(ts.values))
    if sigma < 1e-10:
        return 0.0
    return mu / sigma * math.sqrt(annualise)


def sortino(ts: TimeSeries, annualise: int = 252) -> float:
    """Annualised Sortino ratio (downside deviation)."""
    if len(ts) < 2:
        return 0.0
    mu = float(np.nanmean(ts.values))
    downside = ts.values[ts.values < 0]
    if len(downside) < 2:
        return 0.0
    dd = float(np.std(downside)) * math.sqrt(annualise)
    if dd < 1e-10:
        return 0.0
    return mu * annualise / dd


def max_drawdown(ts: TimeSeries) -> float:
    """Maximum drawdown as a fraction (0 to 1)."""
    if len(ts) == 0:
        return 0.0
    cumulative = np.cumsum(ts.values)
    peak = np.maximum.accumulate(cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - cumulative) / peak, 0.0)
    return float(np.nanmax(dd))


def drawdown_series(ts: TimeSeries) -> TimeSeries:
    """Running drawdown at each date (as fraction)."""
    if len(ts) == 0:
        return TimeSeries.empty(ts.name)
    cumulative = np.cumsum(ts.values)
    peak = np.maximum.accumulate(cumulative)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, (peak - cumulative) / peak, 0.0)
    return TimeSeries(ts.dates.copy(), dd, f"{ts.name}_drawdown")


def recovery_time(ts: TimeSeries) -> int:
    """Maximum drawdown duration in periods."""
    if len(ts) == 0:
        return 0
    cumulative = np.cumsum(ts.values)
    peak = np.maximum.accumulate(cumulative)
    in_dd = peak > cumulative
    dur = 0
    max_dur = 0
    for d in in_dd:
        if d:
            dur += 1
            max_dur = max(max_dur, dur)
        else:
            dur = 0
    return max_dur


def performance(ts: TimeSeries, initial_capital: float = 1_000_000.0):
    """Full performance metrics — delegates to backtest.compute_metrics."""
    from pricebook.backtest import compute_metrics
    return compute_metrics(ts.values, initial_capital)
