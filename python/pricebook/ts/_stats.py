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
    from pricebook.risk.backtest import compute_metrics
    return compute_metrics(ts.values, initial_capital)


def information_ratio(ts: TimeSeries, benchmark: TimeSeries, annualise: int = 252) -> float:
    """Information ratio: annualised excess return / tracking error.

    IR = mean(r - b) / std(r - b) × sqrt(annualise)
    """
    from pricebook.ts._core import _align_intersect
    a, b = _align_intersect(ts, benchmark)
    if len(a) < 2:
        return 0.0
    excess = a.values - b.values
    te = float(np.nanstd(excess))
    if te < 1e-10:
        return 0.0
    return float(np.nanmean(excess)) / te * math.sqrt(annualise)


def tracking_error(ts: TimeSeries, benchmark: TimeSeries, annualise: int = 252) -> float:
    """Annualised tracking error: std(excess returns) × sqrt(annualise)."""
    from pricebook.ts._core import _align_intersect
    a, b = _align_intersect(ts, benchmark)
    if len(a) < 2:
        return 0.0
    return float(np.nanstd(a.values - b.values)) * math.sqrt(annualise)


def treynor_ratio(ts: TimeSeries, benchmark: TimeSeries,
                  risk_free: float = 0.0, annualise: int = 252) -> float:
    """Treynor ratio: (portfolio return - rf) / beta.

    Measures excess return per unit of systematic risk.
    """
    from pricebook.ts._core import _align_intersect
    a, b = _align_intersect(ts, benchmark)
    if len(a) < 2:
        return 0.0
    cov = np.cov(a.values, b.values)
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 1e-15 else 0.0
    if abs(beta) < 1e-10:
        return 0.0
    port_ret = float(np.nanmean(a.values)) * annualise
    return (port_ret - risk_free) / beta


def omega_ratio(ts: TimeSeries, threshold: float = 0.0) -> float:
    """Omega ratio: sum(gains above threshold) / sum(losses below threshold).

    Omega > 1 means the distribution is favourable.
    Uses the full return distribution, not just mean/variance.
    """
    if len(ts) == 0:
        return 0.0
    gains = np.sum(np.maximum(ts.values - threshold, 0.0))
    losses = np.sum(np.maximum(threshold - ts.values, 0.0))
    if losses < 1e-15:
        return float('inf') if gains > 0 else 0.0
    return float(gains / losses)


def gain_to_pain(ts: TimeSeries) -> float:
    """Gain-to-pain ratio: sum(returns) / sum(abs(negative returns))."""
    if len(ts) == 0:
        return 0.0
    total = float(np.sum(ts.values))
    pain = float(np.sum(np.abs(np.minimum(ts.values, 0.0))))
    if pain < 1e-15:
        return float('inf') if total > 0 else 0.0
    return total / pain


def kelly_fraction(win_prob: float, win_loss_ratio: float) -> float:
    """Kelly criterion (discrete): f* = p - (1-p)/b.

    Args:
        win_prob: probability of winning (0 to 1).
        win_loss_ratio: average win / average loss.
    """
    if win_loss_ratio < 1e-10:
        return 0.0
    return win_prob - (1.0 - win_prob) / win_loss_ratio


def kelly_continuous(mean_return: float, variance: float) -> float:
    """Kelly criterion (continuous): f* = mu / sigma^2.

    Optimal fraction of capital to invest assuming log-normal returns.
    """
    if variance < 1e-15:
        return 0.0
    return mean_return / variance
