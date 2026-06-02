"""Unified portfolio analytics: risk-adjusted returns, drawdowns.

* :func:`portfolio_metrics` — Sharpe, Sortino, Calmar, max drawdown, etc.
* :func:`rolling_metrics` — rolling window analytics.
* :func:`tracking_metrics` — tracking error, information ratio.

References:
    Sharpe, *The Sharpe Ratio*, JPM, 1994.
    Sortino & van der Meer, *Downside Risk*, JPM, 1991.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PortfolioMetrics:
    """Complete portfolio performance metrics."""
    total_return: float
    annualised_return: float
    annualised_vol: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # periods
    skewness: float
    kurtosis: float
    var_95: float
    cvar_95: float
    hit_ratio: float            # fraction of positive returns
    best_period: float
    worst_period: float

    def to_dict(self) -> dict:
        return vars(self)


def portfolio_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PortfolioMetrics:
    """Compute complete portfolio performance metrics.

    Args:
        returns: (T,) array of period returns.
        risk_free_rate: annualised risk-free rate.
        periods_per_year: 252 for daily, 12 for monthly, etc.
    """
    T = len(returns)
    rf_per_period = risk_free_rate / periods_per_year

    # Basic
    total = float(np.prod(1 + returns) - 1)
    ann_ret = float((1 + total) ** (periods_per_year / max(T, 1)) - 1)
    ann_vol = float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))

    # Sharpe
    excess = returns - rf_per_period
    sharpe = float(np.mean(excess)) / float(np.std(excess, ddof=1)) * math.sqrt(periods_per_year) if np.std(excess) > 0 else 0

    # Sortino (downside deviation)
    downside = returns[returns < rf_per_period] - rf_per_period
    downside_dev = float(np.std(downside, ddof=1) * math.sqrt(periods_per_year)) if len(downside) > 1 else ann_vol
    sortino = (ann_ret - risk_free_rate) / downside_dev if downside_dev > 0 else 0

    # Drawdown
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (running_max - cum) / running_max
    max_dd = float(np.max(drawdowns))

    # Max drawdown duration
    in_dd = drawdowns > 0
    dd_lengths = []
    current = 0
    for d in in_dd:
        if d:
            current += 1
        else:
            if current > 0:
                dd_lengths.append(current)
            current = 0
    if current > 0:
        dd_lengths.append(current)
    max_dd_duration = max(dd_lengths) if dd_lengths else 0

    # Calmar
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    # Higher moments
    skew = float(np.mean((returns - np.mean(returns))**3) / np.std(returns)**3) if np.std(returns) > 0 else 0
    kurt = float(np.mean((returns - np.mean(returns))**4) / np.std(returns)**4 - 3) if np.std(returns) > 0 else 0

    # VaR / CVaR
    var_95 = float(np.percentile(-returns, 95))
    tail = -returns[-returns >= -np.percentile(returns, 5)]
    cvar_95 = float(np.mean(tail)) if len(tail) > 0 else var_95

    return PortfolioMetrics(
        total_return=total,
        annualised_return=ann_ret,
        annualised_vol=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=max_dd_duration,
        skewness=skew,
        kurtosis=kurt,
        var_95=var_95,
        cvar_95=cvar_95,
        hit_ratio=float(np.mean(returns > 0)),
        best_period=float(np.max(returns)),
        worst_period=float(np.min(returns)),
    )


def tracking_metrics(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    periods_per_year: int = 252,
) -> dict:
    """Tracking error and information ratio vs benchmark.

    Args:
        portfolio_returns: portfolio period returns.
        benchmark_returns: benchmark period returns.
    """
    active = portfolio_returns - benchmark_returns
    te = float(np.std(active, ddof=1) * math.sqrt(periods_per_year))
    ir = float(np.mean(active)) / float(np.std(active, ddof=1)) * math.sqrt(periods_per_year) if np.std(active) > 0 else 0

    # Active share (weight-based, approximate from returns)
    beta = float(np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)) if np.var(benchmark_returns) > 0 else 1
    alpha = float(np.mean(portfolio_returns) - beta * np.mean(benchmark_returns)) * periods_per_year

    return {
        "tracking_error": te,
        "information_ratio": ir,
        "alpha": alpha,
        "beta": beta,
        "active_return": float(np.mean(active)) * periods_per_year,
        "active_risk": te,
    }
