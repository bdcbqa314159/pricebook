"""Rolling analytics: delegates to statistics.rolling_stats."""

from __future__ import annotations

import math

import numpy as np

from pricebook.ts._core import TimeSeries, _align_intersect


def rolling_mean(ts: TimeSeries, window: int = 60) -> TimeSeries:
    """Rolling annualised mean."""
    from pricebook.statistics import rolling_stats
    rs = rolling_stats(ts.values, window)
    return TimeSeries(ts.dates.copy(), rs.rolling_mean, f"{ts.name}_rmean")


def rolling_vol(ts: TimeSeries, window: int = 60) -> TimeSeries:
    """Rolling annualised volatility."""
    from pricebook.statistics import rolling_stats
    rs = rolling_stats(ts.values, window)
    return TimeSeries(ts.dates.copy(), rs.rolling_vol, f"{ts.name}_rvol")


def rolling_sharpe(ts: TimeSeries, window: int = 60) -> TimeSeries:
    """Rolling annualised Sharpe ratio."""
    from pricebook.statistics import rolling_stats
    rs = rolling_stats(ts.values, window)
    return TimeSeries(ts.dates.copy(), rs.rolling_sharpe, f"{ts.name}_rsharpe")


def rolling_skew(ts: TimeSeries, window: int = 60) -> TimeSeries:
    """Rolling skewness."""
    from pricebook.statistics import rolling_stats
    rs = rolling_stats(ts.values, window)
    return TimeSeries(ts.dates.copy(), rs.rolling_skew, f"{ts.name}_rskew")


def rolling_kurtosis(ts: TimeSeries, window: int = 60) -> TimeSeries:
    """Rolling kurtosis."""
    from pricebook.statistics import rolling_stats
    rs = rolling_stats(ts.values, window)
    return TimeSeries(ts.dates.copy(), rs.rolling_kurt, f"{ts.name}_rkurt")


def rolling_beta(
    ts: TimeSeries,
    benchmark: TimeSeries,
    window: int = 60,
) -> TimeSeries:
    """Rolling beta vs benchmark: cov(ts, bench) / var(bench)."""
    a, b = _align_intersect(ts, benchmark)
    n = len(a)
    beta = np.full(n, np.nan)

    for i in range(window, n):
        x = a.values[i - window:i]
        y = b.values[i - window:i]
        var_y = np.var(y)
        if var_y > 1e-15:
            beta[i] = np.cov(x, y)[0, 1] / var_y
        else:
            beta[i] = 0.0

    return TimeSeries(a.dates.copy(), beta, f"{ts.name}_beta")
