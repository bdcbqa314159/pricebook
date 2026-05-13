"""Pricebook time series abstraction and backtesting replay.

Clean API for time series operations — no raw pandas/numpy in notebooks.

    from pricebook.ts import TimeSeries

    ts = TimeSeries.from_lists(dates, values, name="desk_pnl")
    returns = ts.simple_returns()
    print(ts.sharpe())
    print(ts.max_drawdown())
"""

from pricebook.ts._core import TimeSeries

# Returns
from pricebook.ts._returns import simple_returns, log_returns, period_returns

# Statistics
from pricebook.ts._stats import (
    mean, vol, sharpe, sortino, max_drawdown,
    drawdown_series, recovery_time, performance,
)

# Rolling
from pricebook.ts._rolling import (
    rolling_mean, rolling_vol, rolling_sharpe,
    rolling_beta, rolling_skew, rolling_kurtosis,
)

# ── Bind methods on TimeSeries for fluent API ──
TimeSeries.simple_returns = lambda self: simple_returns(self)
TimeSeries.log_returns = lambda self: log_returns(self)
TimeSeries.period_returns = lambda self, period=1: period_returns(self, period)
TimeSeries.mean = lambda self: mean(self)
TimeSeries.vol = lambda self, ann=252: vol(self, ann)
TimeSeries.sharpe = lambda self, ann=252: sharpe(self, ann)
TimeSeries.sortino = lambda self, ann=252: sortino(self, ann)
TimeSeries.max_drawdown = lambda self: max_drawdown(self)
TimeSeries.drawdown_series = lambda self: drawdown_series(self)
TimeSeries.recovery_time = lambda self: recovery_time(self)
TimeSeries.performance = lambda self, cap=1_000_000: performance(self, cap)
TimeSeries.rolling_mean = lambda self, window=60: rolling_mean(self, window)
TimeSeries.rolling_vol = lambda self, window=60: rolling_vol(self, window)
TimeSeries.rolling_sharpe = lambda self, window=60: rolling_sharpe(self, window)
TimeSeries.rolling_beta = lambda self, bench, window=60: rolling_beta(self, bench, window)

__all__ = [
    "TimeSeries",
    "simple_returns", "log_returns", "period_returns",
    "mean", "vol", "sharpe", "sortino", "max_drawdown",
    "drawdown_series", "recovery_time", "performance",
    "rolling_mean", "rolling_vol", "rolling_sharpe",
    "rolling_beta", "rolling_skew", "rolling_kurtosis",
]
