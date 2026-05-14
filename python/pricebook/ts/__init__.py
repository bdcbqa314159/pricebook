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
    information_ratio, tracking_error, treynor_ratio,
    omega_ratio, gain_to_pain, kelly_fraction, kelly_continuous,
)

# Rolling
from pricebook.ts._rolling import (
    rolling_mean, rolling_vol, rolling_sharpe,
    rolling_beta, rolling_skew, rolling_kurtosis,
)

# I/O
from pricebook.ts._io import from_db, from_db_book, from_db_desk, from_csv, greeks_from_db

# Replay
from pricebook.ts._replay import (
    replay, replay_book, replay_desk,
    drawdown_analysis, rolling_performance,
    ReplayResult, DrawdownPeriod,
)
from pricebook.ts._replay_viz import (
    plot_equity_curve, plot_pnl_histogram, plot_rolling_sharpe,
    plot_drawdowns, plot_dashboard,
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
TimeSeries.information_ratio = lambda self, bench, ann=252: information_ratio(self, bench, ann)
TimeSeries.tracking_error = lambda self, bench, ann=252: tracking_error(self, bench, ann)
TimeSeries.treynor_ratio = lambda self, bench, rf=0.0: treynor_ratio(self, bench, rf)
TimeSeries.omega_ratio = lambda self, threshold=0.0: omega_ratio(self, threshold)
TimeSeries.gain_to_pain = lambda self: gain_to_pain(self)

__all__ = [
    "TimeSeries",
    "simple_returns", "log_returns", "period_returns",
    "mean", "vol", "sharpe", "sortino", "max_drawdown",
    "drawdown_series", "recovery_time", "performance",
    "information_ratio", "tracking_error", "treynor_ratio",
    "omega_ratio", "gain_to_pain", "kelly_fraction", "kelly_continuous",
    "rolling_mean", "rolling_vol", "rolling_sharpe",
    "rolling_beta", "rolling_skew", "rolling_kurtosis",
    "from_db", "from_db_book", "from_db_desk", "from_csv", "greeks_from_db",
    "replay", "replay_book", "replay_desk",
    "drawdown_analysis", "rolling_performance",
    "ReplayResult", "DrawdownPeriod",
    "plot_equity_curve", "plot_pnl_histogram", "plot_rolling_sharpe",
    "plot_drawdowns", "plot_dashboard",
]
