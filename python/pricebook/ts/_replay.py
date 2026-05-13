"""Backtesting replay: load historical P&L, compute metrics, analyse drawdowns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pricebook.ts._core import TimeSeries
from pricebook.ts._io import from_db, from_db_book, from_db_desk, greeks_from_db
from pricebook.ts._stats import drawdown_series, performance


@dataclass
class DrawdownPeriod:
    """A single peak-to-trough-to-recovery drawdown episode."""
    start_date: str
    trough_date: str
    end_date: str | None    # None if still in drawdown
    depth: float            # max drawdown fraction during this period
    duration: int           # business days from start to end (or current)
    recovery: int | None    # days from trough to recovery


@dataclass
class ReplayResult:
    """Complete replay analysis of historical P&L."""
    pnl: TimeSeries
    cumulative_pnl: TimeSeries
    metrics: object             # PerformanceMetrics from backtest.py
    drawdown: TimeSeries
    drawdown_periods: list[DrawdownPeriod]
    greeks: dict[str, TimeSeries] | None


def _build_replay(pnl: TimeSeries, greeks: dict[str, TimeSeries] | None = None) -> ReplayResult:
    """Build a ReplayResult from a P&L TimeSeries."""
    cum = pnl.cumsum()
    dd = drawdown_series(pnl)
    metrics = performance(pnl)
    periods = drawdown_analysis(pnl)
    return ReplayResult(
        pnl=pnl,
        cumulative_pnl=cum,
        metrics=metrics,
        drawdown=dd,
        drawdown_periods=periods,
        greeks=greeks,
    )


def replay(db, trade_id: str) -> ReplayResult:
    """Replay P&L history for a single trade."""
    pnl = from_db(db, trade_id, "pnl_1d")
    greeks = greeks_from_db(db, trade_id)
    return _build_replay(pnl, greeks)


def replay_book(db, book: str) -> ReplayResult:
    """Replay aggregated P&L for all trades in a book."""
    pnl = from_db_book(db, book, "pnl_1d")
    return _build_replay(pnl)


def replay_desk(db, desk: str) -> ReplayResult:
    """Replay aggregated P&L for all trades in a desk."""
    pnl = from_db_desk(db, desk, "pnl_1d")
    return _build_replay(pnl)


def drawdown_analysis(ts: TimeSeries) -> list[DrawdownPeriod]:
    """Identify all drawdown periods, sorted by depth (worst first).

    A drawdown period starts when cumulative P&L drops below a prior peak,
    and ends when it recovers above that peak.
    """
    if len(ts) == 0:
        return []

    cumulative = np.cumsum(ts.values)
    peak = np.maximum.accumulate(cumulative)
    in_dd = cumulative < peak

    periods: list[DrawdownPeriod] = []
    start_idx = None
    trough_idx = None
    trough_val = 0.0

    for i in range(len(in_dd)):
        if in_dd[i]:
            if start_idx is None:
                start_idx = i
                trough_idx = i
                trough_val = cumulative[i]
            if cumulative[i] < trough_val:
                trough_idx = i
                trough_val = cumulative[i]
        else:
            if start_idx is not None:
                depth = (peak[start_idx] - trough_val) / peak[start_idx] if peak[start_idx] > 0 else 0
                periods.append(DrawdownPeriod(
                    start_date=str(ts.dates[start_idx]),
                    trough_date=str(ts.dates[trough_idx]),
                    end_date=str(ts.dates[i]),
                    depth=float(depth),
                    duration=i - start_idx,
                    recovery=i - trough_idx,
                ))
                start_idx = None
                trough_idx = None

    # Still in drawdown at end
    if start_idx is not None:
        depth = (peak[start_idx] - trough_val) / peak[start_idx] if peak[start_idx] > 0 else 0
        periods.append(DrawdownPeriod(
            start_date=str(ts.dates[start_idx]),
            trough_date=str(ts.dates[trough_idx]),
            end_date=None,
            depth=float(depth),
            duration=len(ts) - start_idx,
            recovery=None,
        ))

    periods.sort(key=lambda p: p.depth, reverse=True)
    return periods


def rolling_performance(
    ts: TimeSeries,
    window: int = 60,
) -> dict[str, TimeSeries]:
    """Rolling performance metrics as TimeSeries."""
    from pricebook.ts._rolling import rolling_sharpe, rolling_vol
    return {
        "rolling_sharpe": rolling_sharpe(ts, window),
        "rolling_vol": rolling_vol(ts, window),
    }
