"""Return computation: simple, log, period."""

from __future__ import annotations

import numpy as np

from pricebook.ts._core import TimeSeries


def simple_returns(ts: TimeSeries) -> TimeSeries:
    """Simple returns: (v[t] - v[t-1]) / v[t-1]."""
    if len(ts) < 2:
        return TimeSeries.empty(ts.name)
    r = np.diff(ts.values) / ts.values[:-1]
    return TimeSeries(ts.dates[1:].copy(), r, ts.name)


def log_returns(ts: TimeSeries) -> TimeSeries:
    """Log returns: ln(v[t] / v[t-1])."""
    if len(ts) < 2:
        return TimeSeries.empty(ts.name)
    r = np.log(ts.values[1:] / ts.values[:-1])
    return TimeSeries(ts.dates[1:].copy(), r, ts.name)


def period_returns(ts: TimeSeries, period: int = 1) -> TimeSeries:
    """Multi-period returns: (v[t] - v[t-period]) / v[t-period]."""
    if len(ts) <= period:
        return TimeSeries.empty(ts.name)
    r = (ts.values[period:] - ts.values[:-period]) / ts.values[:-period]
    return TimeSeries(ts.dates[period:].copy(), r, ts.name)
