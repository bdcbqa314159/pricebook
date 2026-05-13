"""Tests for pricebook.ts — TimeSeries core, returns, stats, rolling."""

from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.ts import (
    TimeSeries, simple_returns, log_returns, period_returns,
    mean, vol, sharpe, sortino, max_drawdown, drawdown_series,
    recovery_time, performance,
    rolling_mean, rolling_vol, rolling_sharpe, rolling_beta,
)


def _make_ts(n=100, seed=42, name="test"):
    rng = np.random.default_rng(seed)
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    values = list(rng.normal(0.001, 0.01, n))
    return TimeSeries.from_lists(dates, values, name)


# ── Construction ──

class TestConstruction:
    def test_from_lists(self):
        ts = _make_ts(10)
        assert len(ts) == 10
        assert ts.name == "test"

    def test_from_dict(self):
        ts = TimeSeries.from_dict({"2024-01-03": 3, "2024-01-01": 1, "2024-01-02": 2})
        assert list(ts.values) == [1.0, 2.0, 3.0]

    def test_empty(self):
        ts = TimeSeries.empty("e")
        assert len(ts) == 0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            TimeSeries.from_lists([date(2024, 1, 1)], [1, 2])

    def test_sorted(self):
        ts = TimeSeries.from_lists(
            [date(2024, 1, 3), date(2024, 1, 1), date(2024, 1, 2)],
            [3, 1, 2],
        )
        assert list(ts.values) == [1.0, 2.0, 3.0]


# ── Access ──

class TestAccess:
    def test_getitem_int(self):
        ts = _make_ts(10)
        assert isinstance(ts[0], float)

    def test_getitem_date(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 2)], [10, 20])
        assert ts["2024-01-02"] == 20.0

    def test_getitem_missing(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1)], [10])
        with pytest.raises(KeyError):
            ts["2024-01-05"]

    def test_slice(self):
        ts = _make_ts(10)
        sub = ts[2:5]
        assert len(sub) == 3

    def test_iter(self):
        ts = _make_ts(3)
        items = list(ts)
        assert len(items) == 3


# ── Arithmetic ──

class TestArithmetic:
    def test_scalar_mul(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1)], [5.0])
        r = ts * 2
        assert r[0] == 10.0

    def test_scalar_add(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1)], [5.0])
        r = ts + 3
        assert r[0] == 8.0

    def test_ts_add_aligned(self):
        a = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 2)], [1, 2])
        b = TimeSeries.from_lists([date(2024, 1, 2), date(2024, 1, 3)], [20, 30])
        c = a + b
        assert len(c) == 1  # only date 2024-01-02 in common
        assert c[0] == 22.0

    def test_neg(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1)], [5.0])
        assert (-ts)[0] == -5.0

    def test_cumsum(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 2)], [1, 2])
        eq = ts.cumsum()
        assert list(eq.values) == [1.0, 3.0]

    def test_shift(self):
        ts = TimeSeries.from_lists(
            [date(2024, 1, i) for i in range(1, 4)], [10, 20, 30])
        s = ts.shift(1)
        assert np.isnan(s[0])
        assert s[1] == 10.0


# ── Filtering ──

class TestFiltering:
    def test_between(self):
        ts = _make_ts(30)
        sub = ts.between("2024-01-05", "2024-01-15")
        assert len(sub) == 11

    def test_business_days(self):
        ts = _make_ts(14)  # 2 weeks
        bd = ts.business_days_only()
        assert len(bd) == 10  # 10 weekdays

    def test_dropna(self):
        ts = TimeSeries.from_lists(
            [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
            [1.0, float("nan"), 3.0])
        clean = ts.dropna()
        assert len(clean) == 2


# ── Alignment ──

class TestAlignment:
    def test_align_union(self):
        a = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 2)], [1, 2])
        b = TimeSeries.from_lists([date(2024, 1, 2), date(2024, 1, 3)], [20, 30])
        aa, bb = a.align(b, fill="nan")
        assert len(aa) == 3
        assert np.isnan(bb[0])

    def test_align_ffill(self):
        a = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 3)], [1, 3])
        b = TimeSeries.from_lists(
            [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)], [10, 20, 30])
        aa, bb = a.align(b, fill="ffill")
        assert aa[1] == 1.0  # forward filled


# ── Resample ──

class TestResample:
    def test_monthly(self):
        ts = _make_ts(90)
        m = ts.resample("M")
        assert len(m) >= 3

    def test_weekly(self):
        ts = _make_ts(30)
        w = ts.resample("W")
        assert len(w) >= 4


# ── Returns ──

class TestReturns:
    def test_simple(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 2)], [100, 110])
        r = ts.simple_returns()
        assert len(r) == 1
        assert abs(r[0] - 0.1) < 1e-10

    def test_log(self):
        ts = TimeSeries.from_lists([date(2024, 1, 1), date(2024, 1, 2)], [100, 110])
        r = ts.log_returns()
        assert len(r) == 1
        assert abs(r[0] - np.log(1.1)) < 1e-10

    def test_period(self):
        ts = TimeSeries.from_lists(
            [date(2024, 1, i) for i in range(1, 5)], [100, 110, 121, 133])
        r = ts.period_returns(period=2)
        assert len(r) == 2
        assert abs(r[0] - 0.21) < 1e-10


# ── Statistics ──

class TestStats:
    def test_sharpe(self):
        ts = _make_ts(250)
        s = ts.sharpe()
        assert isinstance(s, float)

    def test_sortino(self):
        ts = _make_ts(250)
        s = ts.sortino()
        assert isinstance(s, float)

    def test_max_drawdown(self):
        ts = _make_ts(100)
        dd = ts.max_drawdown()
        assert dd >= 0

    def test_drawdown_series(self):
        ts = _make_ts(50)
        dd = ts.drawdown_series()
        assert len(dd) == len(ts)

    def test_recovery_time(self):
        ts = _make_ts(100)
        r = ts.recovery_time()
        assert r >= 0

    def test_performance(self):
        ts = _make_ts(250)
        perf = ts.performance()
        assert hasattr(perf, "sharpe")
        assert hasattr(perf, "max_drawdown")

    def test_empty_series(self):
        ts = TimeSeries.empty()
        assert mean(ts) == 0.0
        assert vol(ts) == 0.0
        assert sharpe(ts) == 0.0
        assert max_drawdown(ts) == 0.0


# ── Rolling ──

class TestRolling:
    def test_rolling_sharpe(self):
        ts = _make_ts(100)
        rs = ts.rolling_sharpe(window=20)
        assert len(rs) == len(ts)
        valid = rs.dropna()
        assert len(valid) == 80  # 100 - 20

    def test_rolling_vol(self):
        ts = _make_ts(100)
        rv = rolling_vol(ts, window=20)
        assert len(rv) == 100

    def test_rolling_beta(self):
        ts = _make_ts(100, seed=42)
        bench = _make_ts(100, seed=99, name="bench")
        b = ts.rolling_beta(bench, window=20)
        assert len(b.dropna()) > 0
