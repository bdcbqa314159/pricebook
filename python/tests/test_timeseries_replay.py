"""Tests for pricebook.ts — I/O, replay, drawdown analysis."""

import tempfile
import os
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.db import PricebookDB
from pricebook.ts import (
    TimeSeries,
    from_db, from_db_book, from_db_desk, from_csv, greeks_from_db,
    replay, replay_book, replay_desk,
    drawdown_analysis, rolling_performance,
    ReplayResult, DrawdownPeriod,
)


def _populate_db():
    """Create an in-memory DB with sample trade and P&L."""
    db = PricebookDB(":memory:")
    db.save_trade("t1", {"type": "irs"}, book="book_a", desk="rates")
    db.save_trade("t2", {"type": "bond"}, book="book_a", desk="rates")

    rng = np.random.default_rng(42)
    base = date(2024, 1, 2)
    for i in range(100):
        d = str(base + timedelta(days=i))
        for tid in ("t1", "t2"):
            pnl = float(rng.normal(100, 1000))
            db.save_pnl(tid, d, 1_000_000 + pnl, pnl,
                        float(rng.normal(5000, 500)),
                        float(rng.normal(10, 5)),
                        float(rng.normal(1000, 100)),
                        float(rng.normal(50, 10)),
                        float(rng.normal(20, 5)))
    return db


# ── I/O ──

class TestIO:
    def test_from_db(self):
        db = _populate_db()
        ts = from_db(db, "t1")
        assert len(ts) == 100
        assert ts.name == "t1_pnl_1d"

    def test_from_db_field(self):
        db = _populate_db()
        ts = from_db(db, "t1", field="pv")
        assert len(ts) == 100
        assert ts.name == "t1_pv"

    def test_from_db_book(self):
        db = _populate_db()
        ts = from_db_book(db, "book_a")
        assert len(ts) == 100  # aggregated by date

    def test_from_db_desk(self):
        db = _populate_db()
        ts = from_db_desk(db, "rates")
        assert len(ts) == 100

    def test_from_db_empty(self):
        db = _populate_db()
        ts = from_db(db, "nonexistent")
        assert len(ts) == 0

    def test_greeks_from_db(self):
        db = _populate_db()
        greeks = greeks_from_db(db, "t1")
        assert set(greeks.keys()) == {"delta", "gamma", "vega", "dv01", "cs01"}
        assert len(greeks["delta"]) == 100

    def test_from_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("date,value\n")
            f.write("2024-01-01,1.0\n")
            f.write("2024-01-02,2.0\n")
            f.write("2024-01-03,3.0\n")
            path = f.name
        try:
            ts = from_csv(path, name="test")
            assert len(ts) == 3
            assert ts[0] == 1.0
        finally:
            os.unlink(path)


# ── Replay ──

class TestReplay:
    def test_replay(self):
        db = _populate_db()
        result = replay(db, "t1")
        assert isinstance(result, ReplayResult)
        assert len(result.pnl) == 100
        assert len(result.cumulative_pnl) == 100
        assert hasattr(result.metrics, "sharpe")
        assert result.greeks is not None

    def test_replay_book(self):
        db = _populate_db()
        result = replay_book(db, "book_a")
        assert len(result.pnl) == 100
        assert result.greeks is None  # book-level has no per-trade greeks

    def test_replay_desk(self):
        db = _populate_db()
        result = replay_desk(db, "rates")
        assert len(result.pnl) == 100

    def test_replay_empty(self):
        db = _populate_db()
        result = replay(db, "nonexistent")
        assert len(result.pnl) == 0
        assert len(result.drawdown_periods) == 0


# ── Drawdown Analysis ──

class TestDrawdownAnalysis:
    def test_basic(self):
        ts = TimeSeries.from_lists(
            [date(2024, 1, i) for i in range(1, 11)],
            [1, 1, -3, -2, 1, 1, 1, 1, 1, 1],
        )
        periods = drawdown_analysis(ts)
        assert len(periods) >= 1
        assert periods[0].depth > 0

    def test_no_drawdown(self):
        ts = TimeSeries.from_lists(
            [date(2024, 1, i) for i in range(1, 6)],
            [1, 1, 1, 1, 1],
        )
        periods = drawdown_analysis(ts)
        assert len(periods) == 0

    def test_sorted_by_depth(self):
        db = _populate_db()
        ts = from_db(db, "t1")
        periods = drawdown_analysis(ts)
        if len(periods) >= 2:
            assert periods[0].depth >= periods[1].depth

    def test_drawdown_period_fields(self):
        db = _populate_db()
        ts = from_db(db, "t1")
        periods = drawdown_analysis(ts)
        if periods:
            p = periods[0]
            assert p.start_date is not None
            assert p.trough_date is not None
            assert p.duration > 0
            assert p.depth > 0


# ── Rolling Performance ──

class TestRollingPerformance:
    def test_basic(self):
        db = _populate_db()
        ts = from_db(db, "t1")
        rp = rolling_performance(ts, window=20)
        assert "rolling_sharpe" in rp
        assert "rolling_vol" in rp
        assert len(rp["rolling_sharpe"]) == len(ts)
