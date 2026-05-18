"""Tests for ts._stats."""
import pytest, numpy as np
from datetime import date, timedelta
from pricebook.ts._core import TimeSeries
from pricebook.ts._stats import mean, vol, sharpe, max_drawdown

@pytest.fixture
def returns_ts():
    rng = np.random.default_rng(42)
    dates = [date(2024,1,1) + timedelta(days=i) for i in range(252)]
    return TimeSeries(dates, rng.normal(0.001, 0.01, 252).tolist())

class TestVol:
    def test_positive(self, returns_ts):
        assert vol(returns_ts) > 0

class TestSharpe:
    def test_float(self, returns_ts):
        assert isinstance(sharpe(returns_ts), float)

class TestMaxDrawdown:
    def test_non_negative(self, returns_ts):
        dd = max_drawdown(returns_ts)
        assert dd >= 0  # drawdown magnitude
