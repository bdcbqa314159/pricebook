"""Tests for ts._returns."""
import pytest, numpy as np
from datetime import date
from pricebook.ts._core import TimeSeries
from pricebook.ts._returns import simple_returns, log_returns

class TestSimpleReturns:
    def test_basic(self):
        ts = TimeSeries([date(2024,1,i) for i in range(1,5)], [100.0, 110.0, 105.0, 115.0])
        ret = simple_returns(ts)
        assert abs(float(ret.values[0]) - 0.10) < 1e-10

class TestLogReturns:
    def test_callable(self):
        assert callable(log_returns)
