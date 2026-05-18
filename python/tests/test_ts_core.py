"""Tests for ts._core: TimeSeries."""
import pytest, numpy as np
from datetime import date
from pricebook.ts._core import TimeSeries

class TestConstruction:
    def test_from_arrays(self):
        dates = [date(2024,1,i) for i in range(1,6)]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        ts = TimeSeries(dates, values)
        assert len(ts) == 5

    def test_scalar_mul(self):
        dates = [date(2024,1,i) for i in range(1,4)]
        ts = TimeSeries(dates, [1.0, 2.0, 3.0])
        result = ts * 2
        assert np.allclose(result.values, [2.0, 4.0, 6.0])

    def test_values_array(self):
        dates = [date(2024,1,i) for i in range(1,4)]
        ts = TimeSeries(dates, [10.0, 20.0, 30.0])
        assert ts.values is not None
