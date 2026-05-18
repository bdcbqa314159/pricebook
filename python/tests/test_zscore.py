"""Tests for zscore."""
import pytest
import numpy as np
from pricebook.statistics.zscore import zscore, ZScoreSignal


class TestZScore:
    def test_basic(self):
        result = zscore(5.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert isinstance(result, ZScoreSignal)
        assert result.z_score is not None

    def test_at_mean(self):
        result = zscore(3.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(result.z_score) < 0.1  # at the mean

    def test_extreme(self):
        result = zscore(100.0, [1.0, 2.0, 3.0, 4.0, 5.0])
        assert result.z_score > 2.0  # way above mean
