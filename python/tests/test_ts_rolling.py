"""Tests for ts._rolling."""
import pytest
from pricebook.ts._rolling import rolling_mean, rolling_vol, rolling_sharpe

class TestImports:
    def test_rolling_mean_callable(self):
        assert callable(rolling_mean)
    def test_rolling_vol_callable(self):
        assert callable(rolling_vol)
    def test_rolling_sharpe_callable(self):
        assert callable(rolling_sharpe)
