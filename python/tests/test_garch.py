"""Tests for GARCH, EGARCH, EWMA, realized vol."""
import pytest
import numpy as np
from pricebook.statistics.garch import (
    garch_11_fit, egarch_fit, ewma_vol, realized_vol, garch_var,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.01, 500)


class TestGARCH:
    def test_garch_fit(self, returns):
        result = garch_11_fit(returns)
        assert hasattr(result, "omega")
        assert hasattr(result, "alpha")
        assert hasattr(result, "beta")
        assert result.persistence < 1.0  # stationarity


class TestEGARCH:
    def test_egarch_fit(self, returns):
        result = egarch_fit(returns)
        assert hasattr(result, "omega")
        assert hasattr(result, "alpha")
        assert hasattr(result, "gamma")
        assert hasattr(result, "beta")


class TestEWMA:
    def test_ewma_vol(self, returns):
        vol = ewma_vol(returns)
        assert len(vol) == len(returns)
        assert all(v > 0 for v in vol)

    def test_ewma_decay(self, returns):
        vol_94 = ewma_vol(returns, decay=0.94)
        vol_97 = ewma_vol(returns, decay=0.97)
        assert np.std(np.diff(vol_97)) < np.std(np.diff(vol_94))


class TestRealizedVol:
    def test_realized_vol(self, returns):
        # realized_vol takes PRICES, not returns — convert the fixture first.
        prices = 100.0 * np.cumprod(1.0 + returns)
        rv = realized_vol(prices, window=22)
        assert len(rv) > 0
        assert all(v >= 0 for v in rv if not np.isnan(v))

    def test_realized_vol_rejects_non_positive(self):
        # Passing returns (mix of signs) instead of prices used to silently
        # emit RuntimeWarning from log(≤0); now raises ValueError up front.
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 50)
        with pytest.raises(ValueError, match="strictly-positive prices"):
            realized_vol(returns, window=10)

    def test_realized_vol_rejects_zero_price(self):
        prices = np.array([100.0, 99.0, 0.0, 98.0])
        with pytest.raises(ValueError, match="strictly-positive prices"):
            realized_vol(prices, window=2)


class TestGARCHVaR:
    def test_garch_var(self, returns):
        result = garch_var(returns)
        assert isinstance(result, float)
        assert result > 0  # VaR loss magnitude
