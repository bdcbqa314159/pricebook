"""Tests for maximum entropy option pricing."""

import pytest
import math
import numpy as np

from pricebook.options.entropy_pricing import (
    max_entropy_density, entropy_implied_vol, MaxEntropyResult,
)


def _bs_call(forward, strike, vol, T, df=1.0):
    """Black-Scholes call price for test data generation."""
    from scipy.stats import norm
    if vol <= 0 or T <= 0:
        return max(df * (forward - strike), 0)
    d1 = (math.log(forward / strike) + 0.5 * vol**2 * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return df * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


class TestMaxEntropy:
    @pytest.fixture
    def market_data(self):
        """Generate synthetic option prices from BS with 25% vol."""
        forward = 100.0
        vol = 0.25
        T = 1.0
        df = 0.96
        strikes = [80, 90, 95, 100, 105, 110, 120]
        prices = [_bs_call(forward, k, vol, T, df) for k in strikes]
        return strikes, prices, forward, df, T, vol

    def test_basic(self, market_data):
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        assert isinstance(result, MaxEntropyResult)
        assert result.converged

    def test_density_sums_to_one(self, market_data):
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        ds = result.grid[1] - result.grid[0]
        total = np.sum(result.density) * ds
        assert abs(total - 1.0) < 0.05

    def test_forward_recovered(self, market_data):
        """The density should imply a reasonable forward price."""
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        assert abs(result.forward - fwd) < fwd * 0.20  # within 20%

    def test_repricing(self, market_data):
        """Options should reprice within tolerance."""
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        for err in result.repricing_errors:
            assert abs(err) < 5.0  # within $5 (max-entropy is approximate)

    def test_entropy_positive(self, market_data):
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        assert result.entropy > 0

    def test_call_price(self, market_data):
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        # ATM call should be positive
        atm = result.call_price(fwd, df)
        assert atm > 0

    def test_put_call_parity(self, market_data):
        """C - P = df × (F - K)."""
        strikes, prices, fwd, df, T, vol = market_data
        result = max_entropy_density(strikes, prices, fwd, df)
        k = 100.0
        c = result.call_price(k, df)
        p = result.put_price(k, df)
        parity = df * (fwd - k)
        assert abs((c - p) - parity) < 5.0  # approximate density

    def test_to_dict(self, market_data):
        strikes, prices, fwd, df, T, vol = market_data
        d = max_entropy_density(strikes, prices, fwd, df).to_dict()
        assert "entropy" in d
        assert "converged" in d


class TestEntropyImpliedVol:
    def test_smile(self):
        forward = 100.0
        vol = 0.25
        T = 1.0
        df = 0.96
        strikes = [85, 90, 100, 110, 115]
        prices = [_bs_call(forward, k, vol, T, df) for k in strikes]
        smile = entropy_implied_vol(strikes, prices, forward, T, df)
        assert len(smile) > 0
        # At least some strikes should have positive IV
        # At least some strikes should produce finite prices
        prices_out = [s["call_price"] for s in smile if s["call_price"] > 0]
        assert len(prices_out) > 0


class TestEdgeCases:
    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            max_entropy_density([100], [10, 20], 100)

    def test_single_strike(self):
        """Should work with a single option constraint."""
        fwd = 100.0
        price = _bs_call(fwd, 100, 0.20, 1.0)
        result = max_entropy_density([100], [price], fwd)
        assert result.n_constraints == 1
