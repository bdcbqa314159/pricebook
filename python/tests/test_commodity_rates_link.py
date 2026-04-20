"""Tests for commodity-rates link."""
import numpy as np, pytest
from pricebook.commodity_rates_link import inflation_commodity_factor_model, commodity_inflation_swap

class TestCrossAssetPCA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        data = {"rates": list(rng.normal(0, 1, 100)),
                "oil": list(rng.normal(0, 2, 100)),
                "cpi": list(rng.normal(0, 0.5, 100))}
        r = inflation_commodity_factor_model(data, n_factors=2)
        assert r.n_factors == 2
        assert r.cumulative_explained[-1] <= 1.0
    def test_dominant_factor(self):
        rng = np.random.default_rng(42)
        big = rng.normal(0, 10, 100)
        data = {"big_vol": list(big), "small_vol": list(rng.normal(0, 0.1, 100))}
        r = inflation_commodity_factor_model(data, 1)
        assert r.dominant_factor == "big_vol"

class TestCommodityInflationSwap:
    def test_basic(self):
        rng = np.random.default_rng(42)
        c = np.full((200, 11), 80.0)
        i = np.full((200, 11), 260.0)
        for s in range(10):
            c[:, s+1] = c[:, s] * np.exp(0.005 + 0.05*rng.standard_normal(200))
            i[:, s+1] = i[:, s] * np.exp(0.002 + 0.01*rng.standard_normal(200))
        r = commodity_inflation_swap(c, i, 0.5, 0.5, 0.03, 1000, 0.97)
        assert isinstance(r.price, float)
    def test_zero_fixed_rate(self):
        rng = np.random.default_rng(42)
        c = np.full((200, 11), 80.0)
        i = np.full((200, 11), 260.0)
        for s in range(10):
            c[:, s+1] = c[:, s] * np.exp(0.01*rng.standard_normal(200))
            i[:, s+1] = i[:, s] * np.exp(0.005*rng.standard_normal(200))
        r = commodity_inflation_swap(c, i, 0.5, 0.5, 0.0, 1000, 0.97)
        # Should be near zero since returns are small and zero fixed
        assert abs(r.price) < 500
