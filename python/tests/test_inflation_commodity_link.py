"""Tests for inflation-commodity link."""
import math, numpy as np, pytest
from pricebook.inflation_commodity_link import oil_breakeven_regression, commodity_inflation_hybrid

class TestOilBreakevenRegression:
    def test_basic(self):
        oil = [2, -1, 3, -2, 1, 4, -3, 2, 1, -1, 3, 2]
        be = [5, -2, 7, -4, 3, 10, -6, 4, 2, -3, 8, 5]  # ~2.5 bps per $1
        r = oil_breakeven_regression(oil, be)
        assert r.beta > 0
        assert r.r_squared > 0.5
    def test_no_relationship(self):
        np.random.seed(42)
        oil = list(np.random.randn(50))
        be = list(np.random.randn(50))
        r = oil_breakeven_regression(oil, be)
        assert r.r_squared < 0.3
    def test_insufficient(self):
        r = oil_breakeven_regression([1, 2], [3, 4])
        assert r.n_observations == 2

class TestCommodityInflationHybrid:
    def test_basic(self):
        rng = np.random.default_rng(42)
        c = np.full((500, 51), 80.0)
        i = np.full((500, 51), 260.0)
        for s in range(50):
            c[:, s+1] = c[:, s] * np.exp(0.001 + 0.30*rng.standard_normal(500)*0.14)
            i[:, s+1] = i[:, s] * np.exp(0.0002 + 0.02*rng.standard_normal(500)*0.14)
        r = commodity_inflation_hybrid(c, i, 0.5, 0.5, 0.0, 1000, 0.97)
        assert r.price > 0
    def test_zero_weight(self):
        rng = np.random.default_rng(42)
        c = np.full((200, 11), 80.0)
        i = np.full((200, 11), 260.0)
        for s in range(10):
            c[:, s+1] = c[:, s] * np.exp(0.01*rng.standard_normal(200))
            i[:, s+1] = i[:, s] * np.exp(0.005*rng.standard_normal(200))
        # Zero commodity weight → only inflation matters
        r = commodity_inflation_hybrid(c, i, 0.0, 1.0, -0.05, 1000, 0.97)
        assert r.price >= 0
