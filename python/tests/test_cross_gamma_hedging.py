"""Tests for cross-gamma hedging."""
import numpy as np, pytest
from pricebook.cross_gamma_hedging import (
    optimal_multi_asset_hedge, cross_asset_vega_netting,
    correlation_aware_sizing, minimum_variance_exotic_hedge,
)

class TestOptimalHedge:
    def test_exact_hedge(self):
        target = np.array([100, 50, 30])
        instruments = np.array([[-100, 0, 0], [0, -50, 0], [0, 0, -30]])
        r = optimal_multi_asset_hedge(target, instruments)
        np.testing.assert_allclose(r.hedge_weights, [1, 1, 1], atol=1e-6)
        assert r.variance_reduction_pct > 99

    def test_partial_hedge(self):
        target = np.array([100, 50])
        instruments = np.array([[-80, 0]])
        r = optimal_multi_asset_hedge(target, instruments)
        assert r.portfolio_variance < r.unhedged_variance

    def test_with_covariance(self):
        target = np.array([100, 50])
        instruments = np.array([[-100, 0], [0, -50]])
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        r = optimal_multi_asset_hedge(target, instruments, cov)
        assert r.variance_reduction_pct > 90

class TestVegaNetting:
    def test_basic(self):
        r = cross_asset_vega_netting({"equity": 100, "fx": -80, "rates": 30})
        assert r.gross_vega > 0
        assert r.net_vega >= 0

    def test_with_correlations(self):
        corrs = {("equity", "fx"): -0.3, ("equity", "rates"): 0.2, ("fx", "rates"): 0.1}
        r = cross_asset_vega_netting({"equity": 100, "fx": -80, "rates": 30}, corrs)
        assert r.netting_benefit_pct > 0

    def test_perfectly_offsetting(self):
        r = cross_asset_vega_netting({"a": 100, "b": -100})
        assert r.net_vega < r.gross_vega

class TestCorrelationAwareSizing:
    def test_basic(self):
        r = correlation_aware_sizing(10, 50, 0.3, 100)
        assert r.optimal_size > 0

    def test_high_corr_smaller_size(self):
        low = correlation_aware_sizing(10, 50, 0.1, 100)
        high = correlation_aware_sizing(10, 50, 0.9, 100)
        assert high.optimal_size <= low.optimal_size

    def test_zero_vol(self):
        r = correlation_aware_sizing(10, 0, 0.5, 100)
        assert r.optimal_size == 0

class TestMinVarianceExoticHedge:
    def test_basic(self):
        rng = np.random.default_rng(42)
        exotic = rng.standard_normal(1000)
        hedges = rng.standard_normal((1000, 3))
        r = minimum_variance_exotic_hedge(exotic, hedges)
        assert r.hedged_std <= r.unhedged_std * 1.01  # should reduce or be similar

    def test_perfect_hedge(self):
        """If exotic = -hedge, perfect hedge."""
        rng = np.random.default_rng(42)
        exotic = rng.standard_normal(500)
        hedges = np.column_stack([exotic * -1])
        r = minimum_variance_exotic_hedge(exotic, hedges)
        assert r.hedged_std < 0.01
        assert r.variance_reduction_pct > 99
