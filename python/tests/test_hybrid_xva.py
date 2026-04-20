"""Tests for hybrid XVA."""
import numpy as np, pytest
from pricebook.hybrid_xva import hybrid_cva, wrong_way_risk_adjustment, hybrid_fva

class TestHybridCVA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        exposure = rng.normal(10, 5, (500, 51))
        pd = np.linspace(0, 0.05, 51)
        r = hybrid_cva(exposure, pd, recovery=0.4)
        assert r.cva > 0
        assert r.peak_exposure > 0
    def test_higher_pd_higher_cva(self):
        rng = np.random.default_rng(42)
        exp = rng.normal(10, 5, (500, 51))
        low = hybrid_cva(exp, np.linspace(0, 0.01, 51))
        high = hybrid_cva(exp, np.linspace(0, 0.10, 51))
        assert high.cva > low.cva
    def test_wrong_way_increases(self):
        rng = np.random.default_rng(42)
        exp = rng.normal(10, 5, (500, 51))
        pd = np.linspace(0, 0.05, 51)
        base = hybrid_cva(exp, pd, wrong_way_factor=0.0)
        wwr = hybrid_cva(exp, pd, wrong_way_factor=0.5)
        assert wwr.cva > base.cva

class TestWrongWayRisk:
    def test_basic(self):
        rng = np.random.default_rng(42)
        eq = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 51)), axis=1))
        exp = rng.normal(5, 3, (200, 51))
        pd = np.linspace(0, 0.05, 51)
        r = wrong_way_risk_adjustment(eq, exp, pd)
        assert r.adjusted_cva != r.base_cva

class TestHybridFVA:
    def test_basic(self):
        rng = np.random.default_rng(42)
        exp = rng.normal(10, 5, (500, 51))
        r = hybrid_fva(exp, 50, dt=0.02)
        assert r.fva != 0
    def test_higher_spread_higher_fva(self):
        rng = np.random.default_rng(42)
        exp = rng.normal(10, 5, (500, 51))
        low = hybrid_fva(exp, 10, 0.02)
        high = hybrid_fva(exp, 100, 0.02)
        assert abs(high.fva) > abs(low.fva)
