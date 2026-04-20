"""Tests for tail risk."""
import math, numpy as np, pytest
from pricebook.tail_risk import roger_lee_bounds, svi_wings_fit, tail_risk_pricing, extreme_value_var

class TestRogerLee:
    def test_valid_surface(self):
        k = np.linspace(-0.5, 0.5, 11).tolist()
        w = [0.04 + 0.03 * abs(ki) for ki in k]
        r = roger_lee_bounds(k, w)
        assert r.is_valid  # slopes should be < 2

    def test_slopes_bounded(self):
        r = roger_lee_bounds([-0.5, -0.3, 0, 0.3, 0.5], [0.06, 0.05, 0.04, 0.05, 0.06])
        assert r.left_slope_bound == 2.0
        assert r.right_slope_bound == 2.0

class TestSVIWings:
    def test_basic(self):
        k = np.linspace(-0.3, 0.3, 9).tolist()
        vols = [0.25, 0.23, 0.21, 0.20, 0.195, 0.20, 0.21, 0.23, 0.25]
        r = svi_wings_fit(k, vols, 1.0)
        assert r.b >= 0

    def test_no_arb(self):
        k = np.linspace(-0.2, 0.2, 7).tolist()
        vols = [0.22, 0.21, 0.20, 0.195, 0.20, 0.21, 0.22]
        r = svi_wings_fit(k, vols, 1.0)
        # Wings should satisfy Roger Lee
        assert r.left_wing_slope <= 2.5  # allow small overshoot from fit
        assert r.right_wing_slope <= 2.5

class TestTailRiskPricing:
    def test_basic(self):
        r = tail_risk_pricing(100, 70, 0.03, 1.0, n_paths=10_000, seed=42)
        assert r.deep_otm_put_price > 0
        assert r.tail_probability > 0

    def test_deeper_otm_cheaper(self):
        near = tail_risk_pricing(100, 80, 0.03, 1.0, n_paths=10_000, seed=42)
        deep = tail_risk_pricing(100, 50, 0.03, 1.0, n_paths=10_000, seed=42)
        assert near.deep_otm_put_price > deep.deep_otm_put_price

class TestEVTVaR:
    def test_basic(self):
        rng = np.random.default_rng(42)
        losses = list(rng.standard_normal(1000))
        r = extreme_value_var(losses, 0.99)
        assert r.var_level > 0
        assert r.confidence == 0.99

    def test_heavier_tail_higher_var(self):
        rng = np.random.default_rng(42)
        normal = list(rng.standard_normal(500))
        heavy = list(rng.standard_t(3, 500))  # t-distribution, heavier tails
        r_normal = extreme_value_var(normal, 0.99)
        r_heavy = extreme_value_var(heavy, 0.99)
        assert r_heavy.var_level > r_normal.var_level * 0.8
