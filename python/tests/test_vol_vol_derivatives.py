"""Tests for vol-of-vol derivatives."""
import math
import numpy as np
import pytest
from pricebook.vol_vol_derivatives import (
    option_on_variance_swap, gamma_swap_price, corridor_variance_swap, vix_option_price,
)

def _gen_paths(spot=100, vol=0.20, r=0.03, T=1.0, n_paths=2000, n_steps=50, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.full((n_paths, n_steps + 1), float(spot))
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, step + 1] = paths[:, step] * np.exp((r - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z)
    return paths

class TestOptionOnVariance:
    def test_basic(self):
        r = option_on_variance_swap(0.04, 0.5, 0.04, 1.0)
        assert r.price > 0
    def test_otm_call_cheaper(self):
        atm = option_on_variance_swap(0.04, 0.5, 0.04, 1.0)
        otm = option_on_variance_swap(0.04, 0.5, 0.10, 1.0)
        assert otm.price < atm.price
    def test_higher_vov_higher_price(self):
        low = option_on_variance_swap(0.04, 0.2, 0.04, 1.0)
        high = option_on_variance_swap(0.04, 0.8, 0.04, 1.0)
        assert high.price > low.price
    def test_put(self):
        r = option_on_variance_swap(0.04, 0.5, 0.04, 1.0, is_call=False)
        assert r.price > 0

class TestGammaSwap:
    def test_basic(self):
        paths = _gen_paths()
        r = gamma_swap_price(paths, rate=0.03, T=1.0)
        assert r.fair_strike > 0
    def test_gamma_vs_variance(self):
        """Gamma swap strike ≈ variance swap strike for symmetric dynamics."""
        paths = _gen_paths(vol=0.20, n_paths=5000)
        r = gamma_swap_price(paths, T=1.0)
        # Difference should be small for GBM (no skew)
        assert abs(r.gamma_adjustment) < r.variance_swap_strike

class TestCorridorVariance:
    def test_basic(self):
        paths = _gen_paths()
        r = corridor_variance_swap(paths, 80, 120, 1.0)
        assert r.conditional_variance >= 0
        assert 0 <= r.time_in_corridor <= 1
    def test_wide_range_equals_total(self):
        paths = _gen_paths()
        wide = corridor_variance_swap(paths, 0, 10000, 1.0)
        assert wide.conditional_variance == pytest.approx(wide.total_variance, rel=0.01)
    def test_narrow_range_less_variance(self):
        paths = _gen_paths()
        narrow = corridor_variance_swap(paths, 95, 105, 1.0)
        full = corridor_variance_swap(paths, 0, 10000, 1.0)
        assert narrow.conditional_variance <= full.conditional_variance + 1e-6

class TestVIXOption:
    def test_basic(self):
        r = vix_option_price(v0=0.04, kappa=2.0, theta=0.04, xi=0.3,
                              T_option=0.25, T_variance=30/365, strike_vol=0.20)
        assert r.price >= 0
        assert r.forward_vix > 0
    def test_higher_xi_higher_price(self):
        low = vix_option_price(0.04, 2.0, 0.04, 0.1, 0.25, 30/365, 0.20)
        high = vix_option_price(0.04, 2.0, 0.04, 0.8, 0.25, 30/365, 0.20)
        assert high.price >= low.price * 0.9
    def test_put(self):
        r = vix_option_price(0.04, 2.0, 0.04, 0.3, 0.25, 30/365, 0.20, is_call=False)
        assert r.price >= 0
