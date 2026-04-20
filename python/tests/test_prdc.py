"""Tests for PRDC."""
import math, pytest
from pricebook.prdc import prdc_price, callable_prdc

class TestPRDC:
    def test_basic(self):
        r = prdc_price(110, 0.03, 0.01, 0.10, 0.005, 0.005, -0.3, 0.2, 0.3,
                         1000, 0.03, 1.0, 110, 5.0, n_paths=1000, seed=42)
        assert r.price > 0
        assert r.n_coupons == 10
    def test_higher_fx_vol_different_price(self):
        low = prdc_price(110, 0.03, 0.01, 0.05, 0.005, 0.005, 0, 0, 0,
                           1000, 0.03, 1.0, 110, 5.0, n_paths=1000, seed=42)
        high = prdc_price(110, 0.03, 0.01, 0.20, 0.005, 0.005, 0, 0, 0,
                            1000, 0.03, 1.0, 110, 5.0, n_paths=1000, seed=42)
        assert low.price != high.price

class TestCallablePRDC:
    def test_callable_leq_noncallable(self):
        r = callable_prdc(110, 0.03, 0.01, 0.10, 0.005, 0.005, -0.3, 0.2, 0.3,
                            1000, 0.03, 1.0, 110, 5.0, n_paths=1000, seed=42)
        assert r.price <= r.price_no_call * 1.01
    def test_call_probability(self):
        r = callable_prdc(110, 0.03, 0.01, 0.10, 0.005, 0.005, 0, 0, 0,
                            1000, 0.03, 1.0, 110, 5.0, n_paths=1000, seed=42)
        assert 0 <= r.call_probability <= 1
