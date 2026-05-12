"""Tests for passport option."""
from __future__ import annotations
import math
import pytest
from pricebook.passport_option import passport_option


class TestPassportOption:

    def test_price_positive(self):
        r = passport_option(100, 0.05, 0.20, 1.0, n_paths=10_000)
        assert r.price > 0
        assert r.price_analytical > 0

    def test_mc_below_analytical_bound(self):
        r = passport_option(100, 0.05, 0.20, 1.0, n_paths=50_000)
        # MC (discrete bang-bang) <= analytical (continuous straddle upper bound)
        assert r.price <= r.price_analytical * 1.05
        assert r.price > r.price_analytical * 0.3  # at least 30% of bound

    def test_higher_vol_higher_price(self):
        low = passport_option(100, 0.05, 0.10, 1.0, n_paths=10_000)
        high = passport_option(100, 0.05, 0.40, 1.0, n_paths=10_000)
        assert high.price_analytical > low.price_analytical

    def test_longer_maturity_higher_price(self):
        short = passport_option(100, 0.05, 0.20, 0.5, n_paths=10_000)
        long = passport_option(100, 0.05, 0.20, 2.0, n_paths=10_000)
        assert long.price_analytical > short.price_analytical

    def test_prob_positive_around_half(self):
        r = passport_option(100, 0.05, 0.20, 1.0, n_paths=20_000)
        assert 0.3 < r.prob_positive_account < 0.7

    def test_optimal_strategy_described(self):
        r = passport_option(100, 0.05, 0.20, 1.0)
        assert "bang-bang" in r.optimal_strategy
