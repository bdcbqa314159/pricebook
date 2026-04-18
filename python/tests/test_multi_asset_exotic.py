"""Tests for multi-asset exotic options."""
import math
import numpy as np
import pytest
from pricebook.multi_asset_exotic import (
    rainbow_option, knockout_basket, conditional_barrier, multi_asset_digital_range,
)

def _corr2(rho=0.5):
    return np.array([[1.0, rho], [rho, 1.0]])

def _corr3(rho=0.3):
    return np.array([[1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])

class TestRainbow:
    def test_best_of_call(self):
        r = rainbow_option([100, 100], 100, 0.03, [0.02, 0.02], [0.20, 0.25],
                            _corr2(), 1.0, "best_of_call", n_paths=5000, seed=42)
        assert r.price > 0
        assert r.rainbow_type == "best_of_call"

    def test_best_geq_single(self):
        """Best-of call ≥ single-asset call."""
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp(0.01)
        single = black76_price(F, 100, 0.20, 1.0, math.exp(-0.03), OptionType.CALL)
        best = rainbow_option([100, 100], 100, 0.03, [0.02, 0.02], [0.20, 0.20],
                               _corr2(0.3), 1.0, "best_of_call", n_paths=10_000, seed=42)
        assert best.price >= single * 0.9

    def test_worst_of_call(self):
        r = rainbow_option([100, 100], 100, 0.03, [0.02, 0.02], [0.20, 0.25],
                            _corr2(), 1.0, "worst_of_call", n_paths=5000, seed=42)
        assert r.price >= 0

    def test_best_geq_worst(self):
        best = rainbow_option([100, 100], 100, 0.03, [0.02, 0.02], [0.20, 0.20],
                               _corr2(), 1.0, "best_of_call", n_paths=5000, seed=42)
        worst = rainbow_option([100, 100], 100, 0.03, [0.02, 0.02], [0.20, 0.20],
                                _corr2(), 1.0, "worst_of_call", n_paths=5000, seed=42)
        assert best.price >= worst.price

    def test_atlas_3_assets(self):
        r = rainbow_option([100, 100, 100], 100, 0.03, [0.02]*3, [0.20]*3,
                            _corr3(), 1.0, "atlas", n_paths=5000, seed=42)
        assert r.n_assets == 3
        assert r.price >= 0

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            rainbow_option([100], 100, 0.03, [0.02], [0.20], np.eye(1), 1.0, "unknown")

class TestKnockoutBasket:
    def test_basic(self):
        r = knockout_basket([100, 100], 0.03, [0.02, 0.02], [0.20, 0.25],
                             _corr2(), 1.0, barrier_asset_idx=0, barrier_level=120,
                             is_up_barrier=True, payoff_asset_idx=1, strike=100,
                             n_paths=2000, seed=42)
        assert r.price >= 0
        assert 0 <= r.knockout_probability <= 1

    def test_ko_reduces_price(self):
        """Knockout basket ≤ vanilla basket."""
        ko = knockout_basket([100, 100], 0.03, [0.02, 0.02], [0.20, 0.25],
                              _corr2(), 1.0, 0, 110, True, 1, 100, n_paths=5000, seed=42)
        # Simple vanilla on asset 1
        from pricebook.black76 import black76_price, OptionType
        F = 100 * math.exp(0.01)
        vanilla = black76_price(F, 100, 0.25, 1.0, math.exp(-0.03), OptionType.CALL)
        assert ko.price <= vanilla + 1.0

class TestConditionalBarrier:
    def test_basic(self):
        r = conditional_barrier([100, 100], 0.03, [0.02, 0.02], [0.20, 0.25],
                                 _corr2(), 1.0, ki_asset_idx=0, ki_level=110,
                                 ki_is_up=True, ko_asset_idx=1, ko_level=80,
                                 ko_is_up=False, payoff_asset_idx=0, strike=100,
                                 n_paths=2000, seed=42)
        assert r.price >= 0
        assert 0 <= r.ki_probability <= 1

    def test_probabilities_bounded(self):
        r = conditional_barrier([100, 100], 0.03, [0.02, 0.02], [0.20, 0.25],
                                 _corr2(), 1.0, 0, 110, True, 1, 85, False,
                                 0, 100, n_paths=2000, seed=42)
        assert r.both_triggered <= min(r.ki_probability, r.ko_probability) + 1e-6

class TestMultiAssetDigitalRange:
    def test_basic(self):
        r = multi_asset_digital_range([100, 100], 0.03, [0.02, 0.02], [0.10, 0.10],
                                       _corr2(), 1.0, [90, 90], [110, 110],
                                       n_paths=2000, seed=42)
        assert 0 <= r.all_in_range_probability <= 1
        assert r.price >= 0

    def test_wider_range_higher_prob(self):
        narrow = multi_asset_digital_range([100, 100], 0.03, [0.02]*2, [0.10]*2,
                                            _corr2(), 1.0, [95, 95], [105, 105],
                                            n_paths=2000, seed=42)
        wide = multi_asset_digital_range([100, 100], 0.03, [0.02]*2, [0.10]*2,
                                          _corr2(), 1.0, [50, 50], [200, 200],
                                          n_paths=2000, seed=42)
        assert wide.all_in_range_probability >= narrow.all_in_range_probability

    def test_multi_leq_single(self):
        """Multi-asset DNT ≤ single-asset for each component."""
        multi = multi_asset_digital_range([100, 100], 0.03, [0.02]*2, [0.15]*2,
                                           _corr2(), 1.0, [90, 90], [110, 110],
                                           n_paths=2000, seed=42)
        # Single asset with same range should have higher prob
        single = multi_asset_digital_range([100], 0.03, [0.02], [0.15],
                                            np.eye(1), 1.0, [90], [110],
                                            n_paths=2000, seed=42)
        assert multi.all_in_range_probability <= single.all_in_range_probability + 0.05
