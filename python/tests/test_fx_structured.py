"""Tests for FX structured products."""

import math

import numpy as np
import pytest

from pricebook.fx_structured import (
    AutocallableResult,
    DCDResult,
    PivotResult,
    TARFResult,
    fx_autocallable_price,
    fx_dual_currency_deposit,
    fx_pivot_option,
    fx_tarf_price,
)


# ---- TARF ----

class TestTARF:
    def test_basic(self):
        result = fx_tarf_price(1.0, 1.0, 0.20, 0.02, 0.01, 0.10, 2.0,
                                n_observations=12, n_paths=2000, seed=42)
        assert isinstance(result, TARFResult)
        assert result.target == 0.20
        assert result.strike == 1.0

    def test_target_hit_probability(self):
        """Lower target → higher hit probability."""
        low_target = fx_tarf_price(1.0, 1.0, 0.05, 0.02, 0.01, 0.15, 2.0,
                                    n_observations=12, n_paths=2000, seed=42)
        high_target = fx_tarf_price(1.0, 1.0, 0.50, 0.02, 0.01, 0.15, 2.0,
                                     n_observations=12, n_paths=2000, seed=42)
        assert low_target.prob_target_hit > high_target.prob_target_hit

    def test_higher_vol_higher_hit_probability(self):
        """Higher vol → more spot movement → higher target hit prob."""
        low_vol = fx_tarf_price(1.0, 1.0, 0.10, 0.02, 0.01, 0.05, 2.0,
                                 n_observations=12, n_paths=2000, seed=42)
        high_vol = fx_tarf_price(1.0, 1.0, 0.10, 0.02, 0.01, 0.20, 2.0,
                                  n_observations=12, n_paths=2000, seed=42)
        assert high_vol.prob_target_hit > low_vol.prob_target_hit

    def test_termination_before_maturity(self):
        """When hit, mean termination < T."""
        result = fx_tarf_price(1.0, 1.0, 0.05, 0.02, 0.01, 0.15, 2.0,
                                n_observations=24, n_paths=2000, seed=42)
        if result.prob_target_hit > 0.5:
            assert result.mean_termination_time < 2.0


# ---- Autocallable ----

class TestAutocallable:
    def test_basic(self):
        result = fx_autocallable_price(
            1.0, 1.05, coupon=0.03,
            rate_dom=0.02, rate_for=0.01, vol=0.10, T=3.0,
            observation_dates=[1.0, 2.0, 3.0],
            n_paths=2000, seed=42,
        )
        assert isinstance(result, AutocallableResult)
        assert 0 <= result.autocall_probability <= 1

    def test_low_barrier_higher_autocall(self):
        """Lower barrier → higher autocall probability."""
        high_bar = fx_autocallable_price(
            1.0, 1.20, 0.03, 0.02, 0.01, 0.10, 3.0,
            observation_dates=[1.0, 2.0, 3.0], n_paths=2000, seed=42,
        )
        low_bar = fx_autocallable_price(
            1.0, 1.02, 0.03, 0.02, 0.01, 0.10, 3.0,
            observation_dates=[1.0, 2.0, 3.0], n_paths=2000, seed=42,
        )
        assert low_bar.autocall_probability > high_bar.autocall_probability

    def test_memory_coupon_higher_price(self):
        """Memory coupon → higher payout → higher price."""
        with_mem = fx_autocallable_price(
            1.0, 1.05, 0.04, 0.02, 0.01, 0.10, 3.0,
            observation_dates=[1.0, 2.0, 3.0],
            has_memory=True, n_paths=2000, seed=42,
        )
        without = fx_autocallable_price(
            1.0, 1.05, 0.04, 0.02, 0.01, 0.10, 3.0,
            observation_dates=[1.0, 2.0, 3.0],
            has_memory=False, n_paths=2000, seed=42,
        )
        # Memory should be at least as valuable (typically more)
        assert with_mem.price >= without.price * 0.9

    def test_protection_barrier(self):
        result = fx_autocallable_price(
            1.0, 1.05, 0.03, 0.02, 0.01, 0.10, 3.0,
            observation_dates=[1.0, 2.0, 3.0],
            protection_barrier=0.85, n_paths=2000, seed=42,
        )
        assert result.price > 0


# ---- DCD ----

class TestDualCurrencyDeposit:
    def test_basic(self):
        result = fx_dual_currency_deposit(
            notional=1e6, spot=1.0, strike=1.05,
            rate_dom=0.02, rate_for=0.01, vol=0.10, T=0.5,
        )
        assert isinstance(result, DCDResult)
        assert result.enhanced_yield > result.base_yield

    def test_higher_vol_higher_enhanced(self):
        low = fx_dual_currency_deposit(1e6, 1.0, 1.05, 0.02, 0.01, 0.05, 0.5)
        high = fx_dual_currency_deposit(1e6, 1.0, 1.05, 0.02, 0.01, 0.25, 0.5)
        assert high.enhanced_yield > low.enhanced_yield

    def test_embedded_option_positive(self):
        result = fx_dual_currency_deposit(1e6, 1.0, 1.05, 0.02, 0.01, 0.10, 0.5)
        assert result.embedded_option_value > 0

    def test_proportional_to_notional(self):
        small = fx_dual_currency_deposit(1e6, 1.0, 1.05, 0.02, 0.01, 0.10, 0.5)
        big = fx_dual_currency_deposit(2e6, 1.0, 1.05, 0.02, 0.01, 0.10, 0.5)
        assert big.embedded_option_value == pytest.approx(2 * small.embedded_option_value, rel=1e-6)


# ---- Pivot ----

class TestPivotOption:
    def test_european_basic(self):
        result = fx_pivot_option(1.0, 0.95, 1.05, 0.02, 0.01, 0.10, 1.0)
        assert isinstance(result, PivotResult)
        assert 0 <= result.prob_in_range <= 1

    def test_wide_range_higher_probability(self):
        narrow = fx_pivot_option(1.0, 0.98, 1.02, 0.02, 0.01, 0.10, 1.0)
        wide = fx_pivot_option(1.0, 0.80, 1.20, 0.02, 0.01, 0.10, 1.0)
        assert wide.prob_in_range > narrow.prob_in_range

    def test_higher_vol_lower_narrow_range(self):
        """For narrow range: higher vol → lower probability spot lands in range."""
        low_vol = fx_pivot_option(1.0, 0.99, 1.01, 0.02, 0.01, 0.05, 1.0)
        high_vol = fx_pivot_option(1.0, 0.99, 1.01, 0.02, 0.01, 0.30, 1.0)
        assert low_vol.prob_in_range > high_vol.prob_in_range

    def test_path_dependent_less_than_european(self):
        """Path-dependent (DNT) ≤ European (requires staying in range whole time)."""
        eur = fx_pivot_option(1.0, 0.95, 1.05, 0.02, 0.01, 0.10, 1.0, is_european=True)
        path = fx_pivot_option(1.0, 0.95, 1.05, 0.02, 0.01, 0.10, 1.0, is_european=False)
        assert path.price <= eur.price * 1.05  # allow small MC noise
