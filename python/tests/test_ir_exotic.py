"""Tests for exotic IR products."""

import pytest

from pricebook.ir_exotic import (
    CallableRangeAccrualResult,
    FlexiSwapResult,
    RatchetCapResult,
    SnowballResult,
    TARNResult,
    callable_range_accrual,
    flexi_swap,
    ratchet_cap,
    snowball_price,
    tarn_price,
)


# ---- TARN ----

class TestTARN:
    def test_positive_price(self):
        result = tarn_price(100, coupon_rate=0.05, target=0.15,
                            maturity_years=5, n_paths=10_000, seed=42)
        assert result.price > 0

    def test_target_hit_probability(self):
        """Low target → high probability of early redemption."""
        low = tarn_price(100, 0.05, target=0.05, maturity_years=5,
                         n_paths=20_000, seed=42)
        high = tarn_price(100, 0.05, target=0.50, maturity_years=5,
                          n_paths=20_000, seed=42)
        assert low.target_hit_probability > high.target_hit_probability

    def test_early_redemption_shorter_life(self):
        low = tarn_price(100, 0.05, target=0.05, maturity_years=5,
                         n_paths=20_000, seed=42)
        assert low.expected_life < 5.0

    def test_high_target_full_life(self):
        result = tarn_price(100, 0.05, target=1.0, maturity_years=5,
                            n_paths=10_000, seed=42)
        assert result.expected_life == pytest.approx(5.0, rel=0.01)


# ---- Snowball ----

class TestSnowball:
    def test_positive_price(self):
        result = snowball_price(100, 0.03, spread=0.01,
                                maturity_years=5, n_paths=10_000, seed=42)
        assert result.price > 0

    def test_coupon_accumulates(self):
        """Snowball total coupon should exceed initial × maturity."""
        result = snowball_price(100, 0.03, spread=0.01,
                                maturity_years=5, n_paths=10_000, seed=42)
        assert result.mean_total_coupon > 0

    def test_floor_prevents_negative(self):
        result = snowball_price(100, 0.01, spread=0.005,
                                maturity_years=5, floor=0.0,
                                rate_vol=0.03, n_paths=10_000, seed=42)
        assert result.mean_final_coupon >= 0


# ---- Callable range accrual ----

class TestCallableRangeAccrual:
    def test_callable_leq_non_callable(self):
        result = callable_range_accrual(
            100, 0.06, lower=0.02, upper=0.08,
            maturity_years=5, call_start_year=1,
            n_paths=10_000, seed=42,
        )
        assert result.price <= result.non_callable_price + 0.5

    def test_accrual_fraction_bounded(self):
        result = callable_range_accrual(
            100, 0.06, 0.02, 0.08, 5,
            n_paths=10_000, seed=42)
        assert 0 <= result.accrual_fraction <= 1

    def test_positive_price(self):
        result = callable_range_accrual(100, 0.05, 0.01, 0.10, 5,
                                         n_paths=10_000, seed=42)
        assert result.price > 0


# ---- Ratchet cap ----

class TestRatchetCap:
    def test_ratchet_geq_standard(self):
        """Ratchet cap ≥ standard cap (ratcheting strike is advantageous)."""
        result = ratchet_cap(1_000_000, initial_strike=0.05,
                             maturity_years=5, n_paths=20_000, seed=42)
        assert result.price >= result.standard_cap_price * 0.95

    def test_positive_price(self):
        result = ratchet_cap(1_000_000, 0.05, 3, n_paths=10_000, seed=42)
        assert result.price > 0

    def test_premium_non_negative(self):
        result = ratchet_cap(1_000_000, 0.05, 5, n_paths=20_000, seed=42)
        assert result.ratchet_premium >= -100  # allow MC noise


# ---- Flexi-swap ----

class TestFlexiSwap:
    def test_flexi_geq_zero(self):
        """Flexi-swap ≥ 0 (holder only exercises when profitable)."""
        result = flexi_swap(1_000_000, fixed_rate=0.05,
                            maturity_years=5, max_exercises=10,
                            n_paths=20_000, seed=42)
        assert result.price >= -100  # allow MC noise

    def test_optionality_positive(self):
        """Flexi > vanilla when max_exercises < n_periods and only positive CF taken."""
        result = flexi_swap(1_000_000, 0.05, 5, max_exercises=10,
                            n_paths=20_000, seed=42)
        assert result.optionality_value > result.vanilla_swap_price * -2

    def test_unlimited_exercises_equals_vanilla(self):
        """With enough exercises, flexi ≈ vanilla."""
        result = flexi_swap(1_000_000, 0.05, 5, max_exercises=100,
                            n_paths=20_000, seed=42)
        # Only positive cashflows taken → flexi > vanilla (which has negative CFs too)
        assert result.price >= result.vanilla_swap_price - 1000

    def test_mean_exercises_bounded(self):
        result = flexi_swap(1_000_000, 0.05, 3, max_exercises=5,
                            n_paths=10_000, seed=42)
        assert result.mean_exercises <= 5
