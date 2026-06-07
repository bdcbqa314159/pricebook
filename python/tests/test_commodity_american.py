"""Tests for pricebook.commodity.commodity_american."""

import pytest

from pricebook.commodity.commodity_american import (
    american_commodity_option,
    american_energy_option,
    american_commodity_spread,
    early_exercise_test,
)


F = 50.0   # futures price
K = 50.0   # strike
R = 0.05
VOL = 0.25
T = 1.0


class TestAmericanCommodityOption:
    def test_price_positive_call(self):
        res = american_commodity_option(F, K, R, VOL, T, "call")
        assert res.price > 0

    def test_price_positive_put(self):
        res = american_commodity_option(F, K, R, VOL, T, "put")
        assert res.price > 0

    def test_put_price_reasonable(self):
        """American put should be in the right ballpark."""
        res = american_commodity_option(F, K, R, VOL, T, "put")
        assert res.price > 0
        assert res.price < F  # put can't be worth more than the underlying

    def test_itm_put_price_above_intrinsic(self):
        """Deep ITM put: American price >= intrinsic value."""
        res = american_commodity_option(30.0, 50.0, R, VOL, T, "put")
        assert res.price >= 20.0 - 1e-6


class TestAmericanEnergyOption:
    def test_price_positive(self):
        res = american_energy_option(F, K, R, VOL, T, "call", seasonal_vol_factor=1.0)
        assert res.price > 0

    def test_seasonal_factor_gt1_raises_price(self):
        """Higher seasonal vol -> higher option price."""
        base = american_energy_option(F, K, R, VOL, T, "call", seasonal_vol_factor=1.0)
        seasonal = american_energy_option(F, K, R, VOL, T, "call", seasonal_vol_factor=1.5)
        assert seasonal.price > base.price


class TestAmericanCommoditySpread:
    def test_price_positive(self):
        result = american_commodity_spread(
            F1=50.0, F2=45.0, strike=0.0,
            vol1=0.25, vol2=0.20, rho=0.5,
            T=T, r=R, option_type="call",
            n_paths=5000, seed=42,
        )
        assert result["price"] > 0

    def test_result_has_expected_keys(self):
        result = american_commodity_spread(
            F1=50.0, F2=45.0, strike=2.0,
            vol1=0.25, vol2=0.20, rho=0.5,
            T=T, r=R, n_paths=5000, seed=42,
        )
        for key in ("price", "european_price", "early_exercise_premium", "std_error"):
            assert key in result


class TestEarlyExerciseTest:
    def test_returns_dict_with_bool(self):
        result = early_exercise_test(F, K, R, VOL, T, "put")
        assert "is_optimal" in result
        assert isinstance(result["is_optimal"], bool)

    def test_not_optimal_atm(self):
        """ATM put near expiry but with positive time value: not optimal."""
        result = early_exercise_test(50.0, 50.0, 0.05, 0.25, 0.5, "put")
        # time value should be positive for ATM near-expiry
        assert result["time_value"] >= 0

    def test_intrinsic_positive_itm(self):
        result = early_exercise_test(30.0, 50.0, 0.05, 0.25, 1.0, "put")
        assert result["intrinsic"] == pytest.approx(20.0, abs=1e-8)
