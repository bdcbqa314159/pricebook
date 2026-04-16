"""Tests for weather derivatives."""

import math

import numpy as np
import pytest

from pricebook.weather_derivatives import (
    DegreeDayIndex,
    RainfallResult,
    SeasonalOUTemperature,
    TemperaturePaths,
    WeatherFutureResult,
    WeatherOptionResult,
    WindOptionResult,
    cdd_index,
    hdd_future_price,
    hdd_index,
    hdd_option_price,
    rainfall_derivative_price,
    wind_index_option,
)


# ---- Degree days ----

class TestHDD:
    def test_cold_days_high_hdd(self):
        temps = [40, 45, 50, 55, 60]
        result = hdd_index(temps)
        assert isinstance(result, DegreeDayIndex)
        # HDD = (65-40) + (65-45) + (65-50) + (65-55) + (65-60) = 25+20+15+10+5 = 75
        assert result.total == 75.0
        assert result.index_type == "HDD"

    def test_warm_days_zero_hdd(self):
        temps = [70, 75, 80, 85]
        result = hdd_index(temps)
        assert result.total == 0.0

    def test_reference_param(self):
        temps = [50, 60]
        result = hdd_index(temps, reference=60)
        # (60-50) + (60-60) = 10
        assert result.total == 10.0


class TestCDD:
    def test_warm_days_high_cdd(self):
        temps = [70, 75, 80, 85]
        result = cdd_index(temps)
        # CDD = 5+10+15+20 = 50
        assert result.total == 50.0
        assert result.index_type == "CDD"

    def test_cold_days_zero_cdd(self):
        temps = [40, 45, 50]
        assert cdd_index(temps).total == 0.0


# ---- HDD pricing ----

class TestHDDFuturePrice:
    def test_basic(self):
        result = hdd_future_price(expected_hdd=500, tick_value=20)
        assert isinstance(result, WeatherFutureResult)
        assert result.price == 500 * 20

    def test_discount_factor(self):
        result = hdd_future_price(500, tick_value=20, discount_factor=0.95)
        assert result.price == pytest.approx(500 * 20 * 0.95)


class TestHDDOption:
    def test_basic(self):
        rng = np.random.default_rng(42)
        sim = rng.normal(500, 50, 5000)
        result = hdd_option_price(sim, strike=500, tick_value=20)
        assert isinstance(result, WeatherOptionResult)
        assert result.price > 0

    def test_otm_call_smaller(self):
        rng = np.random.default_rng(42)
        sim = rng.normal(500, 50, 5000)
        atm = hdd_option_price(sim, 500)
        otm = hdd_option_price(sim, 700)
        assert otm.price < atm.price

    def test_cap_reduces_price(self):
        rng = np.random.default_rng(42)
        sim = rng.normal(500, 100, 5000)
        uncapped = hdd_option_price(sim, 500)
        capped = hdd_option_price(sim, 500, cap_level=50)
        assert capped.price <= uncapped.price

    def test_put_call_parity_approx(self):
        rng = np.random.default_rng(42)
        sim = rng.normal(500, 50, 10000)
        call = hdd_option_price(sim, 500, is_call=True)
        put = hdd_option_price(sim, 500, is_call=False)
        # call - put = (E[HDD] - K) × tick (discounted)
        diff = call.price - put.price
        expected = (sim.mean() - 500) * 20
        assert diff == pytest.approx(expected, abs=50)


# ---- OU temperature ----

class TestSeasonalOUTemperature:
    def test_seasonal_mean(self):
        model = SeasonalOUTemperature(a=50, b=0, c=20, t_shift=180)
        # At day 180 (summer), sin term max → T = 50 + 20 = 70
        assert model.seasonal_mean(180 + 91) == pytest.approx(70, abs=2)
        # At day 180 - 91 (winter ≈ day 89), sin near -1 → T ≈ 30
        val_winter = model.seasonal_mean(180 - 91)
        assert val_winter == pytest.approx(30, abs=2)

    def test_simulate_basic(self):
        model = SeasonalOUTemperature()
        result = model.simulate(n_days=30, n_paths=100, seed=42)
        assert isinstance(result, TemperaturePaths)
        assert result.temperatures.shape == (100, 30)

    def test_simulated_mean_near_seasonal(self):
        model = SeasonalOUTemperature(a=50, b=0, c=0, t_shift=0,
                                        kappa=0.5, sigma=0.1)
        result = model.simulate(365, n_paths=1000, seed=42)
        # c=0 → no seasonality → mean ≈ 50
        assert result.temperatures.mean() == pytest.approx(50, abs=3)

    def test_trend_drift(self):
        """Nonzero b → linear trend."""
        model = SeasonalOUTemperature(a=50, b=10, c=0, t_shift=0,
                                        kappa=0.5, sigma=1.0)
        result = model.simulate(365, n_paths=500, seed=42)
        # End of year: +10°F trend
        assert result.temperatures[:, -1].mean() > result.temperatures[:, 0].mean()


# ---- Rainfall ----

class TestRainfall:
    def test_drought_option(self):
        result = rainfall_derivative_price(
            n_days=30, mean_rainy_days=10, mean_rainfall_per_rainy_day=5.0,
            strike=20, is_drought_option=True, n_paths=2000, seed=42,
        )
        assert isinstance(result, RainfallResult)
        assert result.contract_type == "drought"

    def test_flood_option(self):
        result = rainfall_derivative_price(
            30, 10, 5.0, strike=200, is_drought_option=False,
            n_paths=2000, seed=42,
        )
        assert result.contract_type == "flood"

    def test_very_low_strike_drought(self):
        """Very low drought strike → very low probability."""
        result = rainfall_derivative_price(
            30, 20, 10.0, strike=5, is_drought_option=True,
            n_paths=2000, seed=42,
        )
        assert result.prob_below_threshold < 0.5

    def test_expected_total(self):
        result = rainfall_derivative_price(
            30, 15, 5.0, strike=100, n_paths=2000, seed=42,
        )
        # Expected ≈ 15 days × 5 = 75 (Gamma mean = shape × scale = 2 × 2.5 = 5)
        assert 50 < result.expected_total < 100


# ---- Wind ----

class TestWindIndex:
    def test_basic(self):
        rng = np.random.default_rng(42)
        wind_sim = rng.gamma(shape=5, scale=2, size=5000)    # mean=10
        result = wind_index_option(wind_sim, strike=10)
        assert isinstance(result, WindOptionResult)
        assert result.price > 0

    def test_high_strike_cheaper_call(self):
        rng = np.random.default_rng(42)
        wind_sim = rng.normal(10, 2, 5000)
        low = wind_index_option(wind_sim, 5)
        high = wind_index_option(wind_sim, 15)
        assert low.price > high.price

    def test_expected_wind_index(self):
        rng = np.random.default_rng(42)
        wind_sim = rng.normal(10, 2, 5000)
        result = wind_index_option(wind_sim, 10)
        assert result.expected_wind_index == pytest.approx(10, abs=0.2)
