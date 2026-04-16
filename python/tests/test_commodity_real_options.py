"""Tests for commodity real options (mine, power plant, FTR)."""

import math

import numpy as np
import pytest

from pricebook.commodity_real_options import (
    FTRResult,
    MineValuation,
    MineValuationResult,
    PowerPlantResult,
    UnitCommitmentResult,
    pipeline_ftr_value,
    power_plant_dispatch_value,
    unit_commitment_value,
)


# ---- Helpers ----

def _gen_price_paths(spot=80.0, vol=0.30, drift=0.03, T=5.0,
                      n_paths=200, n_steps=5, seed=42):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    paths = np.full((n_paths, n_steps + 1), spot)
    for step in range(n_steps):
        z = rng.standard_normal(n_paths)
        paths[:, step + 1] = paths[:, step] * np.exp(
            (drift - 0.5 * vol**2) * dt + vol * math.sqrt(dt) * z
        )
    df = np.exp(-drift * np.linspace(0, T, n_steps + 1))
    return paths, df


# ---- Mine valuation ----

class TestMineValuation:
    def test_basic(self):
        mine = MineValuation(
            initial_reserves=1000,
            annual_production=100,
            variable_cost=50,
            fixed_cost=1000,
        )
        paths, df = _gen_price_paths(spot=80, n_paths=200, n_steps=10)
        result = mine.value(paths, df)
        assert isinstance(result, MineValuationResult)

    def test_profitable_mine_positive(self):
        """When price > cost, mine value should be positive."""
        mine = MineValuation(1000, 100, variable_cost=40)
        paths, df = _gen_price_paths(spot=80, vol=0.1, n_paths=200, n_steps=10)
        result = mine.value(paths, df)
        assert result.value > 0

    def test_unprofitable_mine_zero(self):
        """When price < cost, mine value with shutdown option should be ≥ 0."""
        mine = MineValuation(1000, 100, variable_cost=200)
        paths, df = _gen_price_paths(spot=50, vol=0.1, n_paths=500, n_steps=10)
        result = mine.value(paths, df)
        assert result.value >= 0

    def test_restart_option_nonnegative(self):
        """Flexibility value ≥ 0."""
        mine = MineValuation(1000, 100, variable_cost=80)
        paths, df = _gen_price_paths(spot=80, vol=0.3, n_paths=300, n_steps=10)
        result = mine.value(paths, df)
        assert result.restart_option >= -1e-3


# ---- Power plant dispatch ----

class TestPowerPlantDispatch:
    def _gen_power_gas(self, n_paths=200, n_hours=24, seed=42):
        rng = np.random.default_rng(seed)
        power = 50 + 30 * rng.standard_normal((n_paths, n_hours))
        gas = 3 + 0.5 * rng.standard_normal((n_paths, n_hours))
        return power, gas

    def test_basic(self):
        power, gas = self._gen_power_gas()
        result = power_plant_dispatch_value(power, gas, heat_rate=7.5)
        assert isinstance(result, PowerPlantResult)

    def test_dispatch_ratio_in_range(self):
        power, gas = self._gen_power_gas()
        result = power_plant_dispatch_value(power, gas)
        assert 0 <= result.dispatch_ratio <= 1

    def test_high_heat_rate_low_value(self):
        """Higher heat rate (less efficient) → lower value."""
        power, gas = self._gen_power_gas(n_paths=500, seed=42)
        efficient = power_plant_dispatch_value(power, gas, heat_rate=7.0)
        inefficient = power_plant_dispatch_value(power, gas, heat_rate=12.0)
        assert efficient.value > inefficient.value

    def test_dispatch_option_positive(self):
        """Optionality → value ≥ 0 (can always shut down)."""
        power, gas = self._gen_power_gas(n_paths=500)
        result = power_plant_dispatch_value(power, gas)
        assert result.value >= 0


# ---- Unit commitment ----

class TestUnitCommitment:
    def test_basic(self):
        rng = np.random.default_rng(42)
        power = 50 + 20 * rng.standard_normal((100, 48))
        gas = 3 + 0.5 * rng.standard_normal((100, 48))
        result = unit_commitment_value(power, gas, startup_cost=5000)
        assert isinstance(result, UnitCommitmentResult)

    def test_startup_cost_reduces_value(self):
        rng = np.random.default_rng(42)
        power = 50 + 20 * rng.standard_normal((100, 48))
        gas = 3 + 0.5 * rng.standard_normal((100, 48))
        free_startup = unit_commitment_value(power, gas, startup_cost=0, shutdown_cost=0)
        with_startup = unit_commitment_value(power, gas, startup_cost=100_000, shutdown_cost=20_000)
        # Expensive startup should give lower value
        assert with_startup.value <= free_startup.value + 1e-3

    def test_n_startups_reasonable(self):
        rng = np.random.default_rng(42)
        power = 50 + 20 * rng.standard_normal((50, 48))
        gas = 3 + 0.5 * rng.standard_normal((50, 48))
        result = unit_commitment_value(power, gas, startup_cost=5000)
        assert result.n_startups_mean >= 0


# ---- FTR ----

class TestPipelineFTR:
    def _gen_lmps(self, n_paths=200, n_hours=24, congestion=5.0, seed=42):
        rng = np.random.default_rng(seed)
        source = 50 + 20 * rng.standard_normal((n_paths, n_hours))
        sink = source + congestion + 10 * rng.standard_normal((n_paths, n_hours))
        return source, sink

    def test_option_ftr_basic(self):
        source, sink = self._gen_lmps()
        result = pipeline_ftr_value(source, sink, capacity_mw=100)
        assert isinstance(result, FTRResult)
        assert result.value >= 0

    def test_obligation_ftr(self):
        source, sink = self._gen_lmps()
        result = pipeline_ftr_value(source, sink, capacity_mw=100, is_obligation=True)
        assert isinstance(result, FTRResult)

    def test_option_ftr_ge_obligation(self):
        """Option FTR ≥ obligation FTR (option never has losses)."""
        source, sink = self._gen_lmps(seed=42)
        option = pipeline_ftr_value(source, sink, 100, is_obligation=False)
        obligation = pipeline_ftr_value(source, sink, 100, is_obligation=True)
        assert option.value >= obligation.value - 1e-6

    def test_higher_capacity_higher_value(self):
        source, sink = self._gen_lmps(seed=42)
        small = pipeline_ftr_value(source, sink, capacity_mw=50)
        big = pipeline_ftr_value(source, sink, capacity_mw=200)
        assert big.value == pytest.approx(4 * small.value, rel=1e-6)

    def test_congestion_hours(self):
        source, sink = self._gen_lmps(congestion=10.0)
        result = pipeline_ftr_value(source, sink, 100)
        # Mostly positive congestion
        assert result.positive_congestion_hours > 10
