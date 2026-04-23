"""Tests for risk framework."""

import math

import numpy as np
import pytest

from pricebook.risk_framework import (
    historical_var, parametric_var, delta_gamma_var, component_var,
    stress_test, StressScenario, STANDARD_SCENARIOS,
    analyse_drawdown, concentration_check,
)


# ---- VaR ----

class TestHistoricalVaR:
    def test_positive_var(self):
        ret = np.random.default_rng(42).standard_normal(1000) * 0.01
        result = historical_var(ret, 0.95)
        assert result.var > 0

    def test_cvar_exceeds_var(self):
        ret = np.random.default_rng(42).standard_normal(1000) * 0.01
        result = historical_var(ret, 0.95)
        assert result.cvar >= result.var

    def test_higher_confidence_higher_var(self):
        ret = np.random.default_rng(42).standard_normal(1000) * 0.01
        var_95 = historical_var(ret, 0.95)
        var_99 = historical_var(ret, 0.99)
        assert var_99.var > var_95.var

    def test_holding_period_scaling(self):
        ret = np.random.default_rng(42).standard_normal(1000) * 0.01
        var_1d = historical_var(ret, 0.95, holding_period=1)
        var_10d = historical_var(ret, 0.95, holding_period=10)
        assert var_10d.var > var_1d.var


class TestParametricVaR:
    def test_basic(self):
        result = parametric_var(10_000_000, 0.01, 0.95)
        assert result.var > 0
        assert result.method == "parametric"

    def test_higher_vol_higher_var(self):
        low = parametric_var(10_000_000, 0.005, 0.95)
        high = parametric_var(10_000_000, 0.02, 0.95)
        assert high.var > low.var


class TestDeltaGammaVaR:
    def test_basic(self):
        result = delta_gamma_var(1.0, 0.01, 0.20, 100.0, 0.95)
        assert result.var > 0


class TestComponentVaR:
    def test_two_positions(self):
        cov = np.array([[0.01, 0.003], [0.003, 0.015]])
        result = component_var({"A": 1_000_000, "B": 500_000}, cov, ["A", "B"])
        assert result.total_var > 0
        assert abs(sum(result.component_vars.values()) - result.total_var) < 1.0


# ---- Stress testing ----

class TestStressTest:
    def test_standard_scenarios(self):
        results = stress_test(1_000_000, {"rates": -50_000, "equity": 200_000})
        assert len(results) == len(STANDARD_SCENARIOS)
        for r in results:
            assert isinstance(r.pnl, float)

    def test_custom_scenario(self):
        sc = StressScenario("Custom", {"rates": 0.01})
        results = stress_test(1_000_000, {"rates": -100_000}, [sc])
        assert len(results) == 1
        assert results[0].pnl == pytest.approx(-100_000 * 0.01)

    def test_gfc_negative_pnl_for_long_equity(self):
        results = stress_test(1_000_000, {"equity": 500_000})
        gfc = [r for r in results if "GFC" in r.scenario][0]
        assert gfc.pnl < 0  # equity crash → loss


# ---- Drawdown ----

class TestDrawdown:
    def test_no_drawdown(self):
        eq = [100, 101, 102, 103, 104]
        dd = analyse_drawdown(eq)
        assert dd.max_drawdown == 0.0
        assert not dd.in_drawdown

    def test_simple_drawdown(self):
        eq = [100, 105, 95, 98, 103]
        dd = analyse_drawdown(eq)
        assert dd.max_drawdown > 0

    def test_current_drawdown(self):
        eq = [100, 110, 100]
        dd = analyse_drawdown(eq)
        assert dd.current_drawdown > 0
        assert dd.in_drawdown


# ---- Concentration ----

class TestConcentration:
    def test_equal_weights(self):
        positions = {f"pos_{i}": 100 for i in range(10)}
        result = concentration_check(positions)
        assert result.herfindahl == pytest.approx(0.10, rel=0.01)
        assert result.effective_n == pytest.approx(10, rel=0.01)

    def test_concentrated(self):
        positions = {"A": 900, "B": 50, "C": 50}
        result = concentration_check(positions)
        assert result.is_concentrated
        assert result.top_1_pct > 0.80

    def test_empty(self):
        result = concentration_check({})
        assert not result.is_concentrated
