"""Tests for regulatory VaR/ES engine."""

import math
import pytest
import numpy as np

from pricebook.regulatory.var_es import (
    parametric_var, parametric_es, historical_var, historical_es,
    monte_carlo_var, portfolio_var, backtest_var, quick_var,
    compare_var_methods, scale_var, BacktestZone,
)


def _returns(n=504, seed=42):
    """2 years of synthetic daily returns (~20% annual vol)."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.20 / math.sqrt(252)
    daily_mean = 0.08 / 252
    return rng.normal(daily_mean, daily_vol, n)


# ---- Scaling ----

class TestScaling:
    def test_sqrt(self):
        assert scale_var(1.0, 10) == pytest.approx(math.sqrt(10))

    def test_linear(self):
        assert scale_var(1.0, 10, "linear") == 10.0


# ---- Parametric VaR ----

class TestParametricVaR:
    def test_positive(self):
        r = parametric_var(_returns(), confidence=0.99)
        assert r["var_pct"] > 0

    def test_es_greater_than_var(self):
        returns = _returns()
        var_r = parametric_var(returns, 0.99)
        es_r = parametric_es(returns, 0.99)
        assert es_r["es_pct"] > var_r["var_pct"]

    def test_higher_confidence_higher_var(self):
        returns = _returns()
        var_95 = parametric_var(returns, 0.95)
        var_99 = parametric_var(returns, 0.99)
        assert var_99["var_pct"] > var_95["var_pct"]

    def test_absolute_var(self):
        r = parametric_var(_returns(), position_value=10_000_000)
        assert r["var_abs"] > 0

    def test_t_distribution(self):
        r = parametric_var(_returns(), distribution="t", df=5)
        assert r["var_pct"] > 0
        assert r["distribution"] == "t"

    def test_horizon_scaling(self):
        r1 = parametric_var(_returns(), horizon_days=1)
        r10 = parametric_var(_returns(), horizon_days=10)
        assert r10["var_pct"] > r1["var_pct"]


# ---- Historical VaR ----

class TestHistoricalVaR:
    def test_positive(self):
        r = historical_var(_returns(), 0.99)
        assert r["var_pct"] > 0

    def test_es_greater(self):
        returns = _returns()
        var_r = historical_var(returns, 0.99)
        es_r = historical_es(returns, 0.99)
        assert es_r["es_pct"] >= var_r["var_pct"]

    def test_tail_observations(self):
        r = historical_es(_returns(), 0.99)
        assert r["tail_observations"] > 0

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            historical_var([0.01, 0.02], 0.99)


# ---- Monte Carlo VaR ----

class TestMCVaR:
    def test_positive(self):
        r = monte_carlo_var(0.0003, 0.013, 0.99)
        assert r["var_pct"] > 0

    def test_deterministic(self):
        r1 = monte_carlo_var(0.0003, 0.013, 0.99, seed=42)
        r2 = monte_carlo_var(0.0003, 0.013, 0.99, seed=42)
        assert r1["var_pct"] == r2["var_pct"]

    def test_es_greater(self):
        r = monte_carlo_var(0.0003, 0.013, 0.99)
        assert r["es_pct"] > r["var_pct"]


# ---- Portfolio VaR ----

class TestPortfolioVaR:
    def test_diversification(self):
        rng = np.random.default_rng(42)
        R = rng.normal(0, 0.01, (252, 3))
        r = portfolio_var([0.4, 0.3, 0.3], R, 0.99)
        assert r["diversification_benefit"] > 0

    def test_component_sum(self):
        rng = np.random.default_rng(42)
        R = rng.normal(0, 0.01, (252, 2))
        r = portfolio_var([0.6, 0.4], R, 0.99)
        total = sum(r["component_var"])
        assert total == pytest.approx(r["portfolio_var_pct"], rel=0.1)

    def test_single_asset(self):
        rng = np.random.default_rng(42)
        R = rng.normal(0, 0.01, (252, 1))
        r = portfolio_var([1.0], R, 0.99)
        assert r["portfolio_var_pct"] > 0


# ---- Backtesting ----

class TestBacktest:
    def test_green_zone(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 250)
        var_est = np.full(250, 0.05)  # very conservative
        r = backtest_var(returns, var_est, 0.99)
        assert r["zone"] == "green"
        assert r["n_exceptions"] <= 4

    def test_red_zone(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 250)
        var_est = np.full(250, 0.001)  # too tight
        r = backtest_var(returns, var_est, 0.99)
        assert r["zone"] == "red"
        assert r["n_exceptions"] > 9

    def test_kupiec(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 250)
        var_est = np.full(250, 0.03)
        r = backtest_var(returns, var_est, 0.99)
        if r["kupiec_p_value"] is not None:
            assert 0 <= r["kupiec_p_value"] <= 1

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            backtest_var([0.01, 0.02], [0.03], 0.99)


# ---- Quick / Compare ----

class TestQuickVar:
    def test_all_methods(self):
        returns = _returns()
        for m in ["parametric", "historical", "monte_carlo"]:
            r = quick_var(returns, method=m)
            assert r["var_pct"] > 0
            assert r["es_pct"] > 0

    def test_compare(self):
        r = compare_var_methods(_returns())
        assert len(r["methods"]) == 3
        assert r["var_range"][0] <= r["var_range"][1]
