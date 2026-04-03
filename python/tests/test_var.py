"""Tests for VaR and stress testing."""

import pytest
import math
import numpy as np
from datetime import date

from pricebook.var import (
    historical_var,
    historical_cvar,
    parametric_var,
    stress_test,
    STANDARD_STRESSES,
)
from pricebook.pricing_context import PricingContext
from pricebook.vol_surface import FlatVol
from pricebook.swaption import Swaption
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestHistoricalVaR:
    def test_basic(self):
        pnl = [-10, -5, -3, -1, 0, 1, 3, 5, 7, 10]
        var_90 = historical_var(pnl, confidence=0.90)
        # 10th percentile of pnl = -10 + 0.1*range ≈ -8
        assert var_90 > 0

    def test_normal_pnl(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 100, 10_000)
        var_95 = historical_var(pnl, 0.95)
        # 95% VaR ≈ 1.645 * 100 ≈ 164.5
        assert var_95 == pytest.approx(164.5, rel=0.10)

    def test_all_positive_pnl(self):
        pnl = [1, 2, 3, 4, 5]
        var = historical_var(pnl, 0.95)
        assert var < 0  # VaR is negative (no loss)

    def test_higher_confidence_higher_var(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 100, 10_000)
        var_95 = historical_var(pnl, 0.95)
        var_99 = historical_var(pnl, 0.99)
        assert var_99 > var_95


class TestCVaR:
    def test_cvar_geq_var(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 100, 10_000)
        var = historical_var(pnl, 0.95)
        cvar = historical_cvar(pnl, 0.95)
        assert cvar >= var

    def test_cvar_positive(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0, 100, 10_000)
        assert historical_cvar(pnl, 0.95) > 0


class TestParametricVaR:
    def test_single_factor(self):
        deltas = np.array([100.0])
        cov = np.array([[0.0001]])  # 1% daily vol
        var_95 = parametric_var(deltas, cov, 0.95)
        # VaR = 1.645 * 100 * 0.01 = 1.645
        assert var_95 == pytest.approx(1.645, rel=0.01)

    def test_two_factors_uncorrelated(self):
        deltas = np.array([100.0, 100.0])
        cov = np.array([[0.0001, 0], [0, 0.0001]])
        var = parametric_var(deltas, cov, 0.95)
        # Portfolio std = sqrt(100^2*0.0001 + 100^2*0.0001) = sqrt(2) * 1
        expected = 1.645 * math.sqrt(2)
        assert var == pytest.approx(expected, rel=0.01)

    def test_correlation_reduces_var(self):
        """Diversification: negative correlation reduces VaR."""
        deltas = np.array([100.0, 100.0])
        cov_pos = np.array([[0.0001, 0.00008], [0.00008, 0.0001]])
        cov_neg = np.array([[0.0001, -0.00008], [-0.00008, 0.0001]])
        var_pos = parametric_var(deltas, cov_pos, 0.95)
        var_neg = parametric_var(deltas, cov_neg, 0.95)
        assert var_neg < var_pos

    def test_zero_delta_zero_var(self):
        deltas = np.array([0.0, 0.0])
        cov = np.array([[0.0001, 0], [0, 0.0001]])
        assert parametric_var(deltas, cov, 0.95) == pytest.approx(0.0)


class TestStressTest:
    def test_basic(self):
        curve = make_flat_curve(REF, 0.05)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=curve,
            vol_surfaces={"ir": FlatVol(0.20)},
        )
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)

        def pricer(c):
            return swn.pv_ctx(c)

        scenarios = [{"rate_shift": 0.01}, {"rate_shift": -0.01}]
        results = stress_test(pricer, ctx, scenarios, ["up_100", "dn_100"])

        assert len(results) == 2
        assert results[0]["name"] == "up_100"
        assert results[1]["name"] == "dn_100"

        # Up and down should give opposite PnL signs
        assert results[0]["pnl"] * results[1]["pnl"] < 0

    def test_zero_shock_zero_pnl(self):
        curve = make_flat_curve(REF, 0.05)
        ctx = PricingContext(
            valuation_date=REF,
            discount_curve=curve,
            vol_surfaces={"ir": FlatVol(0.20)},
        )
        swn = Swaption(date(2025, 1, 15), date(2030, 1, 15), strike=0.05)

        results = stress_test(lambda c: swn.pv_ctx(c), ctx, [{"rate_shift": 0.0}])
        assert results[0]["pnl"] == pytest.approx(0.0, abs=0.01)

    def test_standard_stresses(self):
        assert len(STANDARD_STRESSES) == 4
        assert STANDARD_STRESSES[0][0] == "parallel_up_100bp"
