"""Tests for MC variance reduction, Greeks, and exposure engine."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from pricebook.mc_engine import MCEngine, TimeGrid
from pricebook.mc_processes import BlackScholesProcess, OUProcess
from pricebook.mc_payoffs import (
    european_call, european_put, asian_arithmetic, asian_geometric,
)
from pricebook.mc_variance_reduction import control_variate, moment_matching
from pricebook.mc_greeks_engine import mc_greeks
from pricebook.mc_exposure import ExposureEngine


def bs_call(s0, k, r, sigma, T):
    d1 = (math.log(s0 / k) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return s0 * norm.cdf(d1) - k * math.exp(-r * T) * norm.cdf(d2)


# ── Variance Reduction ──

class TestControlVariate:

    def test_cv_price_finite(self):
        engine = MCEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(1.0, 100),
            n_paths=20_000, seed=42,
        )
        result = control_variate(
            engine, european_call(100), european_put(100),
            control_exact_price=5.0, discount_factor=math.exp(-0.05),
        )
        assert math.isfinite(result.price)


class TestMomentMatching:

    def test_adjusts_mean(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adjusted = moment_matching(values, target_mean=5.0)
        assert np.mean(adjusted) == pytest.approx(5.0)

    def test_no_target_unchanged(self):
        values = np.array([1.0, 2.0, 3.0])
        adjusted = moment_matching(values)
        np.testing.assert_array_equal(values, adjusted)


# ── Greeks ──

class TestMCGreeks:

    def _factory(self, s0, r, sigma):
        return BlackScholesProcess(s0, r, sigma)

    def test_call_delta_positive(self):
        greeks = mc_greeks(
            self._factory, TimeGrid.uniform(1.0, 1),
            european_call(100),
            spot=100, rate=0.05, vol=0.20, T=1.0,
            n_paths=100_000, seed=42,
        )
        assert 0 < greeks.delta < 1

    def test_put_delta_negative(self):
        greeks = mc_greeks(
            self._factory, TimeGrid.uniform(1.0, 1),
            european_put(100),
            spot=100, rate=0.05, vol=0.20, T=1.0,
            n_paths=100_000, seed=42,
        )
        assert -1 < greeks.delta < 0

    def test_gamma_positive(self):
        greeks = mc_greeks(
            self._factory, TimeGrid.uniform(1.0, 1),
            european_call(100),
            spot=100, rate=0.05, vol=0.20, T=1.0,
            n_paths=100_000, seed=42,
        )
        assert greeks.gamma > 0

    def test_vega_positive(self):
        greeks = mc_greeks(
            self._factory, TimeGrid.uniform(1.0, 1),
            european_call(100),
            spot=100, rate=0.05, vol=0.20, T=1.0,
            n_paths=100_000, seed=42,
        )
        assert greeks.vega > 0

    def test_to_dict(self):
        greeks = mc_greeks(
            self._factory, TimeGrid.uniform(1.0, 1),
            european_call(100),
            spot=100, rate=0.05, vol=0.20, T=1.0,
            n_paths=50_000, seed=42,
        )
        d = greeks.to_dict()
        assert "delta" in d and "gamma" in d and "vega" in d


# ── Exposure Engine ──

class TestExposureEngine:

    def test_epe_non_negative(self):
        engine = ExposureEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(5.0, 20),
            n_paths=5_000, seed=42,
        )
        result = engine.compute()
        assert all(ep >= 0 for ep in result.epe)

    def test_cva_positive(self):
        engine = ExposureEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(5.0, 20),
            n_paths=5_000, seed=42,
            counterparty_spread=0.02,
        )
        result = engine.compute()
        assert result.cva > 0

    def test_effective_epe_positive(self):
        engine = ExposureEngine(
            process=BlackScholesProcess(100, 0.05, 0.20),
            time_grid=TimeGrid.uniform(5.0, 20),
            n_paths=5_000, seed=42,
        )
        result = engine.compute()
        assert result.effective_epe > 0

    def test_custom_revalue(self):
        def swap_mtm(paths, step):
            rate = paths[:, step] if paths.ndim == 2 else paths[:, step, 0]
            return (rate - 0.05) * 10e6

        engine = ExposureEngine(
            process=OUProcess(0.05, 2.0, 0.04, 0.01),
            time_grid=TimeGrid.uniform(5.0, 20),
            n_paths=5_000, seed=42,
            revalue=swap_mtm,
        )
        result = engine.compute()
        assert math.isfinite(result.cva)
        assert len(result.epe) == 21

    def test_to_dict(self):
        engine = ExposureEngine(
            process=OUProcess(0.05, 2.0, 0.04, 0.01),
            time_grid=TimeGrid.uniform(3.0, 12),
            n_paths=2_000, seed=42,
        )
        d = engine.compute().to_dict()
        assert "cva" in d and "epe" in d and "fva" in d
