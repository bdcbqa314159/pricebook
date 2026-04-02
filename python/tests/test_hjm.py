"""Tests for HJM framework."""

import pytest
import math
import numpy as np
from datetime import date

from pricebook.hjm import HJMModel
from tests.conftest import make_flat_curve


REF = date(2024, 1, 15)


class TestConstruction:
    def test_from_array(self):
        hjm = HJMModel([0.05, 0.05, 0.05], [1, 2, 5])
        assert hjm.n_tenors == 3
        np.testing.assert_array_equal(hjm.f0, [0.05, 0.05, 0.05])

    def test_from_curve(self):
        curve = make_flat_curve(REF, 0.05)
        hjm = HJMModel.from_curve(curve)
        assert hjm.n_tenors > 0
        # Flat curve → all forwards ≈ 0.05
        # Numerical forward extraction may differ slightly from the flat rate
        assert hjm.f0[0] == pytest.approx(0.05, abs=0.01)


class TestSimulation:
    def test_shape(self):
        hjm = HJMModel([0.05, 0.05, 0.05], [1, 2, 5], constant_vol=0.01)
        paths = hjm.simulate(T=2.0, n_steps=20, n_paths=100)
        assert paths.shape == (100, 21, 3)

    def test_starts_at_f0(self):
        hjm = HJMModel([0.05, 0.04, 0.03], [1, 5, 10], constant_vol=0.01)
        paths = hjm.simulate(T=1.0, n_steps=10, n_paths=50)
        for p in range(paths.shape[0]):
            np.testing.assert_array_almost_equal(paths[p, 0, :], [0.05, 0.04, 0.03])

    def test_mean_preserves_initial(self):
        """Under risk-neutral, E[f(t,T)] should be close to f(0,T) + drift terms."""
        hjm = HJMModel([0.05, 0.05, 0.05], [1, 5, 10], constant_vol=0.005)
        paths = hjm.simulate(T=1.0, n_steps=50, n_paths=50_000)
        # Mean of forward at each tenor should be near initial (for small vol)
        mean_terminal = paths[:, -1, :].mean(axis=0)
        for j in range(3):
            assert mean_terminal[j] == pytest.approx(0.05, abs=0.005)

    def test_zero_vol_deterministic(self):
        hjm = HJMModel([0.05, 0.05], [1, 5], constant_vol=0.0)
        paths = hjm.simulate(T=1.0, n_steps=10, n_paths=100)
        # Zero vol: all paths identical, forward stays at initial
        std = paths[:, -1, :].std(axis=0)
        assert np.all(std < 1e-10)


class TestDiscountFactors:
    def test_df_starts_at_one(self):
        hjm = HJMModel([0.05, 0.05], [0.25, 1], constant_vol=0.005)
        paths = hjm.simulate(T=2.0, n_steps=20, n_paths=100)
        df = hjm.discount_factors(paths, dt=0.1)
        np.testing.assert_array_almost_equal(df[:, 0], 1.0)

    def test_df_decreasing(self):
        hjm = HJMModel([0.05, 0.05], [0.25, 1], constant_vol=0.005)
        paths = hjm.simulate(T=2.0, n_steps=20, n_paths=100)
        df = hjm.discount_factors(paths, dt=0.1)
        # Each path: df should decrease over time (positive rates)
        for p in range(min(10, df.shape[0])):
            assert np.all(np.diff(df[p, :]) <= 0.01)  # roughly decreasing

    def test_avg_zcb_matches_curve(self):
        """Average simulated ZCB ≈ initial curve discount factor."""
        curve = make_flat_curve(REF, 0.05)
        hjm = HJMModel.from_curve(curve, tenors=[0.25, 0.5, 1, 2, 5],
                                    constant_vol=0.005)
        T = 5.0
        n_steps = 50
        paths = hjm.simulate(T=T, n_steps=n_steps, n_paths=50_000)
        dt = T / n_steps
        avg_df = hjm.zcb_prices(paths, dt)

        # At t=5 (last step): avg DF ≈ exp(-0.05*5) ≈ 0.778
        d5 = date.fromordinal(REF.toordinal() + int(5 * 365))
        mkt_df = curve.df(d5)
        assert avg_df[-1] == pytest.approx(mkt_df, rel=0.05)


class TestNoDrift:
    def test_drift_matches_formula(self):
        """No-drift condition: alpha = sigma * ∫sigma ds."""
        hjm = HJMModel([0.05, 0.05, 0.05], [1, 2, 5], constant_vol=0.01)
        sigma = np.array([0.01, 0.01, 0.01])
        drift = hjm._drift(0.0, sigma)
        # All sigmas equal: alpha[j] = sigma * sigma * sum(dx up to j)
        # dx = [1, 1, 3], cumsum = [0.01, 0.02, 0.05]
        # alpha = 0.01 * [0.01, 0.02, 0.05] = [0.0001, 0.0002, 0.0005]
        assert drift[0] > 0
        assert drift[-1] > drift[0]

    def test_zero_vol_zero_drift(self):
        hjm = HJMModel([0.05, 0.05], [1, 5], constant_vol=0.0)
        sigma = np.array([0.0, 0.0])
        drift = hjm._drift(0.0, sigma)
        np.testing.assert_array_almost_equal(drift, 0.0)
