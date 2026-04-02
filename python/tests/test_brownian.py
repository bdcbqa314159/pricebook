"""Tests for Brownian motion framework."""

import pytest
import math
import numpy as np

from pricebook.brownian import WienerProcess, CorrelatedBM, BrownianBridge


class TestWienerProcess:
    def test_starts_at_zero(self):
        wp = WienerProcess(seed=42)
        paths = wp.sample(T=1.0, n_steps=100, n_paths=500)
        assert paths.shape == (500, 101)
        np.testing.assert_array_equal(paths[:, 0], 0.0)

    def test_mean_near_zero(self):
        wp = WienerProcess(seed=42)
        paths = wp.sample(T=1.0, n_steps=1, n_paths=100_000)
        assert paths[:, -1].mean() == pytest.approx(0.0, abs=0.02)

    def test_variance_equals_t(self):
        wp = WienerProcess(seed=42)
        T = 2.0
        paths = wp.sample(T=T, n_steps=1, n_paths=100_000)
        assert paths[:, -1].var() == pytest.approx(T, rel=0.02)

    def test_independent_increments(self):
        """Increments over non-overlapping intervals are uncorrelated."""
        wp = WienerProcess(seed=42)
        paths = wp.sample(T=2.0, n_steps=2, n_paths=50_000)
        inc1 = paths[:, 1] - paths[:, 0]  # [0, 1]
        inc2 = paths[:, 2] - paths[:, 1]  # [1, 2]
        corr = np.corrcoef(inc1, inc2)[0, 1]
        assert abs(corr) < 0.02

    def test_increments_shape(self):
        wp = WienerProcess(seed=42)
        dW = wp.increments(T=1.0, n_steps=50, n_paths=100)
        assert dW.shape == (100, 50)

    def test_reproducible(self):
        w1 = WienerProcess(seed=123).sample(1.0, 10, 5)
        w2 = WienerProcess(seed=123).sample(1.0, 10, 5)
        np.testing.assert_array_equal(w1, w2)


class TestCorrelatedBM:
    def test_shape(self):
        cbm = CorrelatedBM([[1, 0.5], [0.5, 1]], seed=42)
        paths = cbm.sample(T=1.0, n_steps=50, n_paths=100)
        assert paths.shape == (100, 51, 2)

    def test_starts_at_zero(self):
        cbm = CorrelatedBM([[1, 0.5], [0.5, 1]], seed=42)
        paths = cbm.sample(T=1.0, n_steps=10, n_paths=100)
        np.testing.assert_array_equal(paths[:, 0, :], 0.0)

    def test_correlation_matches(self):
        """Simulated terminal correlation ≈ input correlation."""
        rho = 0.7
        cbm = CorrelatedBM([[1, rho], [rho, 1]], seed=42)
        paths = cbm.sample(T=1.0, n_steps=1, n_paths=100_000)
        w1 = paths[:, -1, 0]
        w2 = paths[:, -1, 1]
        sim_corr = np.corrcoef(w1, w2)[0, 1]
        assert sim_corr == pytest.approx(rho, abs=0.02)

    def test_zero_correlation_independent(self):
        cbm = CorrelatedBM([[1, 0], [0, 1]], seed=42)
        paths = cbm.sample(T=1.0, n_steps=1, n_paths=50_000)
        corr = np.corrcoef(paths[:, -1, 0], paths[:, -1, 1])[0, 1]
        assert abs(corr) < 0.02

    def test_perfect_correlation(self):
        cbm = CorrelatedBM([[1, 0.999], [0.999, 1]], seed=42)
        paths = cbm.sample(T=1.0, n_steps=1, n_paths=10_000)
        corr = np.corrcoef(paths[:, -1, 0], paths[:, -1, 1])[0, 1]
        assert corr > 0.99

    def test_three_dimensional(self):
        corr = [[1, 0.3, 0.5], [0.3, 1, 0.2], [0.5, 0.2, 1]]
        cbm = CorrelatedBM(corr, seed=42)
        paths = cbm.sample(T=1.0, n_steps=1, n_paths=100_000)
        for i in range(3):
            for j in range(i + 1, 3):
                sim = np.corrcoef(paths[:, -1, i], paths[:, -1, j])[0, 1]
                assert sim == pytest.approx(corr[i][j], abs=0.03)

    def test_dimension(self):
        cbm = CorrelatedBM([[1, 0.5], [0.5, 1]])
        assert cbm.dimension == 2

    def test_increments_shape(self):
        cbm = CorrelatedBM([[1, 0.5], [0.5, 1]], seed=42)
        dW = cbm.increments(T=1.0, n_steps=50, n_paths=100)
        assert dW.shape == (100, 50, 2)


class TestBrownianBridge:
    def test_endpoints(self):
        bb = BrownianBridge(seed=42)
        paths = bb.sample(T=1.0, n_steps=100, n_paths=500, start=0.0, end=1.0)
        assert paths.shape == (500, 101)
        np.testing.assert_array_equal(paths[:, 0], 0.0)
        np.testing.assert_array_equal(paths[:, -1], 1.0)

    def test_nonzero_start_end(self):
        bb = BrownianBridge(seed=42)
        paths = bb.sample(T=2.0, n_steps=50, n_paths=100, start=3.0, end=5.0)
        np.testing.assert_array_equal(paths[:, 0], 3.0)
        np.testing.assert_array_equal(paths[:, -1], 5.0)

    def test_mean_is_linear(self):
        """E[bridge(t)] = linear interpolation between start and end."""
        bb = BrownianBridge(seed=42)
        T = 1.0
        paths = bb.sample(T=T, n_steps=10, n_paths=100_000, start=0.0, end=2.0)
        # At midpoint t=0.5: mean should be ≈ 1.0
        mid_idx = 5
        assert paths[:, mid_idx].mean() == pytest.approx(1.0, abs=0.02)

    def test_variance_formula(self):
        """Var[bridge(t)] = t*(T-t)/T."""
        bb = BrownianBridge(seed=42)
        T = 1.0
        paths = bb.sample(T=T, n_steps=10, n_paths=100_000, start=0.0, end=0.0)
        t = 0.5
        mid_idx = 5
        expected_var = t * (T - t) / T  # = 0.25
        assert paths[:, mid_idx].var() == pytest.approx(expected_var, rel=0.05)

    def test_conditional_mean(self):
        assert BrownianBridge.conditional_mean(0.5, 1.0, 0.0, 2.0) == pytest.approx(1.0)
        assert BrownianBridge.conditional_mean(0.0, 1.0, 3.0, 5.0) == pytest.approx(3.0)
        assert BrownianBridge.conditional_mean(1.0, 1.0, 3.0, 5.0) == pytest.approx(5.0)

    def test_conditional_variance(self):
        assert BrownianBridge.conditional_variance(0.5, 1.0) == pytest.approx(0.25)
        assert BrownianBridge.conditional_variance(0.0, 1.0) == pytest.approx(0.0)
        assert BrownianBridge.conditional_variance(1.0, 1.0) == pytest.approx(0.0)

    def test_zero_length_bridge(self):
        assert BrownianBridge.conditional_variance(0.5, 0.0) == 0.0
