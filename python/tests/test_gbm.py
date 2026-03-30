"""Tests for GBM path generation."""

import pytest
import numpy as np
import math

from pricebook.gbm import GBMGenerator
from pricebook.rng import PseudoRandom, QuasiRandom


@pytest.fixture
def gen():
    return GBMGenerator(spot=100.0, rate=0.05, vol=0.20)


class TestConstruction:
    def test_basic(self, gen):
        assert gen.spot == 100.0
        assert gen.rate == 0.05
        assert gen.vol == 0.20
        assert gen.div_yield == 0.0

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="spot"):
            GBMGenerator(spot=-1, rate=0.05, vol=0.20)

    def test_negative_vol_raises(self):
        with pytest.raises(ValueError, match="vol"):
            GBMGenerator(spot=100, rate=0.05, vol=-0.1)


class TestGenerate:
    def test_shape(self, gen):
        paths = gen.generate(T=1.0, n_steps=12, n_paths=1000, rng=PseudoRandom(42))
        assert paths.shape == (1000, 13)  # n_steps + 1

    def test_starts_at_spot(self, gen):
        paths = gen.generate(T=1.0, n_steps=10, n_paths=500, rng=PseudoRandom(42))
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_positive_paths(self, gen):
        """GBM paths are always positive."""
        paths = gen.generate(T=1.0, n_steps=50, n_paths=5000, rng=PseudoRandom(42))
        assert np.all(paths > 0)

    def test_mean_terminal_matches_forward(self, gen):
        """E[S(T)] = S(0) * exp((r-q)*T) under risk-neutral measure."""
        n_paths = 100_000
        paths = gen.generate(T=1.0, n_steps=1, n_paths=n_paths, rng=PseudoRandom(42))
        mean_terminal = paths[:, -1].mean()
        expected = gen.spot * math.exp((gen.rate - gen.div_yield) * 1.0)
        assert mean_terminal == pytest.approx(expected, rel=0.01)

    def test_antithetic_shape(self, gen):
        paths = gen.generate(T=1.0, n_steps=12, n_paths=1000,
                             rng=PseudoRandom(42), antithetic=True)
        assert paths.shape == (2000, 13)

    def test_antithetic_starts_at_spot(self, gen):
        paths = gen.generate(T=1.0, n_steps=10, n_paths=500,
                             rng=PseudoRandom(42), antithetic=True)
        np.testing.assert_array_equal(paths[:, 0], 100.0)

    def test_antithetic_reduces_variance(self, gen):
        """Antithetic variates should reduce the variance of the terminal mean."""
        rng1 = PseudoRandom(seed=42)
        paths_plain = gen.generate(T=1.0, n_steps=1, n_paths=5000, rng=rng1)
        mean_plain = paths_plain[:, -1].mean()

        rng2 = PseudoRandom(seed=42)
        paths_anti = gen.generate(T=1.0, n_steps=1, n_paths=5000,
                                  rng=rng2, antithetic=True)
        mean_anti = paths_anti[:, -1].mean()

        expected = gen.spot * math.exp(gen.rate * 1.0)
        # Antithetic mean should be closer to the expected value
        assert abs(mean_anti - expected) <= abs(mean_plain - expected) + 0.5

    def test_with_dividend_yield(self):
        gen = GBMGenerator(spot=100.0, rate=0.05, vol=0.20, div_yield=0.02)
        paths = gen.generate(T=1.0, n_steps=1, n_paths=100_000, rng=PseudoRandom(42))
        mean_terminal = paths[:, -1].mean()
        expected = 100 * math.exp((0.05 - 0.02) * 1.0)
        assert mean_terminal == pytest.approx(expected, rel=0.01)

    def test_multi_step_mean(self, gen):
        """Multi-step terminal should match single-step terminal in expectation."""
        n = 50_000
        single = gen.generate(T=1.0, n_steps=1, n_paths=n, rng=PseudoRandom(42))
        multi = gen.generate(T=1.0, n_steps=252, n_paths=n, rng=PseudoRandom(99))
        # Both should have mean ≈ S0 * exp(r*T)
        expected = gen.spot * math.exp(gen.rate * 1.0)
        assert single[:, -1].mean() == pytest.approx(expected, rel=0.02)
        assert multi[:, -1].mean() == pytest.approx(expected, rel=0.02)


class TestTerminal:
    def test_shape(self, gen):
        st = gen.terminal(T=1.0, n_paths=1000, rng=PseudoRandom(42))
        assert st.shape == (1000,)

    def test_antithetic_shape(self, gen):
        st = gen.terminal(T=1.0, n_paths=1000, rng=PseudoRandom(42), antithetic=True)
        assert st.shape == (2000,)

    def test_mean(self, gen):
        st = gen.terminal(T=1.0, n_paths=100_000, rng=PseudoRandom(42))
        expected = gen.spot * math.exp(gen.rate * 1.0)
        assert st.mean() == pytest.approx(expected, rel=0.01)

    def test_positive(self, gen):
        st = gen.terminal(T=1.0, n_paths=10_000, rng=PseudoRandom(42))
        assert np.all(st > 0)

    def test_matches_generate_terminal(self, gen):
        """terminal() should give same distribution as generate() single-step."""
        rng1 = PseudoRandom(seed=42)
        st = gen.terminal(T=1.0, n_paths=5000, rng=rng1)

        rng2 = PseudoRandom(seed=42)
        paths = gen.generate(T=1.0, n_steps=1, n_paths=5000, rng=rng2)

        np.testing.assert_allclose(st, paths[:, -1])


class TestQuasiRandomPaths:
    def test_quasi_terminal(self, gen):
        qrng = QuasiRandom(dimension=1, seed=42)
        st = gen.terminal(T=1.0, n_paths=4096, rng=qrng)
        expected = gen.spot * math.exp(gen.rate * 1.0)
        assert st.mean() == pytest.approx(expected, rel=0.01)

    def test_quasi_multi_step(self, gen):
        paths = gen.generate(T=1.0, n_steps=12, n_paths=4096,
                             rng=QuasiRandom(dimension=12, seed=42))
        assert paths.shape == (4096, 13)
        assert np.all(paths > 0)
