"""Tests for special stochastic processes."""

import pytest
import math
import numpy as np

from pricebook.special_process import (
    CIRProcess,
    OUProcess,
    BesselProcess,
    GammaProcess,
    InverseGaussianProcess,
)


class TestCIR:
    def test_mean_reversion(self):
        cir = CIRProcess(kappa=2.0, theta=0.04, xi=0.1, seed=42)
        paths = cir.sample(x0=0.10, T=10.0, n_steps=1000, n_paths=50_000)
        # After long time, should revert to theta
        terminal_mean = paths[:, -1].mean()
        assert terminal_mean == pytest.approx(0.04, rel=0.05)

    def test_analytical_mean(self):
        cir = CIRProcess(kappa=2.0, theta=0.04, xi=0.1, seed=42)
        paths = cir.sample(x0=0.10, T=2.0, n_steps=200, n_paths=100_000)
        sim_mean = paths[:, -1].mean()
        ana_mean = cir.mean(0.10, 2.0)
        assert sim_mean == pytest.approx(ana_mean, rel=0.02)

    def test_stays_positive_feller(self):
        """Feller condition satisfied → stays positive."""
        cir = CIRProcess(kappa=2.0, theta=0.04, xi=0.1)  # 2*2*0.04=0.16 > 0.01
        assert cir.feller
        paths = cir.sample(x0=0.04, T=5.0, n_steps=500, n_paths=1000)
        assert np.all(paths >= 0)

    def test_feller_flag(self):
        cir_ok = CIRProcess(kappa=2.0, theta=0.04, xi=0.1)
        assert cir_ok.feller
        cir_bad = CIRProcess(kappa=0.1, theta=0.01, xi=1.0)
        assert not cir_bad.feller

    def test_shape(self):
        cir = CIRProcess(kappa=2.0, theta=0.04, xi=0.1, seed=42)
        paths = cir.sample(x0=0.04, T=1.0, n_steps=10, n_paths=100)
        assert paths.shape == (100, 11)


class TestOU:
    def test_mean_reversion(self):
        ou = OUProcess(a=1.0, mu=0.5, sigma=0.2, seed=42)
        paths = ou.sample(x0=2.0, T=10.0, n_steps=100, n_paths=50_000)
        terminal_mean = paths[:, -1].mean()
        assert terminal_mean == pytest.approx(0.5, rel=0.05)

    def test_stationary_variance(self):
        ou = OUProcess(a=2.0, mu=0.0, sigma=1.0, seed=42)
        paths = ou.sample(x0=0.0, T=20.0, n_steps=200, n_paths=50_000)
        sim_var = paths[:, -1].var()
        assert sim_var == pytest.approx(ou.stationary_variance(), rel=0.05)

    def test_exact_simulation(self):
        """Exact simulation: moments should match analytical formulas."""
        ou = OUProcess(a=1.0, mu=0.0, sigma=1.0, seed=42)
        paths = ou.sample(x0=1.0, T=2.0, n_steps=1, n_paths=100_000)
        # E[X(2)] = 0 + (1-0)*exp(-2) = exp(-2) ≈ 0.135
        assert paths[:, -1].mean() == pytest.approx(math.exp(-2), rel=0.03)

    def test_shape(self):
        ou = OUProcess(a=1.0, seed=42)
        paths = ou.sample(x0=0.0, T=1.0, n_steps=10, n_paths=100)
        assert paths.shape == (100, 11)


class TestBessel:
    def test_mean_squared(self):
        """E[R(t)^2] = r0^2 + d*t."""
        bp = BesselProcess(dimension=3, seed=42)
        paths = bp.sample(r0=1.0, T=2.0, n_steps=200, n_paths=50_000)
        sim = (paths[:, -1]**2).mean()
        expected = bp.mean_squared(1.0, 2.0)  # 1 + 6 = 7
        assert sim == pytest.approx(expected, rel=0.05)

    def test_stays_positive(self):
        """d >= 2: Bessel stays positive."""
        bp = BesselProcess(dimension=3, seed=42)
        paths = bp.sample(r0=0.5, T=5.0, n_steps=500, n_paths=1000)
        assert np.all(paths >= 0)

    def test_invalid_dimension(self):
        with pytest.raises(ValueError):
            BesselProcess(dimension=0)


class TestGamma:
    def test_mean(self):
        gp = GammaProcess(variance_rate=0.5, seed=42)
        G = gp.sample(T=2.0, n_paths=100_000)
        assert G.mean() == pytest.approx(2.0, rel=0.02)

    def test_variance(self):
        gp = GammaProcess(variance_rate=0.5, seed=42)
        G = gp.sample(T=2.0, n_paths=100_000)
        assert G.var() == pytest.approx(0.5 * 2.0, rel=0.05)

    def test_non_negative(self):
        gp = GammaProcess(variance_rate=1.0, seed=42)
        G = gp.sample(T=5.0, n_paths=10_000)
        assert np.all(G >= 0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            GammaProcess(variance_rate=0.0)


class TestInverseGaussian:
    def test_mean(self):
        ig = InverseGaussianProcess(delta=0.5, seed=42)
        samples = ig.sample(T=2.0, n_paths=100_000)
        assert samples.mean() == pytest.approx(2.0, rel=0.05)

    def test_non_negative(self):
        ig = InverseGaussianProcess(delta=1.0, seed=42)
        samples = ig.sample(T=1.0, n_paths=10_000)
        assert np.all(samples >= 0)

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            InverseGaussianProcess(delta=0.0)
