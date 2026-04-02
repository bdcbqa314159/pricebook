"""Tests for jump processes."""

import pytest
import math
import numpy as np

from pricebook.jump_process import (
    PoissonProcess,
    CompoundPoissonProcess,
    MertonJumpDiffusion,
    VarianceGammaProcess,
)


class TestPoisson:
    def test_mean(self):
        pp = PoissonProcess(intensity=5.0, seed=42)
        N = pp.sample(T=2.0, n_paths=100_000)
        assert N.mean() == pytest.approx(10.0, rel=0.02)

    def test_variance(self):
        pp = PoissonProcess(intensity=5.0, seed=42)
        N = pp.sample(T=2.0, n_paths=100_000)
        assert N.var() == pytest.approx(10.0, rel=0.05)

    def test_zero_intensity(self):
        pp = PoissonProcess(intensity=0.0, seed=42)
        N = pp.sample(T=10.0, n_paths=1000)
        assert N.sum() == 0

    def test_inter_arrivals_exponential(self):
        pp = PoissonProcess(intensity=2.0, seed=42)
        ia = pp.inter_arrivals(n_events=1, n_paths=100_000)
        assert ia.mean() == pytest.approx(0.5, rel=0.02)  # 1/λ

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            PoissonProcess(intensity=-1.0)


class TestCompoundPoisson:
    def test_mean(self):
        """E[X(T)] = λ * E[J] * T."""
        cpp = CompoundPoissonProcess(intensity=3.0, jump_mean=0.5, jump_std=0.1, seed=42)
        X = cpp.sample(T=2.0, n_paths=100_000)
        expected = 3.0 * 0.5 * 2.0  # = 3.0
        assert X.mean() == pytest.approx(expected, rel=0.05)

    def test_zero_intensity(self):
        cpp = CompoundPoissonProcess(intensity=0.0, jump_mean=1.0, seed=42)
        X = cpp.sample(T=5.0, n_paths=1000)
        np.testing.assert_array_equal(X, 0.0)


class TestMertonJumpDiffusion:
    def test_positive_prices(self):
        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=1.0,
                                   jump_mean=-0.05, jump_std=0.10)
        st = mjd.terminal(100, T=1.0, n_paths=10_000)
        assert np.all(st > 0)

    def test_zero_jumps_matches_gbm(self):
        """λ=0 → no jumps → GBM."""
        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=0.0,
                                   jump_mean=0.0, jump_std=0.0)
        st = mjd.terminal(100, T=1.0, n_paths=100_000)
        expected_mean = 100 * math.exp(0.05)
        assert st.mean() == pytest.approx(expected_mean, rel=0.02)

    def test_mean_matches_drift(self):
        """E[S_T] = S_0 * exp(mu * T) regardless of jumps (risk-neutral)."""
        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=2.0,
                                   jump_mean=-0.05, jump_std=0.10)
        st = mjd.terminal(100, T=1.0, n_paths=200_000)
        expected = 100 * math.exp(0.05)
        assert st.mean() == pytest.approx(expected, rel=0.02)

    def test_char_func_at_zero(self):
        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=1.0,
                                   jump_mean=-0.05, jump_std=0.10)
        phi = mjd.char_func(T=1.0)
        assert abs(phi(0)) == pytest.approx(1.0, abs=1e-10)

    def test_cos_matches_mc(self):
        """COS with Merton char func ≈ MC price."""
        from pricebook.cos_method import cos_price
        from pricebook.black76 import OptionType

        mjd = MertonJumpDiffusion(mu=0.05, sigma=0.20, lam=1.0,
                                   jump_mean=-0.05, jump_std=0.10)

        # MC price
        st = mjd.terminal(100, T=1.0, n_paths=200_000, seed=42)
        mc_call = math.exp(-0.05) * np.maximum(st - 100, 0).mean()

        # COS price
        cos_call = cos_price(mjd.char_func(T=1.0), 100, 100, 0.05, 1.0,
                             OptionType.CALL, N=128, L=15)

        assert cos_call == pytest.approx(mc_call, rel=0.05)


class TestVarianceGamma:
    def test_positive_prices(self):
        vg = VarianceGammaProcess(sigma=0.20, theta=-0.10, nu=0.25)
        st = vg.terminal(100, rate=0.05, T=1.0, n_paths=10_000)
        assert np.all(st > 0)

    def test_mean_risk_neutral(self):
        """E[S_T] = S_0 * exp(r*T) under risk-neutral measure."""
        vg = VarianceGammaProcess(sigma=0.20, theta=-0.10, nu=0.25)
        st = vg.terminal(100, rate=0.05, T=1.0, n_paths=200_000)
        expected = 100 * math.exp(0.05)
        assert st.mean() == pytest.approx(expected, rel=0.02)

    def test_char_func_at_zero(self):
        vg = VarianceGammaProcess(sigma=0.20, theta=-0.10, nu=0.25)
        phi = vg.char_func(rate=0.05, T=1.0)
        assert abs(phi(0)) == pytest.approx(1.0, abs=1e-10)

    def test_cos_matches_mc(self):
        """COS with VG char func ≈ MC."""
        from pricebook.cos_method import cos_price
        from pricebook.black76 import OptionType

        vg = VarianceGammaProcess(sigma=0.20, theta=-0.10, nu=0.25)

        st = vg.terminal(100, rate=0.05, T=1.0, n_paths=200_000, seed=42)
        mc_call = math.exp(-0.05) * np.maximum(st - 100, 0).mean()

        cos_call = cos_price(vg.char_func(rate=0.05, T=1.0), 100, 100, 0.05, 1.0,
                             OptionType.CALL, N=128, L=15)

        assert cos_call == pytest.approx(mc_call, rel=0.05)

    def test_nu_zero_raises(self):
        with pytest.raises(ValueError):
            VarianceGammaProcess(sigma=0.20, theta=0.0, nu=0.0)
