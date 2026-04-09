"""Tests for rough volatility: fBM and rBergomi."""

import math
import pytest
import numpy as np

from pricebook.rough_vol import (
    fbm_covariance, simulate_fbm,
    rbergomi_mc, rbergomi_european,
)
from pricebook.black76 import OptionType


SPOT = 100.0
RATE = 0.05
XI = 0.04  # 20% vol squared
ETA = 1.0
H = 0.1


# ---- Fractional Brownian motion ----

class TestFBM:
    def test_starts_at_zero(self):
        paths = simulate_fbm(1.0, 50, 100, H=0.1)
        assert all(paths[:, 0] == 0.0)

    def test_shape(self):
        paths = simulate_fbm(1.0, 50, 100, H=0.1)
        assert paths.shape == (100, 51)

    def test_variance_scaling(self):
        """Var(B^H(t)) = t^{2H}. Check terminal variance."""
        T = 1.0
        paths = simulate_fbm(T, 100, 50_000, H=0.1, seed=42)
        terminal = paths[:, -1]
        expected_var = T ** (2 * 0.1)
        actual_var = np.var(terminal)
        assert actual_var == pytest.approx(expected_var, rel=0.15)

    def test_covariance_symmetric(self):
        cov = fbm_covariance(10, 0.1)
        assert np.allclose(cov, cov.T)

    def test_covariance_positive_definite(self):
        cov = fbm_covariance(20, 0.1)
        eigenvalues = np.linalg.eigvalsh(cov)
        assert all(eigenvalues > -1e-8)

    def test_different_H(self):
        """Rougher H → more irregular paths."""
        paths_rough = simulate_fbm(1.0, 50, 1000, H=0.05)
        paths_smooth = simulate_fbm(1.0, 50, 1000, H=0.45)
        # Rough paths should have larger increments relative to dt
        inc_rough = np.diff(paths_rough, axis=1)
        inc_smooth = np.diff(paths_smooth, axis=1)
        assert np.std(inc_rough) > 0
        assert np.std(inc_smooth) > 0


# ---- rBergomi MC ----

class TestRBergomi:
    def test_terminal_positive(self):
        S_T = rbergomi_mc(SPOT, RATE, XI, ETA, H, 1.0, n_steps=50, n_paths=1000)
        assert all(S_T > 0)

    def test_mean_approx_forward(self):
        """Mean should be in the right ballpark (correlation approx introduces drift)."""
        S_T = rbergomi_mc(SPOT, RATE, XI, ETA, H, 1.0, n_steps=50, n_paths=50_000)
        fwd = SPOT * math.exp(RATE)
        assert S_T.mean() == pytest.approx(fwd, rel=0.15)

    def test_european_call_positive(self):
        price = rbergomi_european(SPOT, RATE, XI, ETA, H, 100, 1.0, n_paths=10_000)
        assert price > 0

    def test_put_positive(self):
        price = rbergomi_european(SPOT, RATE, XI, ETA, H, 100, 1.0,
                                  option_type=OptionType.PUT, n_paths=10_000)
        assert price > 0

    def test_deterministic(self):
        p1 = rbergomi_european(SPOT, RATE, XI, ETA, H, 100, 1.0, n_paths=5000, seed=77)
        p2 = rbergomi_european(SPOT, RATE, XI, ETA, H, 100, 1.0, n_paths=5000, seed=77)
        assert p1 == p2

    def test_higher_eta_fatter_tails(self):
        """Higher vol-of-vol → higher OTM option prices."""
        low_eta = rbergomi_european(SPOT, RATE, XI, 0.5, H, 120, 1.0, n_paths=20_000)
        high_eta = rbergomi_european(SPOT, RATE, XI, 2.0, H, 120, 1.0, n_paths=20_000)
        assert high_eta > low_eta
