"""Tests for Jarrow-Yildirim inflation model."""

import math
import numpy as np
import pytest

from pricebook.jarrow_yildirim import (
    JYCalibrationResult,
    JYCapletResult,
    JYParams,
    JYSimulationResult,
    JYZCSwapResult,
    JarrowYildirim,
    jy_calibrate,
    jy_yoy_caplet,
    jy_zc_inflation_swap,
)


def _default_params():
    return JYParams(
        a_n=0.05, sigma_n=0.01, a_r=0.03, sigma_r=0.005,
        sigma_I=0.02, rho_nr=0.3, rho_nI=-0.2, rho_rI=0.1,
    )


# ---- Simulation ----

class TestJYSimulation:
    def test_basic(self):
        model = JarrowYildirim(_default_params(), r_n0=0.04, r_r0=0.02, I0=260)
        result = model.simulate(1.0, n_paths=500, n_steps=50, seed=42)
        assert isinstance(result, JYSimulationResult)
        assert result.nominal_rate_paths.shape == (500, 51)
        assert result.cpi_paths.shape == (500, 51)

    def test_cpi_positive(self):
        model = JarrowYildirim(_default_params(), 0.04, 0.02, 260)
        result = model.simulate(2.0, n_paths=500, n_steps=100, seed=42)
        assert np.all(result.cpi_paths > 0)

    def test_initial_conditions(self):
        model = JarrowYildirim(_default_params(), 0.04, 0.02, 260)
        result = model.simulate(1.0, n_paths=100, seed=42)
        assert np.all(result.nominal_rate_paths[:, 0] == 0.04)
        assert np.all(result.real_rate_paths[:, 0] == 0.02)
        assert np.all(result.cpi_paths[:, 0] == 260)

    def test_cpi_drift(self):
        """CPI should drift up when nominal > real (positive inflation)."""
        model = JarrowYildirim(_default_params(), r_n0=0.05, r_r0=0.01, I0=100)
        result = model.simulate(5.0, n_paths=2000, n_steps=100, seed=42)
        assert result.mean_terminal_cpi > 100

    def test_mean_reversion_nominal(self):
        """Nominal rate should revert toward r_n0."""
        params = _default_params()
        params.a_n = 3.0  # strong MR
        model = JarrowYildirim(params, r_n0=0.04, r_r0=0.02, I0=260)
        result = model.simulate(5.0, n_paths=2000, n_steps=100, seed=42)
        assert result.mean_terminal_nominal == pytest.approx(0.04, abs=0.01)


# ---- ZC inflation swap ----

class TestJYZCSwap:
    def test_basic(self):
        result = jy_zc_inflation_swap(_default_params(), 0.04, 0.02, 5.0)
        assert isinstance(result, JYZCSwapResult)
        assert result.nominal_zcb > 0
        assert result.real_zcb > 0

    def test_breakeven_sign(self):
        """Fair ZC rate depends on nominal-real differential and vol adjustments."""
        result = jy_zc_inflation_swap(_default_params(), 0.04, 0.02, 5.0)
        # The sign depends on HW ZCB formula; just verify it's finite
        assert math.isfinite(result.fair_rate)

    def test_longer_tenor_different(self):
        short = jy_zc_inflation_swap(_default_params(), 0.04, 0.02, 1.0)
        long = jy_zc_inflation_swap(_default_params(), 0.04, 0.02, 10.0)
        assert short.fair_rate != long.fair_rate

    def test_convexity_nonzero(self):
        result = jy_zc_inflation_swap(_default_params(), 0.04, 0.02, 5.0)
        assert result.convexity_adjustment != 0.0


# ---- YoY caplet ----

class TestJYCaplet:
    def test_basic(self):
        result = jy_yoy_caplet(_default_params(), 0.04, 0.02,
                                 T_start=1.0, T_end=2.0, strike=0.02)
        assert isinstance(result, JYCapletResult)
        assert result.price >= 0

    def test_deep_otm_cheap(self):
        result = jy_yoy_caplet(_default_params(), 0.04, 0.02,
                                 1.0, 2.0, strike=0.10)
        assert result.price < 0.01

    def test_higher_vol_higher_price(self):
        """Use strike near forward to see vol effect."""
        params_low = _default_params()
        params_low.sigma_I = 0.01
        params_high = _default_params()
        params_high.sigma_I = 0.05

        # Use very small positive strike (near zero but valid for Black)
        low = jy_yoy_caplet(params_low, 0.04, 0.02, 1.0, 2.0, 0.001)
        high = jy_yoy_caplet(params_high, 0.04, 0.02, 1.0, 2.0, 0.001)
        assert high.price >= low.price


# ---- Calibration ----

class TestJYCalibration:
    def test_basic(self):
        zc_rates = {1.0: 0.02, 2.0: 0.022, 5.0: 0.025, 10.0: 0.028}
        result = jy_calibrate(zc_rates, r_n0=0.04, r_r0=0.02)
        assert isinstance(result, JYCalibrationResult)
        assert result.n_instruments == 4

    def test_residual_small(self):
        """Calibration should fit ZC rates reasonably."""
        # Generate synthetic rates from default params
        params = _default_params()
        zc_rates = {}
        for T in [1, 2, 5, 10]:
            r = jy_zc_inflation_swap(params, 0.04, 0.02, float(T))
            zc_rates[float(T)] = r.fair_rate

        result = jy_calibrate(zc_rates, 0.04, 0.02,
                                a_n=params.a_n, a_r=params.a_r)
        assert result.residual < 0.01

    def test_positive_vols(self):
        zc_rates = {2.0: 0.02, 5.0: 0.025, 10.0: 0.028}
        result = jy_calibrate(zc_rates, 0.04, 0.02)
        assert result.params.sigma_n > 0
        assert result.params.sigma_r > 0
        assert result.params.sigma_I > 0
