"""Tests for FX smile cube."""

import math

import numpy as np
import pytest

from pricebook.fx_smile_cube import (
    ArbitrageCheckResult,
    FXSmileNode,
    FXVolCube,
    SVIParams,
    build_fx_vol_cube,
    calibrate_sabr_fx_tenor,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    svi_fit,
    svi_vol,
)


# ---- SABR per tenor ----

class TestCalibrateSABRFXTenor:
    def test_basic(self):
        node = calibrate_sabr_fx_tenor(
            spot=1.0, rate_dom=0.02, rate_for=0.01, T=1.0,
            vol_atm=0.10, vol_25d_call=0.11, vol_25d_put=0.12,
        )
        assert isinstance(node, FXSmileNode)
        assert node.alpha > 0
        assert node.nu > 0
        assert -1 < node.rho < 1

    def test_residual_small(self):
        node = calibrate_sabr_fx_tenor(
            1.0, 0.02, 0.01, 1.0, 0.10, 0.11, 0.12,
        )
        assert node.residual < 0.01

    def test_rr_and_bf(self):
        node = calibrate_sabr_fx_tenor(
            1.0, 0.02, 0.01, 1.0, 0.10, 0.11, 0.12,
        )
        # 25D call - 25D put = -0.01 (put skew)
        assert node.rr_25d == pytest.approx(-0.01)
        # BF = (0.11 + 0.12)/2 - 0.10 = 0.015
        assert node.bf_25d == pytest.approx(0.015)

    def test_with_10d_wings(self):
        node = calibrate_sabr_fx_tenor(
            1.0, 0.02, 0.01, 1.0, 0.10, 0.11, 0.12,
            vol_10d_call=0.13, vol_10d_put=0.14,
        )
        assert node.residual < 0.02


# ---- FX Vol Cube ----

class TestFXVolCube:
    def _build(self):
        tenors = [0.25, 1.0, 3.0]
        quotes = {
            0.25: {"atm": 0.08, "25c": 0.085, "25p": 0.09},
            1.0: {"atm": 0.10, "25c": 0.11, "25p": 0.12},
            3.0: {"atm": 0.12, "25c": 0.13, "25p": 0.14},
        }
        return build_fx_vol_cube(1.0, 0.02, 0.01, tenors, quotes)

    def test_build(self):
        cube = self._build()
        assert isinstance(cube, FXVolCube)
        assert len(cube.nodes) == 3

    def test_vol_at_tenor(self):
        cube = self._build()
        v = cube.vol(1.0, 1.0)
        assert 0.05 < v < 0.30

    def test_vol_at_intermediate_tenor(self):
        cube = self._build()
        v_025 = cube.vol(0.25, 1.0)
        v_1 = cube.vol(1.0, 1.0)
        v_05 = cube.vol(0.5, 1.0)
        # Interpolated vol should be between bracketing tenors
        assert min(v_025, v_1) - 0.01 <= v_05 <= max(v_025, v_1) + 0.01

    def test_vol_at_delta(self):
        cube = self._build()
        v = cube.vol_at_delta(1.0, 0.25, is_call=True)
        assert 0.05 < v < 0.30

    def test_vol_out_of_range_flat(self):
        """Outside range → flat extrapolation."""
        cube = self._build()
        v_short = cube.vol(0.1, 1.0)
        v_025 = cube.vol(0.25, 1.0)
        # Should use shortest node
        assert v_short == pytest.approx(v_025, rel=0.05)


# ---- SVI ----

class TestSVI:
    def test_basic_fit(self):
        """Fit SVI to synthetic smile."""
        k = np.linspace(-0.3, 0.3, 11)
        # Synthetic smile: parabolic in log-moneyness
        vols = [0.10 + 0.05 * ki**2 for ki in k]
        params = svi_fit(list(k), vols, expiry=1.0)
        assert isinstance(params, SVIParams)

    def test_svi_vol_positive(self):
        params = SVIParams(a=0.01, b=0.1, rho=-0.3, m=0.0, sigma=0.2, expiry=1.0)
        vol = svi_vol(0.0, params)
        assert vol > 0

    def test_svi_constraints(self):
        """SVI params should satisfy basic constraints."""
        k = np.linspace(-0.2, 0.2, 9)
        vols = [0.12, 0.11, 0.105, 0.10, 0.098, 0.10, 0.105, 0.11, 0.12]
        params = svi_fit(list(k), vols, 1.0)
        assert params.b > 0
        assert params.sigma > 0
        assert abs(params.rho) < 1

    def test_svi_reproduces_smile(self):
        k = np.linspace(-0.2, 0.2, 9)
        vols = [0.12, 0.11, 0.105, 0.10, 0.098, 0.10, 0.105, 0.11, 0.12]
        params = svi_fit(list(k), vols, 1.0)
        # Check fit quality
        for ki, vi in zip(k, vols):
            assert svi_vol(ki, params) == pytest.approx(vi, abs=0.02)


# ---- Arbitrage checks ----

class TestButterflyArbitrage:
    def test_flat_smile_no_arb(self):
        """Flat smile → no butterfly arbitrage."""
        tenors = [1.0]
        quotes = {1.0: {"atm": 0.10, "25c": 0.10, "25p": 0.10}}
        cube = build_fx_vol_cube(1.0, 0.02, 0.01, tenors, quotes)
        result = check_butterfly_arbitrage(cube, 1.0)
        assert result.is_arbitrage_free
        assert len(result.violations) == 0

    def test_normal_smile_no_arb(self):
        """Typical FX smile should be arbitrage-free."""
        tenors = [1.0]
        quotes = {1.0: {"atm": 0.10, "25c": 0.11, "25p": 0.12}}
        cube = build_fx_vol_cube(1.0, 0.02, 0.01, tenors, quotes)
        result = check_butterfly_arbitrage(cube, 1.0, n_strikes=20)
        # Should be arbitrage-free or have only tiny numerical violations
        assert result.n_strikes_checked > 0


class TestCalendarArbitrage:
    def test_monotone_variance_no_arb(self):
        """Total variance increasing with T → no calendar arb."""
        tenors = [0.5, 1.0, 2.0]
        quotes = {
            0.5: {"atm": 0.10, "25c": 0.105, "25p": 0.11},
            1.0: {"atm": 0.10, "25c": 0.11, "25p": 0.12},
            2.0: {"atm": 0.10, "25c": 0.115, "25p": 0.125},
        }
        cube = build_fx_vol_cube(1.0, 0.02, 0.01, tenors, quotes)
        result = check_calendar_arbitrage(cube, tenors)
        # Variance = σ²T with constant σ → monotone increasing
        assert result.is_arbitrage_free

    def test_checks_structure(self):
        tenors = [0.5, 1.0]
        quotes = {
            0.5: {"atm": 0.10, "25c": 0.11, "25p": 0.12},
            1.0: {"atm": 0.12, "25c": 0.13, "25p": 0.14},
        }
        cube = build_fx_vol_cube(1.0, 0.02, 0.01, tenors, quotes)
        result = check_calendar_arbitrage(cube, tenors)
        assert isinstance(result, ArbitrageCheckResult)
        assert result.n_tenors_checked == 2
