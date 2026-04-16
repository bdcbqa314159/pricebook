"""Tests for equity vol surface: SSVI, vol cube, forward vol."""

import math

import numpy as np
import pytest

from pricebook.equity_smile import (
    EquityCubeNode,
    EquityVolCube,
    ForwardVolResult,
    SSVIParams,
    build_equity_vol_cube,
    calibrate_equity_sabr_tenor,
    forward_vol,
    ssvi_fit,
    ssvi_vol,
    sticky_delta_dynamics,
    sticky_strike_dynamics,
)


# ---- SSVI ----

class TestSSVI:
    def _sample_data(self):
        """Generate synthetic smile data."""
        tenors = [0.5, 1.0, 2.0]
        ks = [-0.2, -0.1, 0.0, 0.1, 0.2]
        vols = {}
        kgrid = {}
        for T in tenors:
            # Synthetic smile
            kgrid[T] = ks
            vols[T] = [0.20 + 0.05 * abs(k) - 0.02 * k for k in ks]
        return tenors, kgrid, vols

    def test_basic_fit(self):
        tenors, kgrid, vols = self._sample_data()
        params = ssvi_fit(tenors, kgrid, vols)
        assert isinstance(params, SSVIParams)
        assert abs(params.rho) <= 1
        assert params.eta > 0

    def test_atm_vars_populated(self):
        tenors, kgrid, vols = self._sample_data()
        params = ssvi_fit(tenors, kgrid, vols)
        for T in tenors:
            assert T in params.atm_vars
            assert params.atm_vars[T] > 0

    def test_ssvi_vol_evaluation(self):
        tenors, kgrid, vols = self._sample_data()
        params = ssvi_fit(tenors, kgrid, vols)
        v = ssvi_vol(0.0, 1.0, params)
        assert v > 0

    def test_reproduces_atm(self):
        """SSVI should approximately reproduce ATM variance."""
        tenors, kgrid, vols = self._sample_data()
        params = ssvi_fit(tenors, kgrid, vols)
        # At k=0, variance ≈ atm_vars
        for T in tenors:
            v = ssvi_vol(0.0, T, params)
            expected = math.sqrt(params.atm_vars[T] / T)
            assert v == pytest.approx(expected, abs=0.05)

    def test_extrapolation(self):
        """Out-of-range tenor should still give a vol."""
        tenors, kgrid, vols = self._sample_data()
        params = ssvi_fit(tenors, kgrid, vols)
        # Very short and very long
        v_short = ssvi_vol(0.0, 0.1, params)
        v_long = ssvi_vol(0.0, 5.0, params)
        assert v_short > 0
        assert v_long > 0


# ---- Vol cube ----

class TestEquityVolCube:
    def _build(self):
        tenors = [0.5, 1.0, 2.0]
        quotes = {
            0.5: {"atm": 0.20, "25c": 0.19, "25p": 0.23},
            1.0: {"atm": 0.22, "25c": 0.21, "25p": 0.25},
            2.0: {"atm": 0.23, "25c": 0.22, "25p": 0.27},
        }
        return build_equity_vol_cube(100, 0.03, 0.02, tenors, quotes)

    def test_build(self):
        cube = self._build()
        assert isinstance(cube, EquityVolCube)
        assert len(cube.nodes) == 3

    def test_atm_vol(self):
        cube = self._build()
        v = cube.atm_vol(1.0)
        assert 0.15 < v < 0.30

    def test_vol_at_strike(self):
        cube = self._build()
        v = cube.vol(1.0, 100)
        assert v > 0

    def test_put_skew(self):
        """OTM put vol > ATM vol (negative skew typical for equity)."""
        cube = self._build()
        v_atm = cube.vol(1.0, 100)
        v_otm_put = cube.vol(1.0, 80)
        assert v_otm_put >= v_atm * 0.95  # typically true with put skew

    def test_intermediate_tenor(self):
        cube = self._build()
        v_05 = cube.atm_vol(0.5)
        v_10 = cube.atm_vol(1.0)
        v_075 = cube.atm_vol(0.75)
        # Should be bracketed
        assert min(v_05, v_10) - 0.02 <= v_075 <= max(v_05, v_10) + 0.02


class TestCalibrateEquitySabrTenor:
    def test_basic(self):
        node = calibrate_equity_sabr_tenor(100, 0.03, 0.02, 1.0,
                                             atm_vol=0.20, vol_25d_call=0.19,
                                             vol_25d_put=0.23, beta=1.0)
        assert isinstance(node, EquityCubeNode)
        assert node.alpha > 0
        assert node.nu > 0

    def test_atm_vol_matches(self):
        node = calibrate_equity_sabr_tenor(100, 0.03, 0.02, 1.0, 0.20, 0.19, 0.23)
        assert node.atm_vol == pytest.approx(0.20, abs=0.02)


# ---- Forward vol ----

class TestForwardVol:
    def _build_cube(self):
        tenors = [0.5, 1.0, 2.0]
        quotes = {
            0.5: {"atm": 0.20, "25c": 0.20, "25p": 0.20},
            1.0: {"atm": 0.22, "25c": 0.22, "25p": 0.22},
            2.0: {"atm": 0.24, "25c": 0.24, "25p": 0.24},
        }
        return build_equity_vol_cube(100, 0.03, 0.02, tenors, quotes)

    def test_basic(self):
        cube = self._build_cube()
        result = forward_vol(cube, 0.5, 1.0)
        assert isinstance(result, ForwardVolResult)
        assert result.forward_vol > 0

    def test_variance_additivity(self):
        """T2 × σ²(T2) = T1 × σ²(T1) + (T2-T1) × σ²_fwd."""
        cube = self._build_cube()
        result = forward_vol(cube, 0.5, 1.0)
        v1 = cube.atm_vol(0.5)
        v2 = cube.atm_vol(1.0)
        expected = (v2**2 * 1.0 - v1**2 * 0.5) / 0.5
        assert result.forward_variance == pytest.approx(expected, rel=0.01)

    def test_raise_when_t2_le_t1(self):
        cube = self._build_cube()
        with pytest.raises(ValueError):
            forward_vol(cube, 1.0, 0.5)

    def test_with_strike(self):
        cube = self._build_cube()
        result = forward_vol(cube, 0.5, 1.0, strike=100)
        assert result.forward_vol > 0


# ---- Smile dynamics ----

class TestSmileDynamics:
    def _build_cube(self):
        tenors = [1.0]
        quotes = {1.0: {"atm": 0.20, "25c": 0.19, "25p": 0.23}}
        return build_equity_vol_cube(100, 0.03, 0.02, tenors, quotes)

    def test_sticky_strike(self):
        cube = self._build_cube()
        v = sticky_strike_dynamics(cube, 1.0, 100)
        assert v > 0

    def test_sticky_delta(self):
        cube = self._build_cube()
        v = sticky_delta_dynamics(cube, 1.0, 0.0)
        assert v > 0
