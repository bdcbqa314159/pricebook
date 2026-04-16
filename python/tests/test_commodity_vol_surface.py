"""Tests for commodity vol surface and Kirk spread."""

import math

import numpy as np
import pytest

from pricebook.commodity_vol_surface import (
    CommoditySmileNode,
    CommodityVolCube,
    KirkResult,
    build_commodity_cube,
    calibrate_commodity_sabr,
    kirk_spread_smile,
)
from pricebook.sabr import sabr_implied_vol


# ---- SABR per tenor ----

class TestCalibrateCommoditySABR:
    def _synthetic_smile(self, F=80.0, T=1.0, beta=1.0):
        alpha, rho, nu = 0.3, -0.3, 0.5
        offsets = [-10, -5, 0, 5, 10]
        strikes = [F + o for o in offsets]
        vols = [sabr_implied_vol(F, k, T, alpha, beta, rho, nu) for k in strikes]
        return strikes, vols

    def test_basic(self):
        strikes, vols = self._synthetic_smile()
        node = calibrate_commodity_sabr(80.0, 1.0, strikes, vols, beta=1.0)
        assert isinstance(node, CommoditySmileNode)

    def test_residual_small(self):
        strikes, vols = self._synthetic_smile()
        node = calibrate_commodity_sabr(80.0, 1.0, strikes, vols)
        assert node.residual < 0.01

    def test_recovers_parameters(self):
        strikes, vols = self._synthetic_smile()
        node = calibrate_commodity_sabr(80.0, 1.0, strikes, vols, beta=1.0)
        # Should roughly recover alpha=0.3, rho=-0.3, nu=0.5
        assert node.alpha == pytest.approx(0.3, rel=0.3)
        assert node.rho == pytest.approx(-0.3, abs=0.2)
        assert node.nu == pytest.approx(0.5, rel=0.3)

    def test_different_beta(self):
        """β=0.5 for metals."""
        strikes, vols = self._synthetic_smile(beta=0.5)
        node = calibrate_commodity_sabr(80.0, 1.0, strikes, vols, beta=0.5)
        assert node.beta == 0.5


# ---- Commodity vol cube ----

class TestCommodityVolCube:
    def _build(self):
        tenors = [0.25, 1.0, 3.0]
        forwards = {0.25: 80.0, 1.0: 82.0, 3.0: 85.0}
        smiles = {}
        for T, F in forwards.items():
            strikes = [F - 10, F - 5, F, F + 5, F + 10]
            vols = [0.35, 0.32, 0.30, 0.32, 0.35]
            smiles[T] = (strikes, vols)
        return build_commodity_cube("WTI", tenors, forwards, smiles, beta=1.0)

    def test_build(self):
        cube = self._build()
        assert isinstance(cube, CommodityVolCube)
        assert cube.commodity == "WTI"
        assert len(cube.nodes) == 3

    def test_vol_at_tenor(self):
        cube = self._build()
        v = cube.vol(1.0, 82.0)
        assert 0.20 < v < 0.50

    def test_atm_vol(self):
        cube = self._build()
        v = cube.atm_vol(1.0)
        assert v > 0

    def test_interpolation(self):
        cube = self._build()
        v_short = cube.atm_vol(0.25)
        v_mid = cube.atm_vol(1.0)
        v_half = cube.atm_vol(0.5)
        # Should be bracketed (approximately)
        assert min(v_short, v_mid) - 0.05 <= v_half <= max(v_short, v_mid) + 0.05

    def test_forward_interpolation(self):
        cube = self._build()
        F = cube.forward(2.0)
        # Between 1.0 forward (82) and 3.0 forward (85)
        assert 82 <= F <= 85


# ---- Kirk spread ----

class TestKirkSpread:
    def test_basic(self):
        result = kirk_spread_smile(
            forward1=80, forward2=70, strike=5,
            vol1=0.30, vol2=0.25, correlation=0.8,
            rate=0.03, T=0.5,
        )
        assert isinstance(result, KirkResult)
        assert result.price > 0

    def test_higher_correlation_lower_vol(self):
        """Higher correlation → lower spread vol → lower option price."""
        low = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.3, 0.03, 0.5)
        high = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.95, 0.03, 0.5)
        assert high.implied_spread_vol < low.implied_spread_vol

    def test_zero_smile_adjustment(self):
        result = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.8, 0.03, 0.5,
                                    smile_adjustment_factor=0.0)
        assert result.smile_price == result.flat_price

    def test_positive_smile_adjustment_higher_price(self):
        base = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.8, 0.03, 0.5, 0.0)
        bumped = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.8, 0.03, 0.5, 0.05)
        assert bumped.smile_price > base.smile_price

    def test_put_call(self):
        call = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.8, 0.03, 0.5, is_call=True)
        put = kirk_spread_smile(80, 70, 5, 0.30, 0.25, 0.8, 0.03, 0.5, is_call=False)
        # Spread forward = 80 - 70 = 10, strike = 5, so ITM for call
        assert call.price > put.price
