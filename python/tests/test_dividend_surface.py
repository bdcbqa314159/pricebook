"""Tests for dividend surface and joint calibration."""

import pytest
import math
import numpy as np

from pricebook.equity.dividend_surface import (
    DividendSurface, build_dividend_surface, simulate_dividend_surface,
)
from pricebook.equity.joint_calibration import (
    joint_calibrate, decompose_forward_error,
)


class TestDividendSurface:
    def test_build_from_futures(self):
        s = build_dividend_surface(
            100,
            [{"T": 0.25, "price": 0.5}, {"T": 0.5, "price": 1.0},
             {"T": 1.0, "price": 2.0}, {"T": 2.0, "price": 4.0}],
        )
        assert len(s.tenors) == 4
        assert all(y > 0 for y in s.yield_levels)

    def test_build_with_options(self):
        s = build_dividend_surface(
            100,
            [{"T": 0.5, "price": 1.0}, {"T": 1.0, "price": 2.0}],
            [{"T": 0.5, "iv": 0.15}, {"T": 1.0, "iv": 0.20}],
        )
        assert s.yield_vols[0] == pytest.approx(0.15)

    def test_interpolate(self):
        s = DividendSurface(
            np.array([0.5, 1.0, 2.0]),
            np.array([0.02, 0.025, 0.03]),
            np.array([0.15, 0.18, 0.20]),
            -0.3,
        )
        level, vol = s.interpolate(0.75)
        assert 0.02 < level < 0.025
        assert 0.15 < vol < 0.18

    def test_to_dict(self):
        s = build_dividend_surface(100, [{"T": 1.0, "price": 2.0}])
        d = s.to_dict()
        assert "tenors" in d
        assert "yield_levels" in d

    def test_simulate(self):
        s = DividendSurface(
            np.array([1.0, 2.0]),
            np.array([0.02, 0.025]),
            np.array([0.005, 0.006]),
            -0.3,
        )
        result = simulate_dividend_surface(s, 100, 0.05, 1.0, n_paths=5000)
        assert result.spot_paths.shape == (5000, 101)
        assert result.terminal_spot_mean > 50
        assert result.terminal_yield_mean > 0

    def test_simulate_custom_params(self):
        """spot_vol and kappa_q should be parameterised, not hardcoded."""
        s = DividendSurface(
            np.array([1.0]), np.array([0.02]),
            np.array([0.005]), -0.3,
        )
        r1 = simulate_dividend_surface(s, 100, 0.05, 1.0, spot_vol=0.10, kappa_q=5.0, n_paths=1000)
        r2 = simulate_dividend_surface(s, 100, 0.05, 1.0, spot_vol=0.40, kappa_q=0.5, n_paths=1000)
        # Higher vol → higher std
        assert r2.terminal_spot_std > r1.terminal_spot_std

    def test_simulate_to_dict(self):
        s = DividendSurface(
            np.array([1.0]), np.array([0.02]),
            np.array([0.005]), -0.3,
        )
        result = simulate_dividend_surface(s, 100, 0.05, 1.0, n_paths=500)
        d = result.to_dict()
        assert "terminal_spot_mean" in d
        assert d["n_paths"] == 500


class TestJointCalibration:
    def _synthetic_data(self, spot, rate, q, sigma):
        """Generate consistent vol surface + forward data."""
        vol_data = []
        div_data = []
        for T in [0.25, 0.5, 1.0]:
            fwd = spot * math.exp((rate - q) * T)
            strikes = [spot * m for m in [0.90, 0.95, 1.00, 1.05, 1.10]]
            vols = [sigma] * 5  # flat vol
            vol_data.append({"T": T, "strikes": strikes, "vols": vols})
            div_data.append({"T": T, "fwd": fwd})
        return vol_data, div_data

    def test_bsm_continuous_roundtrip(self):
        spot, rate, q, sigma = 100, 0.05, 0.02, 0.20
        vol_data, div_data = self._synthetic_data(spot, rate, q, sigma)

        result = joint_calibrate(spot, vol_data, div_data, rate, model="bsm+continuous")
        assert result.rmse_vol < 0.005
        assert result.rmse_fwd < 0.005
        assert result.vol_params["sigma"] == pytest.approx(sigma, abs=0.01)
        assert result.div_params["q"] == pytest.approx(q, abs=0.005)

    def test_term_continuous(self):
        spot, rate, q, sigma = 100, 0.05, 0.02, 0.20
        vol_data, div_data = self._synthetic_data(spot, rate, q, sigma)

        result = joint_calibrate(spot, vol_data, div_data, rate, model="term+continuous")
        assert result.rmse_vol < 0.005
        assert result.div_params["q"] == pytest.approx(q, abs=0.005)

    def test_unknown_model(self):
        with pytest.raises(ValueError, match="Unknown model"):
            joint_calibrate(100, [], [], 0.05, model="unknown")

    def test_to_dict(self):
        vol_data, div_data = self._synthetic_data(100, 0.05, 0.02, 0.20)
        result = joint_calibrate(100, vol_data, div_data, 0.05)
        d = result.to_dict()
        assert "rmse_vol" in d
        assert "rmse_fwd" in d


class TestForwardErrorDecomp:
    def test_decomposition(self):
        market_fwds = [{"T": 0.5, "fwd": 101.5}, {"T": 1.0, "fwd": 103.0}]
        result = decompose_forward_error(100, {"sigma": 0.20}, {"q": 0.02},
                                          market_fwds, 0.05)
        assert math.isfinite(result.total_error)
        assert math.isfinite(result.div_component)
        assert len(result.per_tenor) == 2

    def test_to_dict(self):
        result = decompose_forward_error(100, {"sigma": 0.20}, {"q": 0.02},
                                          [{"T": 1.0, "fwd": 103}], 0.05)
        d = result.to_dict()
        assert "total_error" in d
