"""Tests for LMM calibration and multi-factor SABR."""

import math
import pytest
import numpy as np

from pricebook.lmm_calibration import (
    rebonato_swaption_vol, exponential_correlation,
    calibrate_lmm_vols,
    MultiFactorSABR, SABRSlice, calibrate_multi_factor_sabr,
)
from pricebook.sabr import sabr_implied_vol


# ---- Rebonato approximation ----

class TestRebonato:
    def test_positive_vol(self):
        rates = [0.04, 0.045, 0.05, 0.055, 0.06]
        vols = [0.20] * 5
        v = rebonato_swaption_vol(rates, vols, 0.25, 0, 4)
        assert v > 0

    def test_flat_vol_reproduces(self):
        """Flat vol + flat rates → swaption vol ≈ flat vol."""
        rates = [0.05] * 8
        vols = [0.20] * 8
        v = rebonato_swaption_vol(rates, vols, 0.25, 0, 4)
        assert v == pytest.approx(0.20, rel=0.2)

    def test_higher_vol_higher_swaption_vol(self):
        rates = [0.05] * 8
        low = rebonato_swaption_vol(rates, [0.15] * 8, 0.25, 0, 4)
        high = rebonato_swaption_vol(rates, [0.25] * 8, 0.25, 0, 4)
        assert high > low


class TestCorrelation:
    def test_all_ones_at_zero_beta(self):
        """exp(-0×|i-j|) = 1 for all i,j → perfect correlation."""
        rho = exponential_correlation(5, 0.0)
        assert np.allclose(rho, np.ones((5, 5)))

    def test_symmetric(self):
        rho = exponential_correlation(5, 0.1)
        assert np.allclose(rho, rho.T)

    def test_diagonal_one(self):
        rho = exponential_correlation(5, 0.1)
        assert all(rho[i, i] == pytest.approx(1.0) for i in range(5))

    def test_off_diagonal_less_than_one(self):
        rho = exponential_correlation(5, 0.1)
        assert rho[0, 4] < 1.0


# ---- LMM calibration ----

class TestLMMCalibration:
    def test_calibrates(self):
        rates = [0.04, 0.045, 0.05, 0.055, 0.06, 0.055, 0.05, 0.045]
        targets = {
            (0, 4): 0.20,
            (4, 4): 0.18,
        }
        result = calibrate_lmm_vols(rates, targets)
        assert result.rmse < 0.02

    def test_fitted_vols_returned(self):
        rates = [0.05] * 8
        targets = {(0, 4): 0.20}
        result = calibrate_lmm_vols(rates, targets)
        assert (0, 4) in result.fitted_swaption_vols
        assert result.fitted_swaption_vols[(0, 4)] > 0

    def test_calibrated_vols_positive(self):
        rates = [0.05] * 8
        targets = {(0, 4): 0.20, (2, 4): 0.19}
        result = calibrate_lmm_vols(rates, targets)
        assert all(v > 0 for v in result.calibrated_vols)


# ---- Multi-factor SABR ----

class TestMultiFactorSABR:
    def test_single_slice(self):
        slices = [SABRSlice(1.0, 0.2, 0.5, -0.3, 0.4)]
        model = MultiFactorSABR(slices)
        v = model.vol(0.05, 0.05, 1.0)
        expected = sabr_implied_vol(0.05, 0.05, 1.0, 0.2, 0.5, -0.3, 0.4)
        assert v == pytest.approx(expected)

    def test_interpolation(self):
        slices = [
            SABRSlice(1.0, 0.20, 0.5, -0.3, 0.4),
            SABRSlice(5.0, 0.15, 0.5, -0.2, 0.3),
        ]
        model = MultiFactorSABR(slices)
        v1 = model.vol(0.05, 0.05, 1.0)
        v5 = model.vol(0.05, 0.05, 5.0)
        v3 = model.vol(0.05, 0.05, 3.0)
        # Interpolated vol should be between endpoints
        assert min(v1, v5) <= v3 <= max(v1, v5) + 0.01

    def test_extrapolation_flat(self):
        slices = [SABRSlice(1.0, 0.20, 0.5, -0.3, 0.4)]
        model = MultiFactorSABR(slices)
        v = model.vol(0.05, 0.05, 10.0)
        assert v > 0

    def test_smile(self):
        """ATM and OTM should differ with non-zero rho/nu."""
        slices = [SABRSlice(1.0, 0.20, 0.5, -0.3, 0.4)]
        model = MultiFactorSABR(slices)
        v_atm = model.vol(0.05, 0.05, 1.0)
        v_otm = model.vol(0.05, 0.06, 1.0)
        assert v_atm != pytest.approx(v_otm, abs=1e-6)

    def test_expiries_sorted(self):
        slices = [SABRSlice(5.0, 0.15, 0.5, -0.2, 0.3), SABRSlice(1.0, 0.20, 0.5, -0.3, 0.4)]
        model = MultiFactorSABR(slices)
        assert model.expiries == [1.0, 5.0]


class TestCalibrateMultiFactorSABR:
    def test_calibrates(self):
        forwards = [0.05, 0.05]
        expiries = [1.0, 5.0]
        strikes = [[0.04, 0.05, 0.06], [0.04, 0.05, 0.06]]
        # Generate target vols from known SABR params
        vols = [
            [sabr_implied_vol(0.05, k, 1.0, 0.20, 0.5, -0.3, 0.4) for k in strikes[0]],
            [sabr_implied_vol(0.05, k, 5.0, 0.15, 0.5, -0.2, 0.3) for k in strikes[1]],
        ]
        model = calibrate_multi_factor_sabr(forwards, expiries, strikes, vols)
        assert len(model.slices) == 2
        # Should reprice ATM
        v1 = model.vol(0.05, 0.05, 1.0)
        target1 = vols[0][1]
        assert v1 == pytest.approx(target1, rel=0.05)
