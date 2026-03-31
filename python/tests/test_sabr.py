"""Tests for SABR model."""

import pytest
import math

from pricebook.sabr import sabr_implied_vol, sabr_price, sabr_calibrate
from pricebook.black76 import OptionType, black76_price


F = 0.03  # forward swap rate
T = 5.0
ALPHA = 0.035
BETA = 0.5
RHO = -0.25
NU = 0.4
DF = math.exp(-0.04 * T)


class TestSABRImpliedVol:
    def test_atm_vol_positive(self):
        vol = sabr_implied_vol(F, F, T, ALPHA, BETA, RHO, NU)
        assert vol > 0

    def test_otm_call_vol(self):
        vol = sabr_implied_vol(F, F * 1.5, T, ALPHA, BETA, RHO, NU)
        assert vol > 0

    def test_itm_put_vol(self):
        vol = sabr_implied_vol(F, F * 0.5, T, ALPHA, BETA, RHO, NU)
        assert vol > 0

    def test_negative_rho_skews_left(self):
        """Negative rho → higher vol for low strikes (left skew)."""
        vol_low = sabr_implied_vol(F, F * 0.8, T, ALPHA, BETA, -0.4, NU)
        vol_high = sabr_implied_vol(F, F * 1.2, T, ALPHA, BETA, -0.4, NU)
        assert vol_low > vol_high

    def test_positive_rho_skews_right(self):
        vol_low = sabr_implied_vol(F, F * 0.8, T, ALPHA, BETA, 0.4, NU)
        vol_high = sabr_implied_vol(F, F * 1.2, T, ALPHA, BETA, 0.4, NU)
        assert vol_high > vol_low

    def test_zero_nu_reduced_smile(self):
        """Low vol-of-vol → much flatter smile than high nu."""
        vol_otm_low = sabr_implied_vol(F, F * 1.3, T, ALPHA, BETA, RHO, 0.01)
        vol_atm_low = sabr_implied_vol(F, F, T, ALPHA, BETA, RHO, 0.01)
        vol_otm_high = sabr_implied_vol(F, F * 1.3, T, ALPHA, BETA, RHO, 0.5)
        vol_atm_high = sabr_implied_vol(F, F, T, ALPHA, BETA, RHO, 0.5)
        smile_low = abs(vol_otm_low - vol_atm_low)
        smile_high = abs(vol_otm_high - vol_atm_high)
        assert smile_low < smile_high

    def test_higher_nu_wider_smile(self):
        """Higher vol-of-vol → wider smile."""
        vol_otm_low_nu = sabr_implied_vol(F, F * 1.5, T, ALPHA, BETA, 0.0, 0.2)
        vol_otm_high_nu = sabr_implied_vol(F, F * 1.5, T, ALPHA, BETA, 0.0, 0.8)
        vol_atm_low = sabr_implied_vol(F, F, T, ALPHA, BETA, 0.0, 0.2)
        vol_atm_high = sabr_implied_vol(F, F, T, ALPHA, BETA, 0.0, 0.8)
        # Smile width = OTM vol - ATM vol
        smile_low = vol_otm_low_nu - vol_atm_low
        smile_high = vol_otm_high_nu - vol_atm_high
        assert smile_high > smile_low

    def test_beta_one_lognormal(self):
        """Beta=1: SABR with nu=0 → ATM vol ≈ alpha."""
        vol = sabr_implied_vol(F, F, T, ALPHA, 1.0, 0.0, 0.001)
        assert vol == pytest.approx(ALPHA, rel=0.05)

    def test_zero_T(self):
        vol = sabr_implied_vol(F, F * 1.1, 0.0, ALPHA, BETA, RHO, NU)
        assert vol == ALPHA


class TestSABRPrice:
    def test_call_positive(self):
        p = sabr_price(F, F, T, DF, ALPHA, BETA, RHO, NU, OptionType.CALL)
        assert p > 0

    def test_put_call_parity(self):
        c = sabr_price(F, F * 1.1, T, DF, ALPHA, BETA, RHO, NU, OptionType.CALL)
        p = sabr_price(F, F * 1.1, T, DF, ALPHA, BETA, RHO, NU, OptionType.PUT)
        K = F * 1.1
        expected = DF * (F - K)
        assert c - p == pytest.approx(expected, abs=1e-10)


class TestSABRCalibration:
    def test_recovers_known_params(self):
        """Generate smile from known params, calibrate back."""
        strikes = [F * m for m in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]]
        market_vols = [sabr_implied_vol(F, k, T, ALPHA, BETA, RHO, NU) for k in strikes]

        result = sabr_calibrate(F, strikes, market_vols, T, beta=BETA)

        assert result["alpha"] == pytest.approx(ALPHA, rel=0.05)
        assert result["rho"] == pytest.approx(RHO, abs=0.05)
        assert result["nu"] == pytest.approx(NU, rel=0.1)
        assert result["rmse"] < 0.001

    def test_reprices_market_vols(self):
        """Calibrated model reprices input vols."""
        strikes = [F * m for m in [0.8, 0.9, 1.0, 1.1, 1.2]]
        market_vols = [sabr_implied_vol(F, k, T, ALPHA, BETA, RHO, NU) for k in strikes]

        result = sabr_calibrate(F, strikes, market_vols, T, beta=BETA)

        for k, mv in zip(strikes, market_vols):
            model_vol = sabr_implied_vol(
                F, k, T, result["alpha"], BETA, result["rho"], result["nu"],
            )
            assert model_vol == pytest.approx(mv, abs=0.001)

    def test_returns_all_keys(self):
        strikes = [F * 0.9, F, F * 1.1]
        vols = [0.30, 0.25, 0.28]
        result = sabr_calibrate(F, strikes, vols, T, beta=BETA)
        assert "alpha" in result
        assert "beta" in result
        assert "rho" in result
        assert "nu" in result
        assert "rmse" in result
