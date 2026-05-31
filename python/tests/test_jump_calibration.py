"""Tests for jump model calibration to implied vol surfaces."""

import pytest
import math
import numpy as np

from pricebook.models.jump_calibration import (
    calibrate_jump_model, calibrate_jump_surface, jump_model_comparison,
)


def _synthetic_merton_surface(spot, rate, T, sigma=0.20, lam=1.0, mu_j=-0.10, sigma_j=0.15):
    """Generate synthetic implied vols from a known Merton model."""
    from pricebook.models.char_func_protocol import merton_char_func
    from pricebook.models.cos_method import cos_price
    from pricebook.models.black76 import OptionType
    from pricebook.options.implied_vol import implied_vol_black76

    strikes = [spot * m for m in [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]]
    phi = merton_char_func(rate, sigma, lam, mu_j, sigma_j, T)
    fwd = spot * math.exp(rate * T)
    df = math.exp(-rate * T)

    vols = []
    for k in strikes:
        price = cos_price(phi, spot, k, rate, T, OptionType.CALL, N=256)
        iv = implied_vol_black76(price, fwd, k, T, df, OptionType.CALL)
        vols.append(iv)

    return strikes, vols


class TestCalibrateSingleExpiry:
    def test_merton_roundtrip(self):
        """Calibrate Merton to its own surface — should recover params."""
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)

        result = calibrate_jump_model("merton", strikes, vols, spot, rate, T,
                                       maxiter=100, seed=42)

        assert result.rmse_vol < 0.005  # < 0.5 vol point
        assert result.model_type == "merton"
        assert "sigma" in result.params

    def test_vg_on_merton_surface(self):
        """VG should fit Merton surface reasonably (both have jumps)."""
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)

        result = calibrate_jump_model("vg", strikes, vols, spot, rate, T,
                                       maxiter=100, seed=42)
        assert result.rmse_vol < 0.02  # < 2 vol points (approx fit)

    def test_nig_calibration(self):
        """NIG should calibrate to Merton surface."""
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)

        result = calibrate_jump_model("nig", strikes, vols, spot, rate, T,
                                       maxiter=100, seed=42)
        assert result.rmse_vol < 0.02

    def test_kou_calibration(self):
        """Kou should calibrate to Merton surface."""
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)

        result = calibrate_jump_model("kou", strikes, vols, spot, rate, T,
                                       maxiter=100, seed=42)
        assert result.rmse_vol < 0.02

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            calibrate_jump_model("unknown", [100], [0.2], 100, 0.05, 1.0)

    def test_result_has_model_vols(self):
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)
        result = calibrate_jump_model("merton", strikes, vols, spot, rate, T,
                                       maxiter=50, seed=42)
        assert len(result.model_vols) == len(strikes)
        assert all(v > 0 for v in result.model_vols)

    def test_to_dict(self):
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)
        result = calibrate_jump_model("vg", strikes, vols, spot, rate, T,
                                       maxiter=50, seed=42)
        d = result.to_dict()
        assert "rmse_vol" in d


class TestMultiExpiry:
    def test_surface_calibration(self):
        spot, rate = 100, 0.05
        data = []
        for T in [0.25, 0.5, 1.0]:
            strikes, vols = _synthetic_merton_surface(spot, rate, T)
            data.append({"T": T, "strikes": strikes, "vols": vols})

        results = calibrate_jump_surface("merton", data, spot, rate, maxiter=50, seed=42)
        assert len(results) == 3
        assert all(r.rmse_vol < 0.01 for r in results)


class TestModelComparison:
    def test_comparison_ranks(self):
        """Model comparison should produce a ranking."""
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)

        result = jump_model_comparison(
            strikes, vols, spot, rate, T,
            models=["merton", "vg", "nig"],
            maxiter=50, seed=42,
        )

        assert len(result.results) == 3
        assert len(result.ranking) == 3
        # All models should fit reasonably
        for r in result.results:
            assert r.rmse_vol < 0.02

    def test_aic_values(self):
        spot, rate, T = 100, 0.05, 1.0
        strikes, vols = _synthetic_merton_surface(spot, rate, T)
        result = jump_model_comparison(
            strikes, vols, spot, rate, T,
            models=["merton", "vg"],
            maxiter=50, seed=42,
        )
        assert "merton" in result.aic_values
        assert "vg" in result.aic_values
        # Both should have finite AIC
        assert math.isfinite(result.aic_values["merton"])
        assert math.isfinite(result.aic_values["vg"])
