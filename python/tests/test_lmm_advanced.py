"""Tests for LMM deepening: swaption calibration, SABR-LMM, predictor-corrector, Greeks."""

import math

import numpy as np
import pytest

from pricebook.lmm_advanced import (
    SABRLMM,
    LMMCalibrationResult,
    LMMGreeksResult,
    PredictorCorrectorResult,
    SABRLMMResult,
    lmm_cascade_calibration,
    lmm_global_calibration,
    lmm_pathwise_greeks,
    lmm_predictor_corrector,
)


# ---- Cascade calibration ----

class TestCascadeCalibration:
    def test_basic_calibration(self):
        fwd = [0.05, 0.05, 0.05, 0.05]
        market_vols = {(0, 1): 0.20, (1, 1): 0.22, (2, 1): 0.18, (3, 1): 0.21}
        result = lmm_cascade_calibration(market_vols, fwd)
        assert isinstance(result, LMMCalibrationResult)
        assert result.method == "cascade"
        assert result.n_swaptions == 4
        assert len(result.vols) == 4

    def test_positive_vols(self):
        fwd = [0.04, 0.045, 0.05]
        market_vols = {(0, 1): 0.20, (1, 1): 0.22}
        result = lmm_cascade_calibration(market_vols, fwd)
        assert np.all(result.vols > 0)

    def test_residual_small(self):
        fwd = [0.05] * 4
        market_vols = {(0, 1): 0.20, (1, 1): 0.20, (2, 1): 0.20}
        result = lmm_cascade_calibration(market_vols, fwd)
        assert result.residual < 0.05

    def test_numpy_forward_rates(self):
        fwd = np.array([0.05, 0.05, 0.05])
        market_vols = {(0, 1): 0.20, (1, 1): 0.22}
        result = lmm_cascade_calibration(market_vols, fwd)
        assert result.n_swaptions == 2


# ---- Global calibration ----

class TestGlobalCalibration:
    def test_basic_calibration(self):
        fwd = [0.05, 0.05, 0.05, 0.05]
        market_vols = {(0, 1): 0.20, (1, 1): 0.22, (2, 1): 0.18, (3, 1): 0.21}
        result = lmm_global_calibration(market_vols, fwd)
        assert isinstance(result, LMMCalibrationResult)
        assert result.method == "global"
        assert result.n_swaptions == 4

    def test_vols_in_bounds(self):
        fwd = [0.04, 0.045, 0.05, 0.055]
        market_vols = {(0, 1): 0.20, (1, 1): 0.25, (2, 1): 0.18}
        result = lmm_global_calibration(market_vols, fwd)
        assert np.all(result.vols >= 0.01)
        assert np.all(result.vols <= 1.0)

    def test_residual_reasonable(self):
        fwd = [0.05] * 4
        market_vols = {(0, 1): 0.20, (1, 1): 0.20, (2, 1): 0.20}
        result = lmm_global_calibration(market_vols, fwd)
        assert result.residual < 0.10

    def test_global_vs_cascade(self):
        """Both methods should produce reasonable results."""
        fwd = [0.05] * 5
        market_vols = {(i, 1): 0.20 for i in range(4)}
        cascade = lmm_cascade_calibration(market_vols, fwd)
        glob = lmm_global_calibration(market_vols, fwd)
        assert cascade.residual < 0.10
        assert glob.residual < 0.10


# ---- SABR-LMM ----

class TestSABRLMM:
    def test_basic_simulation(self):
        model = SABRLMM([0.05, 0.05, 0.05])
        result = model.simulate(1.0, n_steps=50, n_paths=1_000, seed=42)
        assert isinstance(result, SABRLMMResult)
        assert result.forward_paths.shape == (1_000, 51, 3)
        assert result.vol_paths.shape == (1_000, 51, 3)

    def test_forward_paths_positive(self):
        model = SABRLMM([0.05, 0.04, 0.06])
        result = model.simulate(1.0, n_steps=50, n_paths=2_000, seed=42)
        assert np.all(result.forward_paths >= 0)

    def test_vol_paths_positive(self):
        model = SABRLMM([0.05, 0.05])
        result = model.simulate(1.0, n_steps=50, n_paths=1_000, seed=42)
        assert np.all(result.vol_paths > 0)

    def test_swaption_price_positive(self):
        model = SABRLMM([0.05, 0.05, 0.05])
        result = model.simulate(1.0, n_steps=50, n_paths=5_000, seed=42)
        assert result.swaption_price > 0

    def test_custom_sabr_params(self):
        model = SABRLMM(
            [0.05, 0.05],
            betas=[0.5, 0.5],
            alphas=[0.4, 0.4],
            rhos=[-0.2, -0.2],
            init_vols=[0.25, 0.25],
        )
        result = model.simulate(1.0, n_steps=50, n_paths=1_000, seed=42)
        assert result.forward_paths.shape[2] == 2

    def test_initial_conditions(self):
        fwd = [0.04, 0.05, 0.06]
        vols = [0.15, 0.20, 0.25]
        model = SABRLMM(fwd, init_vols=vols)
        result = model.simulate(1.0, n_steps=50, n_paths=100, seed=42)
        np.testing.assert_allclose(result.forward_paths[:, 0, :], [fwd] * 100)
        np.testing.assert_allclose(result.vol_paths[:, 0, :], [vols] * 100)


# ---- Predictor-corrector ----

class TestPredictorCorrector:
    def test_basic(self):
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_predictor_corrector(fwd, vols, T=1.0, n_steps=50,
                                         n_paths=1_000, seed=42)
        assert isinstance(result, PredictorCorrectorResult)
        assert result.forward_paths.shape == (1_000, 51, 3)

    def test_positive_paths(self):
        fwd = [0.05, 0.04, 0.06]
        vols = [0.20, 0.22, 0.18]
        result = lmm_predictor_corrector(fwd, vols, T=1.0, n_steps=50,
                                         n_paths=2_000, seed=42)
        assert np.all(result.forward_paths >= 0)

    def test_caplet_price_positive(self):
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_predictor_corrector(fwd, vols, T=1.0, n_steps=50,
                                         n_paths=5_000, seed=42)
        assert result.caplet_price > 0

    def test_initial_condition(self):
        fwd = [0.04, 0.05, 0.06]
        vols = [0.20, 0.20, 0.20]
        result = lmm_predictor_corrector(fwd, vols, T=1.0, n_steps=50,
                                         n_paths=100, seed=42)
        np.testing.assert_allclose(result.forward_paths[:, 0, :], [fwd] * 100)

    def test_mean_near_forward(self):
        """Mean terminal rate should be near initial (low vol)."""
        fwd = [0.05, 0.05, 0.05]
        vols = [0.05, 0.05, 0.05]
        result = lmm_predictor_corrector(fwd, vols, T=1.0, n_steps=50,
                                         n_paths=10_000, seed=42)
        mean_terminal = result.forward_paths[:, -1, 0].mean()
        assert mean_terminal == pytest.approx(0.05, rel=0.15)


# ---- Pathwise Greeks ----

class TestPathwiseGreeks:
    def test_basic(self):
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_pathwise_greeks(fwd, vols, strike=0.05, expiry_idx=0,
                                     n_paths=5_000, n_steps=30, seed=42)
        assert isinstance(result, LMMGreeksResult)
        assert len(result.deltas) == 3

    def test_atm_delta_positive(self):
        """ATM caplet delta should be positive."""
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_pathwise_greeks(fwd, vols, strike=0.05, expiry_idx=0,
                                     n_paths=20_000, n_steps=30, seed=42)
        assert result.deltas[0] > 0

    def test_deep_otm_delta_small(self):
        """Deep OTM caplet: delta should be near zero."""
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_pathwise_greeks(fwd, vols, strike=0.15, expiry_idx=0,
                                     n_paths=20_000, n_steps=30, seed=42)
        assert abs(result.deltas[0]) < 0.05

    def test_only_relevant_forward_has_delta(self):
        """Only the forward at expiry_idx should have nonzero delta."""
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_pathwise_greeks(fwd, vols, strike=0.05, expiry_idx=1,
                                     n_paths=10_000, n_steps=30, seed=42)
        # Delta should be nonzero only for index 1
        assert result.deltas[1] != 0
        assert result.deltas[0] == 0
        assert result.deltas[2] == 0

    def test_total_delta_equals_sum(self):
        fwd = [0.05, 0.05, 0.05]
        vols = [0.20, 0.20, 0.20]
        result = lmm_pathwise_greeks(fwd, vols, strike=0.05, expiry_idx=0,
                                     n_paths=5_000, n_steps=30, seed=42)
        assert result.total_delta == pytest.approx(sum(result.deltas), rel=1e-10)
