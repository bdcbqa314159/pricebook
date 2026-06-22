"""Tests for `CalibrationResult` integration on LMM + SABR calibration (G1 P1 Slice 4).

LMM follows the dataclass pattern (calibration_result field + to_calibration_result()).
SABR returns a dict; the new `calibration_result` is added under that key.
"""

import math

import pytest

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.models.lmm_calibration import (
    LMMCalibrationResult,
    calibrate_lmm_vols,
)
from pricebook.options.sabr import sabr_calibrate, sabr_implied_vol


# ============================================================
# LMM
# ============================================================

class TestLMMCalibrationResult:
    def _calibrate(self):
        target = {(0, 5): 0.20, (4, 5): 0.18, (8, 5): 0.16}
        forwards = [0.05] * 12
        return calibrate_lmm_vols(forwards, target, tau=0.25, max_iter=20)

    def test_calibration_result_populated(self):
        r = self._calibrate()
        assert r.calibration_result is not None
        assert isinstance(r.calibration_result, CalibrationResult)

    def test_model_class_and_optimiser(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert cr.fit.model_class == "lmm"
        assert cr.optimiser_run.spec.algorithm == "iterative_scaling"
        assert cr.optimiser_run.spec.max_iterations == 20

    def test_parameters_are_per_forward_sigmas(self):
        r = self._calibrate()
        cr = r.calibration_result
        # Twelve forwards → twelve sigma parameters
        assert len(cr.fit.parameters) == 12
        # Keys are sigma_0 .. sigma_11
        assert set(cr.fit.parameters.keys()) == {f"sigma_{i}" for i in range(12)}
        # Values match the dataclass field
        assert list(cr.fit.parameters.values()) == r.calibrated_vols

    def test_residuals_in_vol_units(self):
        r = self._calibrate()
        cr = r.calibration_result
        # Three swaption targets → three residuals
        assert len(cr.fit.residuals) == 3
        # Each residual = fitted - target, in vol units (not bp)
        for residual, k in zip(cr.fit.residuals, sorted(r.target_swaption_vols)):
            expected = r.fitted_swaption_vols[k] - r.target_swaption_vols[k]
            assert residual == pytest.approx(expected, rel=1e-9)

    def test_quotes_fitted(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert list(cr.fit.quotes_fitted) == [
            "swaption_0x5",
            "swaption_4x5",
            "swaption_8x5",
        ]

    def test_diagnostics_rmse_vol(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert cr.diagnostics.extra["rmse_vol"] == pytest.approx(r.rmse)

    def test_to_calibration_result_returns_stored(self):
        r = self._calibrate()
        assert r.to_calibration_result() is r.calibration_result

    def test_unique_id_per_run(self):
        r1 = self._calibrate()
        r2 = self._calibrate()
        assert r1.calibration_result.provenance.id != r2.calibration_result.provenance.id

    def test_to_dict_has_calibration_id(self):
        r = self._calibrate()
        d = r.to_dict()
        assert d["calibration_id"] == str(r.calibration_result.provenance.id)


class TestLMMBackCompat:
    def _hand_built(self) -> LMMCalibrationResult:
        return LMMCalibrationResult(
            calibrated_vols=[0.20, 0.19],
            target_swaption_vols={(0, 5): 0.18, (4, 5): 0.20},
            fitted_swaption_vols={(0, 5): 0.181, (4, 5): 0.198},
            rmse=0.001,
        )

    def test_hand_construction_without_calibration_result(self):
        r = self._hand_built()
        assert r.calibration_result is None

    def test_on_demand_to_calibration_result(self):
        r = self._hand_built()
        cr = r.to_calibration_result()
        assert isinstance(cr, CalibrationResult)
        assert cr.fit.model_class == "lmm"
        assert cr.fit.parameters == {"sigma_0": 0.20, "sigma_1": 0.19}
        assert len(cr.fit.residuals) == 2

    def test_to_dict_has_none_calibration_id(self):
        assert self._hand_built().to_dict()["calibration_id"] is None


# ============================================================
# SABR
# ============================================================

class TestSABRCalibrationResult:
    """SABR returns a dict; `calibration_result` is added under that key."""

    def _calibrate(self):
        return sabr_calibrate(
            forward=0.05,
            strikes=[0.03, 0.04, 0.05, 0.06, 0.07],
            market_vols=[0.30, 0.25, 0.22, 0.21, 0.22],
            T=5.0,
            beta=0.5,
        )

    def test_calibration_result_in_dict(self):
        res = self._calibrate()
        assert "calibration_result" in res
        assert isinstance(res["calibration_result"], CalibrationResult)

    def test_existing_keys_preserved(self):
        """Backward compat: alpha, beta, rho, nu, rmse keys still present."""
        res = self._calibrate()
        for key in ("alpha", "beta", "rho", "nu", "rmse"):
            assert key in res

    def test_parameters_match_dict_values(self):
        res = self._calibrate()
        cr = res["calibration_result"]
        assert cr.fit.parameters["alpha"] == pytest.approx(res["alpha"])
        assert cr.fit.parameters["beta"] == pytest.approx(res["beta"])
        assert cr.fit.parameters["rho"] == pytest.approx(res["rho"])
        assert cr.fit.parameters["nu"] == pytest.approx(res["nu"])

    def test_optimiser_recorded(self):
        cr = self._calibrate()["calibration_result"]
        assert cr.optimiser_run.spec.algorithm == "nelder_mead"
        assert cr.optimiser_run.spec.tolerance == 1e-12
        assert cr.optimiser_run.spec.max_iterations == 2000
        assert cr.optimiser_run.spec.extra["beta_fixed"] == 0.5
        assert cr.optimiser_run.spec.extra["forward"] == 0.05
        assert cr.optimiser_run.spec.extra["T"] == 5.0

    def test_quotes_fitted_named_by_strike(self):
        cr = self._calibrate()["calibration_result"]
        assert list(cr.fit.quotes_fitted) == [
            "smile_K=0.0300",
            "smile_K=0.0400",
            "smile_K=0.0500",
            "smile_K=0.0600",
            "smile_K=0.0700",
        ]

    def test_residuals_match_model_minus_market(self):
        res = self._calibrate()
        cr = res["calibration_result"]
        strikes = [0.03, 0.04, 0.05, 0.06, 0.07]
        market_vols = [0.30, 0.25, 0.22, 0.21, 0.22]
        for i, (k, mv) in enumerate(zip(strikes, market_vols)):
            model = sabr_implied_vol(0.05, k, 5.0, res["alpha"], res["beta"],
                                     res["rho"], res["nu"])
            assert cr.fit.residuals[i] == pytest.approx(model - mv, abs=1e-9)

    def test_unique_id_per_run(self):
        r1 = self._calibrate()
        r2 = self._calibrate()
        assert r1["calibration_result"].provenance.id != r2["calibration_result"].provenance.id

    def test_diagnostics_rmse_vol(self):
        res = self._calibrate()
        cr = res["calibration_result"]
        assert cr.diagnostics.extra["rmse_vol"] == pytest.approx(res["rmse"])

    def test_model_class(self):
        cr = self._calibrate()["calibration_result"]
        assert cr.fit.model_class == "sabr"
