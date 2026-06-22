"""Tests for `CalibrationResult` integration on jump calibration (G1 P1 Slice 6).

Closes out G1 P1: all seven calibration families now produce the canonical
artefact. Tests use hand-constructed `JumpCalibrationResult` for back-compat
+ a single fast `calibrate_jump_model("merton", ...)` smoke for end-to-end.
"""

import math

import pytest

from pricebook.calibration import CalibrationResult, ObjectiveKind
from pricebook.models.jump_calibration import (
    JumpCalibrationResult,
    calibrate_jump_model,
)


# ============================================================
# End-to-end Merton calibration
# ============================================================

class TestJumpCalibrationResult:
    def _calibrate(self):
        # Small problem to keep test fast — 3 strikes, low maxiter
        return calibrate_jump_model(
            model_type="merton",
            strikes=[80.0, 100.0, 120.0],
            market_vols=[0.25, 0.22, 0.24],
            spot=100.0,
            rate=0.04,
            T=1.0,
            maxiter=30,
        )

    def test_calibration_result_populated(self):
        r = self._calibrate()
        assert r.calibration_result is not None
        assert isinstance(r.calibration_result, CalibrationResult)

    def test_model_class_includes_jump_prefix(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert cr.fit.model_class == "jump_merton"

    def test_parameters_match_params_dict(self):
        r = self._calibrate()
        cr = r.calibration_result
        # Merton has 4 parameters: sigma, lam, mu_j, sigma_j
        assert set(cr.fit.parameters.keys()) == {"sigma", "lam", "mu_j", "sigma_j"}
        for k, v in r.params.items():
            assert cr.fit.parameters[k] == pytest.approx(v)

    def test_residuals_match_model_minus_market(self):
        r = self._calibrate()
        cr = r.calibration_result
        for i, (mv, mkv) in enumerate(zip(r.model_vols, r.market_vols)):
            assert cr.fit.residuals[i] == pytest.approx(mv - mkv)

    def test_optimiser_recorded(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert "differential_evolution" in cr.optimiser_run.spec.algorithm
        assert cr.optimiser_run.spec.seed == 42
        assert cr.optimiser_run.spec.max_iterations == 30
        # Setup recorded
        assert cr.optimiser_run.spec.extra["spot"] == 100.0
        assert cr.optimiser_run.spec.extra["T"] == 1.0
        assert cr.optimiser_run.spec.extra["polish"] is True

    def test_quotes_fitted_named_by_strike(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert list(cr.fit.quotes_fitted) == [
            "smile_K=80.0000",
            "smile_K=100.0000",
            "smile_K=120.0000",
        ]

    def test_diagnostics_rmse_vol(self):
        r = self._calibrate()
        cr = r.calibration_result
        assert cr.diagnostics.extra["rmse_vol"] == pytest.approx(r.rmse_vol)

    def test_to_calibration_result_returns_stored(self):
        r = self._calibrate()
        assert r.to_calibration_result() is r.calibration_result

    def test_to_dict_has_calibration_id(self):
        r = self._calibrate()
        d = r.to_dict()
        assert d["calibration_id"] == str(r.calibration_result.provenance.id)


# ============================================================
# Back-compat
# ============================================================

class TestJumpBackCompat:
    def _hand_build(self) -> JumpCalibrationResult:
        return JumpCalibrationResult(
            model_type="vg",
            params={"sigma": 0.20, "nu": 0.5, "theta": -0.1},
            rmse_vol=0.001,
            market_vols=[0.25, 0.22],
            model_vols=[0.251, 0.219],
            strikes=[80.0, 100.0],
            n_params=3,
        )

    def test_hand_construction_without_calibration_result(self):
        r = self._hand_build()
        assert r.calibration_result is None

    def test_on_demand_to_calibration_result(self):
        r = self._hand_build()
        cr = r.to_calibration_result()
        assert isinstance(cr, CalibrationResult)
        assert cr.fit.model_class == "jump_vg"
        assert cr.fit.parameters == {"sigma": 0.20, "nu": 0.5, "theta": -0.1}
        # Residuals are model - market in order
        assert list(cr.fit.residuals) == [
            pytest.approx(0.251 - 0.25),
            pytest.approx(0.219 - 0.22),
        ]
        assert list(cr.fit.quotes_fitted) == ["smile_K=80.0000", "smile_K=100.0000"]

    def test_to_dict_has_none_when_unpopulated(self):
        r = self._hand_build()
        d = r.to_dict()
        assert d["calibration_id"] is None
