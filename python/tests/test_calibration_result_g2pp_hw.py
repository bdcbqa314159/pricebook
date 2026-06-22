"""Tests for `CalibrationResult` integration on HW and G2++ calibration (G1 P1 Slice 3).

The HW tests run the full `calibrate_hull_white` on a small swaption grid
(fast — a few seconds). The G2++ tests verify the on-demand
`to_calibration_result()` path and the dataclass shape; the full
`calibrate_g2pp` integration is covered by the existing slow tests
in `test_g2pp_calibration.py` (which are deselected in CI runs).
"""

from datetime import date

import pytest

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.models.hw_calibration import (
    HWCalibrationResult,
    calibrate_hull_white,
)
from pricebook.models.g2pp_calibration import G2PPCalibrationResult
from pricebook.models.hull_white import HullWhite
from pricebook.models.vasicek import G2PlusPlus


REF = date(2026, 6, 11)


def _flat_rf():
    return DiscountCurve.flat(REF, 0.04)


def _small_vol_grid():
    return {
        (1, 5): 0.0065,
        (5, 5): 0.0055,
        (10, 10): 0.0045,
    }


# ============================================================
# Hull-White calibration — full end-to-end
# ============================================================

class TestHWCalibrationResult:
    def test_calibration_result_populated(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        assert result.calibration_result is not None
        assert isinstance(result.calibration_result, CalibrationResult)

    def test_model_class_and_parameters(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        cr = result.calibration_result
        assert cr.fit.model_class == "hull_white"
        assert set(cr.fit.parameters.keys()) == {"a", "sigma"}
        assert cr.fit.parameters["a"] == pytest.approx(result.a)
        assert cr.fit.parameters["sigma"] == pytest.approx(result.sigma)

    def test_optimiser_recorded(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        cr = result.calibration_result
        assert cr.optimiser_run.spec.algorithm == "Nelder-Mead"
        assert cr.optimiser_run.spec.extra.get("n_steps") == 50

    def test_residuals_in_vol_bp(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        cr = result.calibration_result
        # Residuals match the per_swaption_errors error_bp field
        expected = [e["error_bp"] for e in result.per_swaption_errors]
        assert list(cr.fit.residuals) == expected

    def test_diagnostics_includes_rmse_vol(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        cr = result.calibration_result
        assert "rmse_vol" in cr.diagnostics.extra
        assert cr.diagnostics.extra["rmse_vol"] == pytest.approx(result.rmse_vol)

    def test_quotes_fitted_named(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        cr = result.calibration_result
        assert list(cr.fit.quotes_fitted) == [
            "swaption_1x5",
            "swaption_5x5",
            "swaption_10x10",
        ]

    def test_to_calibration_result_returns_stored(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        assert result.to_calibration_result() is result.calibration_result

    def test_unique_id_per_run(self):
        rf = _flat_rf()
        r1 = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        r2 = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        assert r1.calibration_result.provenance.id != r2.calibration_result.provenance.id

    def test_to_dict_has_calibration_id(self):
        rf = _flat_rf()
        result = calibrate_hull_white(rf, _small_vol_grid(), method="nelder_mead")
        d = result.to_dict()
        assert d["calibration_id"] == str(result.calibration_result.provenance.id)


class TestHWBackCompat:
    def test_hand_construction_without_calibration_result(self):
        """Existing callers that hand-build HWCalibrationResult must still work."""
        rf = _flat_rf()
        hw = HullWhite(a=0.05, sigma=0.01, curve=rf)
        r = HWCalibrationResult(
            model=hw, a=0.05, sigma=0.01,
            rmse_vol=0.001,
            per_swaption_errors=[{"expiry": 1, "tenor": 5, "market_vol": 0.0065,
                                  "model_vol": 0.0066, "error_bp": 1.0}],
            n_swaptions=1,
            converged=True,
        )
        assert r.calibration_result is None

    def test_on_demand_to_calibration_result(self):
        rf = _flat_rf()
        hw = HullWhite(a=0.05, sigma=0.01, curve=rf)
        r = HWCalibrationResult(
            model=hw, a=0.05, sigma=0.01,
            rmse_vol=0.001,
            per_swaption_errors=[
                {"expiry": 1, "tenor": 5, "market_vol": 0.0065,
                 "model_vol": 0.0066, "error_bp": 1.0},
                {"expiry": 5, "tenor": 5, "market_vol": 0.0055,
                 "model_vol": 0.0053, "error_bp": -2.0},
            ],
            n_swaptions=2,
            converged=True,
        )
        cr = r.to_calibration_result()
        assert isinstance(cr, CalibrationResult)
        assert cr.fit.model_class == "hull_white"
        assert cr.fit.parameters == {"a": 0.05, "sigma": 0.01}
        assert list(cr.fit.residuals) == [1.0, -2.0]
        assert list(cr.fit.quotes_fitted) == ["swaption_1x5", "swaption_5x5"]
        assert cr.optimiser_run.converged is True


# ============================================================
# G2++ calibration — dataclass shape + on-demand path
# (full end-to-end coverage lives in test_g2pp_calibration.py, slow tests)
# ============================================================

class TestG2PPBackCompat:
    def _make_hand_result(self) -> G2PPCalibrationResult:
        rf = _flat_rf()
        g2 = G2PlusPlus(a=0.05, b=0.10, sigma1=0.01, sigma2=0.008, rho=-0.5, curve=rf)
        return G2PPCalibrationResult(
            model=g2, a=0.05, b=0.10, sigma1=0.01, sigma2=0.008, rho=-0.5,
            rmse_vol=0.0008,
            per_swaption_errors=[
                {"expiry": 1, "tenor": 5, "market_vol": 0.0065,
                 "model_vol": 0.0064, "error_bp": -1.0},
                {"expiry": 5, "tenor": 10, "market_vol": 0.0050,
                 "model_vol": 0.0053, "error_bp": 3.0},
            ],
            n_swaptions=2,
            converged=True,
        )

    def test_hand_construction_without_calibration_result(self):
        r = self._make_hand_result()
        assert r.calibration_result is None

    def test_on_demand_to_calibration_result(self):
        r = self._make_hand_result()
        cr = r.to_calibration_result()
        assert isinstance(cr, CalibrationResult)
        assert cr.fit.model_class == "g2pp"
        assert cr.fit.parameters == {
            "a": 0.05, "b": 0.10,
            "sigma1": 0.01, "sigma2": 0.008, "rho": -0.5,
        }
        assert list(cr.fit.residuals) == [-1.0, 3.0]
        assert list(cr.fit.quotes_fitted) == ["swaption_1x5", "swaption_5x10"]
        assert cr.optimiser_run.converged is True

    def test_to_dict_has_calibration_id_none_when_unpopulated(self):
        r = self._make_hand_result()
        d = r.to_dict()
        assert d["calibration_id"] is None

    def test_to_dict_has_calibration_id_when_populated(self):
        r = self._make_hand_result()
        # Manually attach (simulating what calibrate_g2pp would do)
        cr = r.to_calibration_result()
        r2 = G2PPCalibrationResult(
            model=r.model, a=r.a, b=r.b, sigma1=r.sigma1, sigma2=r.sigma2, rho=r.rho,
            rmse_vol=r.rmse_vol, per_swaption_errors=r.per_swaption_errors,
            n_swaptions=r.n_swaptions, converged=r.converged,
            calibration_result=cr,
        )
        d = r2.to_dict()
        assert d["calibration_id"] == str(cr.provenance.id)

    def test_g2pp_parameters_are_five(self):
        """Sanity guard — G2++ has exactly 5 calibrated parameters."""
        r = self._make_hand_result()
        cr = r.to_calibration_result()
        assert len(cr.fit.parameters) == 5
