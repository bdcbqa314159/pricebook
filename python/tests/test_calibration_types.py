"""Tests for the new `pricebook.calibration` types (G1 P1 slice 1)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from datetime import datetime
from uuid import UUID

import pytest

import pricebook
from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    Calibrator,
    ObjectiveKind,
    OptimiserSpec,
)


class TestObjectiveKind:
    def test_string_values_match_names(self):
        # Values are lower-case strings of the canonical name.
        assert ObjectiveKind.SSE.value == "sse"
        assert ObjectiveKind.WEIGHTED_SSE.value == "weighted_sse"
        assert ObjectiveKind.MAX_ERROR.value == "max_error"
        assert ObjectiveKind.L1.value == "l1"
        assert ObjectiveKind.HUBER.value == "huber"
        assert ObjectiveKind.RMSE.value == "rmse"

    def test_string_enum_round_trip(self):
        # str-enum means we can persist and reload via the value string.
        kind = ObjectiveKind.WEIGHTED_SSE
        assert ObjectiveKind(kind.value) is kind


class TestOptimiserSpec:
    def test_construction(self):
        spec = OptimiserSpec(
            algorithm="L-BFGS-B",
            tolerance=1e-9,
            max_iterations=500,
        )
        assert spec.algorithm == "L-BFGS-B"
        assert spec.tolerance == 1e-9
        assert spec.max_iterations == 500
        assert spec.seed is None
        assert spec.extra == {}

    def test_seed_for_stochastic(self):
        spec = OptimiserSpec(
            algorithm="differential_evolution",
            tolerance=1e-6,
            max_iterations=200,
            seed=42,
        )
        assert spec.seed == 42

    def test_frozen(self):
        spec = OptimiserSpec(algorithm="x", tolerance=1.0, max_iterations=1)
        with pytest.raises(FrozenInstanceError):
            spec.tolerance = 2.0  # type: ignore[misc]


class TestCalibrationDiagnostics:
    def test_defaults_are_empty(self):
        d = CalibrationDiagnostics()
        assert d.objective_history == ()
        assert d.parameter_history == ()
        assert d.timing_ms is None
        assert d.warnings == ()
        assert d.extra == {}

    def test_population(self):
        d = CalibrationDiagnostics(
            objective_history=(1.0, 0.5, 0.1),
            parameter_history=({"a": 0.1}, {"a": 0.2}),
            timing_ms=42.0,
            warnings=("non-monotone iterate at step 17",),
        )
        assert len(d.objective_history) == 3
        assert d.timing_ms == 42.0
        assert d.warnings == ("non-monotone iterate at step 17",)

    def test_frozen(self):
        d = CalibrationDiagnostics()
        with pytest.raises(FrozenInstanceError):
            d.timing_ms = 1.0  # type: ignore[misc]


class TestCalibrationResultNewFactory:
    """`CalibrationResult.new(...)` is the normal construction path."""

    def _spec(self) -> OptimiserSpec:
        return OptimiserSpec(algorithm="L-BFGS-B", tolerance=1e-9, max_iterations=500)

    def test_basic_construction(self):
        r = CalibrationResult.new(
            model_class="hazard_pwc",
            parameters={"h_1y": 0.02, "h_5y": 0.03},
            residuals=[0.5, -0.3, 0.1],
            optimiser=self._spec(),
            iterations=42,
            converged=True,
        )
        assert isinstance(r.id, UUID)
        assert isinstance(r.timestamp, datetime)
        assert r.model_class == "hazard_pwc"
        assert r.parameters == {"h_1y": 0.02, "h_5y": 0.03}
        assert r.iterations == 42
        assert r.converged is True
        assert r.objective is ObjectiveKind.SSE  # default

    def test_rms_and_max_residual_computed(self):
        residuals = [3.0, -4.0, 0.0]   # RMSE = sqrt((9 + 16 + 0) / 3) = sqrt(25/3)
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=residuals,
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        assert r.rms_residual == pytest.approx(math.sqrt(25.0 / 3.0))
        assert r.max_residual == 4.0

    def test_empty_residuals_means_zero_rms_and_max(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[],
            optimiser=self._spec(),
            iterations=0,
            converged=False,
        )
        assert r.rms_residual == 0.0
        assert r.max_residual == 0.0

    def test_unique_id_per_call(self):
        kwargs = dict(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        r1 = CalibrationResult.new(**kwargs)
        r2 = CalibrationResult.new(**kwargs)
        assert r1.id != r2.id

    def test_default_weights_are_unity_per_residual(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[1.0, 2.0, 3.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        assert r.weights == [1.0, 1.0, 1.0]

    def test_explicit_weights_preserved(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[1.0, 2.0, 3.0],
            weights=[0.5, 1.0, 2.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        assert r.weights == [0.5, 1.0, 2.0]

    def test_default_diagnostics_is_empty(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        assert r.diagnostics.objective_history == ()
        assert r.diagnostics.timing_ms is None

    def test_explicit_diagnostics_preserved(self):
        diag = CalibrationDiagnostics(
            objective_history=(1.0, 0.5, 0.25),
            timing_ms=12.5,
        )
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=self._spec(),
            iterations=3,
            converged=True,
            diagnostics=diag,
        )
        assert r.diagnostics is diag

    def test_code_version_picks_up_pricebook_version_by_default(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        assert r.code_version == pricebook.__version__

    def test_code_version_explicit_override(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
            code_version="custom-1.2.3",
        )
        assert r.code_version == "custom-1.2.3"

    def test_missing_pricebook_version_propagates(self, monkeypatch):
        # T-CAL1 regression: pre-fix `_detect_code_version` swallowed every
        # exception and returned the string "unknown", hiding real packaging
        # / import bugs. Removing __version__ must now surface AttributeError.
        monkeypatch.delattr(pricebook, "__version__")
        with pytest.raises(AttributeError):
            CalibrationResult.new(
                model_class="x",
                parameters={"a": 1.0},
                residuals=[0.0],
                optimiser=self._spec(),
                iterations=1,
                converged=True,
            )

    def test_market_snapshot_id_optional(self):
        r = CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=self._spec(),
            iterations=1,
            converged=True,
        )
        # Until G1 P2 lands MarketSnapshot, this is None.
        assert r.market_snapshot_id is None


class TestCalibrationResultIsFrozen:
    def _make(self) -> CalibrationResult:
        return CalibrationResult.new(
            model_class="x",
            parameters={"a": 1.0},
            residuals=[0.0],
            optimiser=OptimiserSpec(algorithm="x", tolerance=1.0, max_iterations=1),
            iterations=1,
            converged=True,
        )

    def test_cannot_mutate_top_level(self):
        r = self._make()
        with pytest.raises(FrozenInstanceError):
            r.rms_residual = 99.0  # type: ignore[misc]

    def test_cannot_mutate_optimiser(self):
        r = self._make()
        with pytest.raises(FrozenInstanceError):
            r.optimiser.tolerance = 99.0  # type: ignore[misc]


class TestCalibratorProtocol:
    """Anything with a `calibrate(...) -> CalibrationResult` is a Calibrator."""

    def test_class_with_calibrate_method_satisfies_protocol(self):
        class Fake:
            def calibrate(self, *args, **kwargs) -> CalibrationResult:
                return CalibrationResult.new(
                    model_class="fake",
                    parameters={"a": 1.0},
                    residuals=[0.0],
                    optimiser=OptimiserSpec(algorithm="x", tolerance=1.0, max_iterations=1),
                    iterations=1,
                    converged=True,
                )

        f: Calibrator = Fake()
        # Static check via assignment + runtime check via call
        result = f.calibrate()
        assert isinstance(result, CalibrationResult)
