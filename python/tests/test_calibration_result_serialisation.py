"""CalibrationResult serialisation + clock — G1 unification Phase 0 Slice 1.

Covers:
- round-trip to_dict/from_dict for OptimiserSpec, CalibrationDiagnostics,
  CalibrationResult (incl. nested optimiser/diagnostics and UUID/datetime fields);
- the serialised payload is JSON-native (json.dumps round-trips);
- `build_calibration_result()` stamps a timezone-aware UTC timestamp;
- `id`/`timestamp` are injectable (for reproducibility / determinism);
- the new core atom support (UUID, datetime, dict, tuple) round-trips.
"""

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.core.serialisable import _deserialise_atom, _serialise_atom
from tests.conftest import build_calibration_result


def _sample_result() -> CalibrationResult:
    return build_calibration_result(
        model_class="hull_white",
        parameters={"a": 0.03, "sigma": 0.012},
        residuals=[0.001, -0.002, 0.0005],
        optimiser=OptimiserSpec(
            algorithm="L-BFGS-B",
            tolerance=1e-10,
            max_iterations=500,
            seed=42,
            extra={"factr": 1e7},
        ),
        iterations=37,
        converged=True,
        objective=ObjectiveKind.WEIGHTED_SSE,
        quotes_fitted=["1Y", "2Y", "5Y"],
        weights=[1.0, 0.5, 0.25],
        diagnostics=CalibrationDiagnostics(
            objective_history=[10.0, 1.0, 0.01],
            parameter_history=[{"a": 0.02}, {"a": 0.03}],
            timing_ms=12.5,
            warnings=["near-bound a"],
            extra={"rmse_vol": 0.0008},
        ),
        market_snapshot_id=uuid4(),
        code_version="9.9.9",
    )


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #

def test_optimiser_spec_round_trip():
    spec = OptimiserSpec("least_squares", 1e-8, 100, seed=7, extra={"k": "v"})
    assert OptimiserSpec.from_dict(spec.to_dict()) == spec


def test_diagnostics_round_trip():
    diag = CalibrationDiagnostics(
        objective_history=[1.0, 0.5],
        parameter_history=[{"x": 1.0}],
        timing_ms=3.0,
        warnings=["w"],
        extra={"note": "ok"},
    )
    assert CalibrationDiagnostics.from_dict(diag.to_dict()) == diag


def test_calibration_result_round_trip():
    cr = _sample_result()
    back = CalibrationResult.from_dict(cr.to_dict())
    assert back == cr
    # spot-check the load-bearing identity fields survive byte-for-byte
    assert back.provenance.id == cr.provenance.id
    assert back.provenance.timestamp == cr.provenance.timestamp
    assert back.provenance.market_snapshot_id == cr.provenance.market_snapshot_id
    assert back.fit.objective is ObjectiveKind.WEIGHTED_SSE
    assert back.optimiser_run.spec == cr.optimiser_run.spec
    assert back.diagnostics == cr.diagnostics


def test_calibration_result_round_trip_no_snapshot():
    cr = build_calibration_result(
        model_class="sabr",
        parameters={"alpha": 0.2},
        residuals=[0.0],
        optimiser=OptimiserSpec("differential_evolution", 1e-6, 1000),
        iterations=1,
        converged=False,
    )
    assert cr.provenance.market_snapshot_id is None
    back = CalibrationResult.from_dict(cr.to_dict())
    assert back == cr
    assert back.provenance.market_snapshot_id is None


def test_payload_is_json_native():
    cr = _sample_result()
    d = cr.to_dict()
    s = json.dumps(d)  # must not raise
    # and reconstructs from the JSON-decoded dict (strings for uuid/datetime)
    back = CalibrationResult.from_dict(json.loads(s))
    assert back == cr


def test_payload_carries_schema_version():
    d = _sample_result().to_dict()
    assert d["_schema_version"] == 3  # bumped: rms/max no longer serialised (derived)
    # the derived metrics are NOT in the payload anymore
    assert "rms_residual" not in d and "max_residual" not in d


def test_rms_max_are_derived_properties():
    cr = build_calibration_result(
        model_class="m", parameters={}, residuals=[3.0, -4.0],
        optimiser=OptimiserSpec("x", 0.0, 0), iterations=0, converged=True,
    )
    assert cr.fit.rms_residual == pytest.approx((25 / 2) ** 0.5)
    assert cr.fit.max_residual == 4.0
    # not constructor params anymore — can't be passed (so can't drift from residuals)
    with pytest.raises(TypeError):
        CalibrationResult(  # type: ignore[call-arg]
            id=uuid4(), timestamp=datetime.now(timezone.utc), code_version="x",
            model_class="m", parameters={}, quotes_fitted=[], weights=[],
            objective=ObjectiveKind.SSE, residuals=[1.0], rms_residual=99.0,
            max_residual=99.0, iterations=0, optimiser=OptimiserSpec("x", 0.0, 0),
            converged=True,
        )


def test_empty_residuals_rejected():
    # A fit with no targets is not a fit — empty residuals are unconstructible
    # (an empty vector would make rms_residual read as a false-perfect 0.0).
    with pytest.raises(ValueError, match="residuals must be non-empty"):
        build_calibration_result(
            model_class="m", parameters={}, residuals=[],
            optimiser=OptimiserSpec("x", 0.0, 0), iterations=0, converged=True,
        )


# --------------------------------------------------------------------------- #
# Clock
# --------------------------------------------------------------------------- #

def test_auto_timestamp_is_tz_aware_utc():
    cr = build_calibration_result(
        model_class="m",
        parameters={},
        residuals=[0.0],
        optimiser=OptimiserSpec("x", 0.0, 0),
        iterations=0,
        converged=True,
    )
    assert cr.provenance.timestamp.tzinfo is not None
    assert cr.provenance.timestamp.utcoffset().total_seconds() == 0.0


def test_id_and_timestamp_injectable():
    fixed_id = UUID("12345678-1234-5678-1234-567812345678")
    fixed_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    cr = build_calibration_result(
        model_class="m",
        parameters={},
        residuals=[0.0],
        optimiser=OptimiserSpec("x", 0.0, 0),
        iterations=0,
        converged=True,
        id=fixed_id,
        timestamp=fixed_ts,
    )
    assert cr.provenance.id == fixed_id
    assert cr.provenance.timestamp == fixed_ts


def test_auto_ids_are_distinct():
    kw = dict(
        model_class="m",
        parameters={},
        residuals=[0.0],
        optimiser=OptimiserSpec("x", 0.0, 0),
        iterations=0,
        converged=True,
    )
    assert build_calibration_result(**kw).provenance.id != build_calibration_result(**kw).provenance.id


# --------------------------------------------------------------------------- #
# Core atom support added by this slice
# --------------------------------------------------------------------------- #

def test_atom_uuid_round_trip():
    u = uuid4()
    assert _serialise_atom(u) == str(u)
    assert _deserialise_atom(str(u), UUID) == u


def test_atom_datetime_round_trip():
    dt = datetime(2026, 6, 20, 14, 30, tzinfo=timezone.utc)
    assert _serialise_atom(dt) == dt.isoformat()
    assert _deserialise_atom(dt.isoformat(), datetime) == dt


def test_atom_dict_and_tuple_recurse():
    u = uuid4()
    assert _serialise_atom((1, u)) == [1, str(u)]
    assert _serialise_atom({"k": u}) == {"k": str(u)}
