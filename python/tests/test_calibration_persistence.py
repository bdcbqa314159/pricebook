"""CalibrationResult persistence — G1 unification Phase 0 Slice 2.

The canonical CalibrationResult is now serialisable (Slice 1); this slice makes
it load-bearing: PricebookDB.save_calibration / load_calibration /
load_calibration_raw / list_calibrations. Covers round-trip reconstruction,
the denormalised audit-chain columns, idempotency, and file-backed durability.
"""

from __future__ import annotations

import os
import tempfile
from uuid import uuid4

import pytest

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)
from pricebook.db.db import PricebookDB


@pytest.fixture
def db():
    d = PricebookDB(":memory:")
    yield d
    d.close()


def _result(model_class="HullWhite", snapshot_id=None, **kw):
    return CalibrationResult.new(
        model_class=model_class,
        parameters={"a": 0.03, "sigma": 0.012},
        residuals=[0.001, -0.002, 0.0005],
        optimiser=OptimiserSpec("L-BFGS-B", 1e-10, 500, seed=42),
        iterations=37,
        converged=True,
        objective=ObjectiveKind.WEIGHTED_SSE,
        quotes_fitted=["1Y", "2Y", "5Y"],
        weights=[1.0, 0.5, 0.25],
        diagnostics=CalibrationDiagnostics(extra={"rmse_vol": 0.0008}),
        market_snapshot_id=snapshot_id,
        **kw,
    )


class TestRoundTrip:

    def test_save_returns_id(self, db):
        cr = _result()
        assert db.save_calibration(cr) == str(cr.id)

    def test_load_reconstructs_equal(self, db):
        cr = _result()
        db.save_calibration(cr)
        back = db.load_calibration(cr.id)
        assert back == cr

    def test_load_accepts_str_or_uuid(self, db):
        cr = _result()
        db.save_calibration(cr)
        assert db.load_calibration(str(cr.id)) == cr
        assert db.load_calibration(cr.id) == cr

    def test_load_missing_returns_none(self, db):
        assert db.load_calibration(uuid4()) is None


class TestDenormalisedColumns:
    """The columns that turn the previously build-and-drop fields live."""

    def test_raw_columns_track_fields(self, db):
        snap = uuid4()
        cr = _result(snapshot_id=snap)
        db.save_calibration(cr)
        raw = db.load_calibration_raw(cr.id)
        assert raw["model_class"] == "HullWhite"
        assert raw["objective"] == ObjectiveKind.WEIGHTED_SSE.value
        assert raw["converged"] == 1            # bool → INTEGER
        assert raw["iterations"] == 37
        assert raw["rms_residual"] == pytest.approx(cr.rms_residual)
        assert raw["max_residual"] == pytest.approx(cr.max_residual)
        assert raw["market_snapshot_id"] == str(snap)
        # convention payload is flat (no "type"/"params" envelope)
        assert raw["result"]["model_class"] == "HullWhite"
        assert raw["result"]["_schema_version"] == 1

    def test_null_snapshot(self, db):
        cr = _result(snapshot_id=None)
        db.save_calibration(cr)
        assert db.load_calibration_raw(cr.id)["market_snapshot_id"] is None


class TestListAndFilter:

    def test_list_all(self, db):
        db.save_calibration(_result("HullWhite"))
        db.save_calibration(_result("SABR"))
        assert len(db.list_calibrations()) == 2

    def test_filter_by_model_class(self, db):
        db.save_calibration(_result("HullWhite"))
        db.save_calibration(_result("HullWhite"))
        db.save_calibration(_result("SABR"))
        hw = db.list_calibrations(model_class="HullWhite")
        assert len(hw) == 2
        assert {r["model_class"] for r in hw} == {"HullWhite"}

    def test_filter_by_snapshot_audit_chain(self, db):
        snap = uuid4()
        db.save_calibration(_result(snapshot_id=snap))
        db.save_calibration(_result(snapshot_id=uuid4()))
        hits = db.list_calibrations(market_snapshot_id=str(snap))
        assert len(hits) == 1


class TestIdempotencyAndDurability:

    def test_resave_updates_not_duplicates(self, db):
        cr = _result()
        db.save_calibration(cr)
        db.save_calibration(cr)
        assert len(db.list_calibrations()) == 1

    def test_calibration_results_is_system_table(self, db):
        with pytest.raises(ValueError):
            db.drop_table("calibration_results")

    def test_survives_file_reopen(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        try:
            cr = _result()
            db1 = PricebookDB(path)
            db1.save_calibration(cr)
            db1.close()

            db2 = PricebookDB(path)
            assert db2.load_calibration(cr.id) == cr
            db2.close()
        finally:
            os.remove(path)
