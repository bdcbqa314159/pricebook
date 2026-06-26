"""End-to-end calibration loop — G1 unification Phase 3a.

Proves the build → store → read loop is closed and uniform: `db.save_calibration`
now accepts any family result exposing `to_calibration_result()` (not just a
canonical `CalibrationResult`), so persistence is the single consumer of every
calibrator's record. Uses `JumpCalibrationResult` as a cheap real adopter
(all-primitive fields, no heavy model object).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from pricebook.calibration import CalibrationResult, ObjectiveKind, OptimiserSpec
from pricebook.db.db import PricebookDB
from pricebook.models.jump_calibration import JumpCalibrationResult
from tests.conftest import build_calibration_result


@pytest.fixture
def db():
    d = PricebookDB(":memory:")
    yield d
    d.close()


def _family_with_stored_cr(snapshot_id=None) -> JumpCalibrationResult:
    cr = build_calibration_result(
        model_class="jump_merton",
        parameters={"lambda": 0.3, "muJ": -0.1, "sigmaJ": 0.15},
        residuals=[0.001, -0.0005],
        optimiser=OptimiserSpec("L-BFGS-B", 1e-8, 200),
        iterations=12,
        converged=True,
        objective=ObjectiveKind.SSE,
        market_snapshot_id=snapshot_id,
    )
    return JumpCalibrationResult(
        model_type="merton",
        params={"lambda": 0.3, "muJ": -0.1, "sigmaJ": 0.15},
        rmse_vol=0.004,
        market_vols=[0.20, 0.21],
        model_vols=[0.201, 0.2095],
        strikes=[95.0, 105.0],
        n_params=3,
        calibration_result=cr,
    )


def _family_without_cr() -> JumpCalibrationResult:
    # calibration_result=None → to_calibration_result() rebuilds on demand
    return JumpCalibrationResult(
        model_type="kou",
        params={"lambda": 0.5, "p": 0.4},
        rmse_vol=0.006,
        market_vols=[0.18, 0.19, 0.20],
        model_vols=[0.182, 0.189, 0.203],
        strikes=[90.0, 100.0, 110.0],
        n_params=2,
        calibration_result=None,
    )


class TestPolymorphicSave:

    def test_canonical_result_still_passes_through(self, db):
        cr = build_calibration_result(
            model_class="m", parameters={}, residuals=[0.0],
            optimiser=OptimiserSpec("x", 0.0, 0), iterations=0, converged=True,
        )
        assert db.save_calibration(cr) == str(cr.provenance.id)
        assert db.load_calibration(cr.provenance.id) == cr

    def test_family_result_with_stored_cr(self, db):
        fam = _family_with_stored_cr()
        cid = db.save_calibration(fam)
        assert cid == str(fam.calibration_result.provenance.id)
        # the loop: what we load back equals the family's canonical record
        assert db.load_calibration(cid) == fam.to_calibration_result()

    def test_family_result_rebuilds_on_demand(self, db):
        # No stored cr → save() rebuilds via to_calibration_result(), which
        # mints a fresh id each call. So we assert content + that the persisted
        # id is the one returned (not equality against a second rebuild).
        fam = _family_without_cr()
        cid = db.save_calibration(fam)
        loaded = db.load_calibration(cid)
        assert str(loaded.provenance.id) == cid
        assert loaded.fit.model_class == "jump_kou"
        assert loaded.fit.parameters == {"lambda": 0.5, "p": 0.4}
        # Reconstructed from a hand-built result — no optimiser ran, so
        # convergence is not captured (None), not guessed.
        assert loaded.optimiser_run.converged is None


class TestLoopAndAuditChain:

    def test_denormalised_columns_from_family(self, db):
        fam = _family_with_stored_cr()
        db.save_calibration(fam)
        raw = db.load_calibration_raw(fam.calibration_result.provenance.id)
        assert raw["model_class"] == "jump_merton"
        assert raw["converged"] == 1

    def test_audit_query_by_snapshot_through_family(self, db):
        snap = uuid4()
        db.save_calibration(_family_with_stored_cr(snapshot_id=snap))
        db.save_calibration(_family_with_stored_cr(snapshot_id=uuid4()))
        hits = db.list_calibrations(market_snapshot_id=str(snap))
        assert len(hits) == 1
        assert hits[0]["model_class"] == "jump_merton"


def test_list_calibrations_filters_converged_none():
    """Regression: a not-captured record (converged=None, the tri-state) must be
    findable via list_calibrations(converged=None). Before the fix, `_build_where`
    emitted `converged = NULL` (never true in SQL) and returned zero rows."""
    captured = build_calibration_result(
        model_class="m_captured", parameters={}, residuals=[0.0], quotes_fitted=["q"],
        optimiser=OptimiserSpec("x", 0.0, 0), iterations=5, converged=True,
    )
    not_captured = build_calibration_result(
        model_class="m_reconstructed", parameters={}, residuals=[0.0], quotes_fitted=["q"],
        optimiser=OptimiserSpec("unspecified", 0.0, 0), iterations=0, converged=None,
    )
    with PricebookDB(":memory:") as db:
        db.save_calibration(captured)
        db.save_calibration(not_captured)
        none_rows = db.list_calibrations(converged=None)
        assert [r["model_class"] for r in none_rows] == ["m_reconstructed"]
        assert [r["model_class"] for r in db.list_calibrations(converged=True)] == ["m_captured"]
