"""CanonicalCalibrationResult mixin contract — G1 unification Phase 4.

The mixin factors out the field/accessor scaffolding every family result was
duplicating: `to_calibration_result()` (stored-or-lazy-build+cache),
`_build_calibration_record()` (abstract), and the `calibration_id` property.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pricebook.calibration import (
    CalibrationResult,
    CanonicalCalibrationResult,
    ObjectiveKind,
    OptimiserSpec,
)


@dataclass
class _Fam(CanonicalCalibrationResult):
    x: float
    calibration_result: CalibrationResult | None = None

    def _build_calibration_record(self) -> CalibrationResult:
        return CalibrationResult.new(
            model_class="toy",
            parameters={"x": self.x},
            residuals=[self.x],
            optimiser=OptimiserSpec("none", 0.0, 0),
            iterations=0,
            converged=True,
            objective=ObjectiveKind.SSE,
        )


def test_lazy_build_and_cache():
    f = _Fam(x=1.5)
    assert f.calibration_result is None
    assert f.calibration_id is None          # no build side-effect
    cr = f.to_calibration_result()
    assert cr.model_class == "toy"
    assert f.calibration_result is cr        # cached
    assert f.to_calibration_result() is cr   # stable across calls
    assert f.calibration_id == str(cr.id)


def test_stored_record_is_returned_verbatim():
    pre = CalibrationResult.new(
        model_class="pre", parameters={}, residuals=[],
        optimiser=OptimiserSpec("x", 0.0, 0), iterations=0, converged=True,
    )
    f = _Fam(x=2.0, calibration_result=pre)
    assert f.to_calibration_result() is pre   # eager population wins, no rebuild
    assert f.calibration_id == str(pre.id)


def test_abstract_method_raises():
    @dataclass
    class _Bare(CanonicalCalibrationResult):
        calibration_result: CalibrationResult | None = None

    with pytest.raises(NotImplementedError):
        _Bare().to_calibration_result()


def test_persists_via_db():
    from pricebook.db.db import PricebookDB
    f = _Fam(x=3.0)
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(f)          # polymorphic save → to_calibration_result()
        assert db.load_calibration(cid) == f.to_calibration_result()
