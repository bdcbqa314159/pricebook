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
from tests.conftest import build_calibration_result


@dataclass
class _Fam(CanonicalCalibrationResult):
    x: float
    calibration_result: CalibrationResult | None = None

    def _build_calibration_record(self) -> CalibrationResult:
        return build_calibration_result(
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
    assert cr.fit.model_class == "toy"
    assert f.calibration_result is cr        # cached
    assert f.to_calibration_result() is cr   # stable across calls
    assert f.calibration_id == str(cr.provenance.id)


def test_stored_record_is_returned_verbatim():
    pre = build_calibration_result(
        model_class="pre", parameters={}, residuals=[0.0],
        optimiser=OptimiserSpec("x", 0.0, 0), iterations=0, converged=True,
    )
    f = _Fam(x=2.0, calibration_result=pre)
    assert f.to_calibration_result() is pre   # eager population wins, no rebuild
    assert f.calibration_id == str(pre.provenance.id)


def test_abstract_method_enforced_at_instantiation():
    # ABC: a subclass that doesn't implement _build_calibration_record can't be
    # instantiated at all — the contract is enforced up-front, not on first use.
    @dataclass
    class _Bare(CanonicalCalibrationResult):
        calibration_result: CalibrationResult | None = None

    with pytest.raises(TypeError, match="abstract"):
        _Bare()


def test_missing_field_enforced_at_class_definition():
    # __init_subclass__: a family that inherits the mixin but forgets the
    # `calibration_result` field fails when the class is DEFINED, not later.
    with pytest.raises(TypeError, match="calibration_result"):
        @dataclass
        class _NoField(CanonicalCalibrationResult):
            y: float = 0.0

            def _build_calibration_record(self):  # implemented, but no field
                raise AssertionError("unreachable")


def test_persists_via_db():
    from pricebook.db.db import PricebookDB
    f = _Fam(x=3.0)
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(f)          # polymorphic save → to_calibration_result()
        assert db.load_calibration(cid) == f.to_calibration_result()


def test_save_calibration_rejects_non_conforming():
    # The persistence boundary refuses anything that isn't a CalibrationResult
    # and can't produce one — a non-conforming result can't enter the audit chain.
    from pricebook.db.db import PricebookDB
    with PricebookDB(":memory:") as db:
        with pytest.raises(TypeError, match="CalibrationResult"):
            db.save_calibration(object())
