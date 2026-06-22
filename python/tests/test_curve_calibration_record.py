"""Shared curve-bootstrapper provenance helper — F1 of the bootstrapper campaign."""

from datetime import date
from uuid import uuid4

import pytest

from pricebook.calibration import (
    CalibrationResult,
    ObjectiveKind,
    curve_calibration_record,
    pillar_parameters,
)
from pricebook.db.db import PricebookDB


def test_pillar_parameters_labels_by_date():
    p = pillar_parameters([date(2027, 1, 1), date(2030, 6, 30)], [0.97, 0.88], label="df")
    assert p == {"df(2027-01-01)": 0.97, "df(2030-06-30)": 0.88}


def test_pillar_parameters_custom_label():
    p = pillar_parameters([date(2027, 1, 1)], [0.99], label="survival")
    assert p == {"survival(2027-01-01)": 0.99}


def _record(**kw):
    base = dict(
        model_class="discount_curve_bootstrap",
        parameters={"df(2027-01-01)": 0.97},
        residuals=[1e-13],
        quotes_fitted=["swap_2027"],
        algorithm="brentq-sequential",
        iterations=1,
    )
    base.update(kw)
    return curve_calibration_record(**base)


def test_builds_a_valid_record():
    cr = _record()
    assert isinstance(cr, CalibrationResult)
    assert cr.fit.model_class == "discount_curve_bootstrap"
    assert cr.optimiser_run.spec.algorithm == "brentq-sequential"
    assert cr.optimiser_run.converged is True
    assert cr.fit.objective is ObjectiveKind.SSE


def test_round_trip_and_persist():
    snap = uuid4()
    cr = _record(market_snapshot_id=snap,
                 optimiser_extra={"interpolation": "log_linear"},
                 diagnostics_extra={"n_swaps": 1})
    assert cr.provenance.market_snapshot_id == snap
    assert CalibrationResult.from_dict(cr.to_dict()) == cr
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
        assert db.list_calibrations(model_class="discount_curve_bootstrap")[0]["calibration_id"] == cid


def test_inherits_calibrationfit_enforcement():
    # snake_case audit-key enforcement comes for free via CalibrationFit
    with pytest.raises(ValueError, match="snake_case"):
        _record(model_class="DiscountCurve")
    # parallel-array length agreement, too
    with pytest.raises(ValueError, match="quotes_fitted length"):
        _record(residuals=[0.1, 0.2], quotes_fitted=["only_one"])
