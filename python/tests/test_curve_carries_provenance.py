"""Curve types carry a `calibration_result` field — F2 of the bootstrapper campaign.

DiscountCurve already did; this pins that SurvivalCurve and AADDiscountCurve do
too, so credit/AAD bootstrappers can attach a record like the rates curves.
"""

from datetime import date

import pytest

from pricebook.calibration import curve_calibration_record, pillar_parameters
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve
from pricebook.curves.aad_curves import AADDiscountCurve
from pricebook.curves.aad import Number
from pricebook.db.db import PricebookDB

REF = date(2026, 1, 1)


def _curves():
    return [
        DiscountCurve(REF, [date(2027, 1, 1), date(2029, 1, 1)], [0.97, 0.92]),
        SurvivalCurve(REF, [date(2027, 1, 1), date(2029, 1, 1)], [0.98, 0.95]),
        AADDiscountCurve(REF, [date(2027, 1, 1), date(2029, 1, 1)], [Number(0.97), Number(0.92)]),
    ]


@pytest.mark.parametrize("curve", _curves(), ids=lambda c: type(c).__name__)
def test_has_calibration_result_field_defaulting_none(curve):
    assert curve.calibration_result is None


def test_record_attaches_and_persists():
    curve = SurvivalCurve(REF, [date(2027, 1, 1), date(2029, 1, 1)], [0.98, 0.95])
    curve.calibration_result = curve_calibration_record(
        model_class="cds_survival_bootstrap",
        parameters=pillar_parameters(
            [date(2027, 1, 1), date(2029, 1, 1)], [0.98, 0.95], label="survival"),
        residuals=[1e-12, -2e-12],
        quotes_fitted=["cds_2027", "cds_2029"],
        algorithm="brentq-sequential",
        iterations=2,
    )
    assert curve.calibration_result is not None
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(curve.calibration_result)
        assert db.load_calibration(cid) == curve.calibration_result
