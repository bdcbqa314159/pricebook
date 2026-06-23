"""Bootstrapper campaign Tier 3a — xccy basis + AAD curve provenance."""

from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.aad import Number
from pricebook.curves.aad_curves import aad_bootstrap
from pricebook.db.db import PricebookDB
from pricebook.fixed_income.xccy_basis import bootstrap_basis_curve

REF = date(2026, 1, 1)


def test_xccy_basis_curve_attaches_record():
    base = DiscountCurve.flat(REF, 0.03)
    quote = DiscountCurve.flat(REF, 0.01)
    fwds = [(date(2027, 1, 1), 1.02), (date(2029, 1, 1), 1.05), (date(2031, 1, 1), 1.08)]
    curve = bootstrap_basis_curve(REF, 1.0, fwds, base, quote)
    cr = curve.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "xccy_basis_bootstrap"
    assert cr.optimiser_run.spec.algorithm == "closed_form"
    assert len(cr.fit.residuals) == len(fwds)
    assert max(abs(r) for r in cr.fit.residuals) < 1e-9  # CIP exact by construction
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr


def test_aad_bootstrap_attaches_record():
    deps = [(date(2027, 1, 1), Number(0.03))]
    swaps = [(date(2029, 1, 1), Number(0.032)), (date(2031, 1, 1), Number(0.034))]
    curve = aad_bootstrap(REF, deps, swaps)
    cr = curve.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "aad_discount_curve_bootstrap"
    assert len(cr.fit.residuals) == len(deps) + len(swaps)
    assert max(abs(r) for r in cr.fit.residuals) < 1e-9  # par by construction
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
        assert db.list_calibrations(model_class="aad_discount_curve_bootstrap")[0]["calibration_id"] == cid
