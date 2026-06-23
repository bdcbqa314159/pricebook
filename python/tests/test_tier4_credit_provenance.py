"""Bootstrapper campaign Tier 4 — credit curve provenance.

cds.bootstrap_credit_curve, cds_market.bootstrap_from_upfronts, and
sovereign_cds.bootstrap_sovereign_hazard all build a SurvivalCurve; each now
attaches its canonical record (curve-carries-provenance).
"""

from datetime import date

from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.cds import bootstrap_credit_curve
from pricebook.credit.cds_market import bootstrap_from_upfronts, build_cds_curve
from pricebook.credit.sovereign_cds import bootstrap_sovereign_hazard
from pricebook.db.db import PricebookDB

REF = date(2026, 1, 1)


def _disc():
    return DiscountCurve.flat(REF, 0.03)


def _spreads():
    return [(date(2027, 1, 1), 0.005), (date(2029, 1, 1), 0.008),
            (date(2031, 1, 1), 0.012), (date(2036, 1, 1), 0.015)]


def test_credit_curve_attaches_record():
    curve = bootstrap_credit_curve(REF, _spreads(), _disc())
    cr = curve.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "credit_curve_bootstrap"
    assert len(cr.fit.residuals) == len(_spreads())
    assert max(abs(r) for r in cr.fit.residuals) < 1e-6  # reprices at par
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr


def test_build_cds_curve_inherits_record():
    # build_cds_curve delegates to bootstrap_credit_curve → inherits provenance.
    curve = build_cds_curve(REF, {1: 0.005, 5: 0.012, 10: 0.015}, _disc())
    assert curve.calibration_result is not None
    assert curve.calibration_result.fit.model_class == "credit_curve_bootstrap"


def test_upfront_bootstrap_attaches_record():
    curve = bootstrap_from_upfronts(REF, {3: 0.01, 5: 0.0218, 10: 0.04}, 0.01, _disc())
    cr = curve.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "cds_upfront_bootstrap"
    assert len(cr.fit.residuals) == 3
    assert max(abs(r) for r in cr.fit.residuals) < 1e-6  # reprices the upfronts
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr


def test_sovereign_hazard_attaches_record():
    result = bootstrap_sovereign_hazard(REF, {1: 50, 5: 120, 10: 180}, _disc(), "BR")
    cr = result.calibration_result  # forwarded from the survival curve
    assert cr is not None
    assert cr is result.survival_curve.calibration_result
    assert cr.fit.model_class == "sovereign_hazard_bootstrap"
    assert len(cr.fit.residuals) == 3
    assert set(cr.fit.parameters) == {"hazard_1y", "hazard_5y", "hazard_10y"}
    assert cr.diagnostics.extra["country_code"] == "BR"
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
