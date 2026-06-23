"""Bootstrapper campaign Tier 5 — inflation + equity curve provenance.

bootstrap_cpi_curve (CPICurve), real_yield_curve_bootstrap (RealYieldCurveResult),
and dividend_curve_bootstrap (DividendCurve) each gained a `calibration_result`
field (F2-equivalent for this tier) and now attach their canonical record.
"""

from datetime import date

from pricebook.equity.dividend_advanced import dividend_curve_bootstrap
from pricebook.fixed_income.inflation import bootstrap_cpi_curve
from pricebook.fixed_income.inflation_bond_advanced import real_yield_curve_bootstrap
from pricebook.db.db import PricebookDB

REF = date(2026, 1, 1)


def test_cpi_curve_attaches_record():
    quotes = [(date(2027, 1, 1), 0.025), (date(2029, 1, 1), 0.027), (date(2031, 1, 1), 0.03)]
    curve = bootstrap_cpi_curve(REF, 100.0, quotes)
    cr = curve.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "cpi_curve_bootstrap"
    assert len(cr.fit.residuals) == len(quotes)
    assert max(abs(r) for r in cr.fit.residuals) < 1e-9  # closed form, exact
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr


def test_real_yield_curve_attaches_record():
    result = real_yield_curve_bootstrap(
        [98.0, 95.0, 90.0], [100, 100, 100], [0.01, 0.012, 0.015],
        [2.0, 5.0, 10.0], 100.0, 110.0,
    )
    cr = result.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "real_yield_curve_bootstrap"
    assert len(cr.fit.residuals) == 3
    assert max(abs(r) for r in cr.fit.residuals) < 1e-6  # reprices each linker
    assert "calibration_result" not in result.to_dict()  # not leaked into the dict
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr


def test_dividend_curve_attaches_record():
    curve = dividend_curve_bootstrap(100.0, 0.03, [1.0, 2.0, 5.0], [2.0, 3.8, 9.0])
    cr = curve.calibration_result
    assert cr is not None
    assert cr.fit.model_class == "dividend_curve_bootstrap"
    assert len(cr.fit.residuals) == 3
    assert max(abs(r) for r in cr.fit.residuals) < 1e-9  # q = D/(S·T), exact
    assert "calibration_result" not in curve.to_dict()
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
