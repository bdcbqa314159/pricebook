"""Bootstrapper campaign Tier 3b — bond curve + tenor basis provenance.

These two return non-bare-curve artifacts (a BondCurveResult wrapper and a
(IBORCurve, TenorBasis) tuple). Provenance is carried by the underlying
DiscountCurve and exposed via a forwarding property on the wrapper / IBORCurve.
"""

from datetime import date, timedelta

import pytest

from pricebook.core.discount_curve import DiscountCurve
from pricebook.curves.bond_curve import BondQuote, bootstrap_curve_from_bonds
from pricebook.db.db import PricebookDB
from pricebook.fixed_income.ibor_curve import (
    EURIBOR_3M_CONVENTIONS,
    EURIBOR_6M_CONVENTIONS,
    bootstrap_ibor,
)
from pricebook.fixed_income.tenor_basis import bootstrap_tenor_basis

REF = date(2026, 4, 27)


def _bond_quotes():
    return [
        BondQuote(date(2027, 6, 1), 0.00, 97.5, frequency=2),
        BondQuote(date(2028, 6, 1), 0.04, 99.0, frequency=2),
        BondQuote(date(2031, 6, 1), 0.045, 97.0, frequency=2),
        BondQuote(date(2036, 6, 1), 0.0475, 95.0, frequency=2),
    ]


@pytest.mark.parametrize("method", ["sequential", "global", "nelson_siegel", "svensson"])
def test_bond_curve_attaches_record(method):
    result = bootstrap_curve_from_bonds(REF, _bond_quotes(), method=method)
    cr = result.calibration_result
    assert cr is not None
    assert cr is result.discount_curve.calibration_result  # carried by the curve
    assert cr.fit.model_class == "bond_curve_bootstrap"
    assert cr.optimiser_run.spec.algorithm == method
    assert len(cr.fit.residuals) == result.n_bonds
    assert cr.diagnostics.extra["rmse_bp"] == pytest.approx(result.rmse_bp)
    if method in ("nelson_siegel", "svensson"):
        assert cr.diagnostics.extra["shape"]  # parametric betas captured
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr


def test_tenor_basis_attaches_record():
    ois = DiscountCurve.flat(REF, 0.03)
    ibor_3m = bootstrap_ibor(
        REF, EURIBOR_3M_CONVENTIONS, ois,
        swaps=[(REF + timedelta(days=730), 0.033),
               (REF + timedelta(days=1825), 0.035),
               (REF + timedelta(days=3650), 0.037)],
    )
    quotes = [(REF + timedelta(days=730), 0.0005),
              (REF + timedelta(days=1825), 0.0010),
              (REF + timedelta(days=3650), 0.0012)]
    ibor_6m, basis = bootstrap_tenor_basis(
        REF, ibor_3m, ois,
        basis_swap_quotes=quotes,
        long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
    )
    cr = ibor_6m.calibration_result  # forwarded from the long-tenor projection
    assert cr is not None
    assert cr.fit.model_class == "tenor_basis_bootstrap"
    assert len(cr.fit.residuals) == len(quotes)
    assert max(abs(r) for r in cr.fit.residuals) < 1e-9  # basis swaps reprice
    assert len(cr.diagnostics.extra["spreads_bp"]) == len(quotes)
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
