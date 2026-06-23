"""Closing conformance gate for the bootstrapper-provenance campaign.

Turns "every curve bootstrapper carries an auditable calibration record" from a
convention into an enforced invariant, mirroring the CanonicalCalibrationResult
ABC/field guard on the mixin side.

Two layers:

1. **Discovery guard** — AST-scan the whole package for public module-level
   ``bootstrap*`` / ``*_bootstrap`` functions and assert each is classified
   either as COVERED (must attach provenance) or ALLOWLISTED (deliberately
   excluded, with a reason). A newly-added bootstrapper that nobody classified
   fails this test — it cannot silently skip provenance.

2. **Behavioural gate** — actually run each COVERED producer on a small fixture
   and assert the returned curve / wrapper carries a non-None
   ``calibration_result`` that round-trips through the DB.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import date

import pytest

from pricebook.db.db import PricebookDB

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "pricebook"

# ── Classification ──────────────────────────────────────────────────────────

# Public bootstrappers that MUST attach a canonical calibration_result.
COVERED = {
    "bootstrap",                    # curves/bootstrap — flagship deposit/swap/FRA/future
    "bootstrap_forward_curve",      # curves/bootstrap — projection curve
    "bootstrap_ibor",               # fixed_income/ibor_curve (IBORCurve forwards)
    "bootstrap_rfr",                # curves/rfr_bootstrap (RFRCurveResult forwards)
    "bootstrap_ois",                # fixed_income/ois
    "bootstrap_spread_curve",       # fixed_income/rfr
    "bootstrap_basis_curve",        # fixed_income/xccy_basis
    "aad_bootstrap",                # curves/aad_curves
    "bootstrap_curve_from_bonds",   # curves/bond_curve (BondCurveResult forwards)
    "bootstrap_tenor_basis",        # fixed_income/tenor_basis (IBORCurve forwards)
    "bootstrap_credit_curve",       # credit/cds
    "bootstrap_from_upfronts",      # credit/cds_market
    "bootstrap_sovereign_hazard",   # credit/sovereign_cds (SovereignHazardResult forwards)
    "bootstrap_cpi_curve",          # fixed_income/inflation
    "real_yield_curve_bootstrap",   # fixed_income/inflation_bond_advanced
    "dividend_curve_bootstrap",     # equity/dividend_advanced
    # Pre-existing — behaviourally tested in test_bond_hazard_calibration_result.py
    "bootstrap_hazard_from_bonds",
    "bootstrap_hazard_mixed",
    "bootstrap_hazard_adaptive",
}

# Deliberately excluded, each with a reason.
ALLOWLIST = {
    "global_bootstrap": "low-level solver primitive; the calling bootstrapper owns provenance",
    "coupled_bootstrap": "low-level dual-curve solver primitive; callers own provenance",
    "bootstrap_ci": "statistical resampling (confidence interval), not a curve calibration",
}


def _discover_bootstrappers() -> dict[str, str]:
    """Map public module-level bootstrapper name -> relative file path."""
    found: dict[str, str] = {}
    for path in PKG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:  # module level only
            if not isinstance(node, ast.FunctionDef):
                continue
            name = node.name
            if name.startswith("_"):
                continue
            if name.startswith("bootstrap") or name.endswith("_bootstrap"):
                found[name] = str(path.relative_to(PKG_ROOT))
    return found


def test_every_bootstrapper_is_classified():
    """No public bootstrapper may be left unclassified (covered or allowlisted)."""
    discovered = _discover_bootstrappers()
    classified = COVERED | set(ALLOWLIST)
    unclassified = {n: f for n, f in discovered.items() if n not in classified}
    assert not unclassified, (
        "Unclassified bootstrappers (add to COVERED or ALLOWLIST in this test):\n"
        + "\n".join(f"  {n}  @ {f}" for n, f in sorted(unclassified.items()))
    )


def test_covered_set_matches_discovery():
    """COVERED must not name a function that no longer exists (drift guard)."""
    discovered = set(_discover_bootstrappers())
    stale = (COVERED | set(ALLOWLIST)) - discovered
    assert not stale, f"Classified names no longer found in the package: {sorted(stale)}"


# ── Behavioural gate ────────────────────────────────────────────────────────

REF = date(2026, 1, 1)


def _flat(rate=0.03):
    from pricebook.core.discount_curve import DiscountCurve
    return DiscountCurve.flat(REF, rate)


def _r_bootstrap():
    from pricebook.curves.bootstrap import bootstrap
    c = bootstrap(REF, [(date(2026, 7, 1), 0.05)],
                  [(date(2028, 1, 1), 0.045), (date(2031, 1, 1), 0.04)])
    return c.calibration_result


def _r_forward_curve():
    from pricebook.curves.bootstrap import bootstrap_forward_curve
    c = bootstrap_forward_curve(
        REF, [(date(2028, 1, 1), 0.045), (date(2031, 1, 1), 0.04)], _flat(),
        deposits=[(date(2026, 7, 1), 0.05)],
    )
    return c.calibration_result


def _r_ois():
    from dateutil.relativedelta import relativedelta
    from pricebook.fixed_income.ois import bootstrap_ois
    rates = [(REF + relativedelta(months=6), 0.051), (REF + relativedelta(years=1), 0.049),
             (REF + relativedelta(years=2), 0.047), (REF + relativedelta(years=5), 0.044)]
    return bootstrap_ois(REF, rates).calibration_result


def _r_spread_curve():
    from dateutil.relativedelta import relativedelta
    from pricebook.fixed_income.rfr import bootstrap_spread_curve
    rates = [(REF + relativedelta(years=1), 0.052), (REF + relativedelta(years=2), 0.050),
             (REF + relativedelta(years=5), 0.047)]
    return bootstrap_spread_curve(REF, rates, _flat()).calibration_result


def _r_rfr():
    from pricebook.curves.rfr_bootstrap import bootstrap_rfr, RFRCurveInputs
    inputs = RFRCurveInputs(
        overnight_rate=0.053,
        deposits=[(date(2026, 4, 15), 0.052), (date(2026, 7, 15), 0.051)],
    )
    return bootstrap_rfr("USD", REF, inputs).calibration_result


def _r_ibor():
    from dateutil.relativedelta import relativedelta
    from pricebook.fixed_income.ibor_curve import bootstrap_ibor, EURIBOR_3M_CONVENTIONS
    swaps = [(REF + relativedelta(years=2), 0.033), (REF + relativedelta(years=5), 0.035),
             (REF + relativedelta(years=10), 0.037)]
    return bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, _flat(), swaps=swaps).calibration_result


def _r_basis_curve():
    from pricebook.fixed_income.xccy_basis import bootstrap_basis_curve
    fwds = [(date(2027, 1, 1), 1.02), (date(2029, 1, 1), 1.05)]
    return bootstrap_basis_curve(REF, 1.0, fwds, _flat(0.03), _flat(0.01)).calibration_result


def _r_aad():
    from pricebook.curves.aad import Number
    from pricebook.curves.aad_curves import aad_bootstrap
    c = aad_bootstrap(REF, [(date(2027, 1, 1), Number(0.03))],
                      [(date(2029, 1, 1), Number(0.032))])
    return c.calibration_result


def _r_bond_curve():
    from pricebook.curves.bond_curve import BondQuote, bootstrap_curve_from_bonds
    qs = [BondQuote(date(2027, 6, 1), 0.0, 97.5, frequency=2),
          BondQuote(date(2031, 6, 1), 0.045, 97.0, frequency=2)]
    return bootstrap_curve_from_bonds(REF, qs, method="sequential").calibration_result


def _r_tenor_basis():
    from dateutil.relativedelta import relativedelta
    from pricebook.fixed_income.ibor_curve import (
        bootstrap_ibor, EURIBOR_3M_CONVENTIONS, EURIBOR_6M_CONVENTIONS,
    )
    from pricebook.fixed_income.tenor_basis import bootstrap_tenor_basis
    ois = _flat()
    ibor_3m = bootstrap_ibor(REF, EURIBOR_3M_CONVENTIONS, ois,
                             swaps=[(REF + relativedelta(years=2), 0.033),
                                    (REF + relativedelta(years=5), 0.035)])
    ibor_6m, _ = bootstrap_tenor_basis(
        REF, ibor_3m, ois,
        basis_swap_quotes=[(REF + relativedelta(years=2), 0.0005),
                           (REF + relativedelta(years=5), 0.0010)],
        long_tenor_conventions=EURIBOR_6M_CONVENTIONS,
    )
    return ibor_6m.calibration_result


def _r_credit_curve():
    from pricebook.credit.cds import bootstrap_credit_curve
    spreads = [(date(2027, 1, 1), 0.005), (date(2031, 1, 1), 0.012)]
    return bootstrap_credit_curve(REF, spreads, _flat()).calibration_result


def _r_upfronts():
    from pricebook.credit.cds_market import bootstrap_from_upfronts
    return bootstrap_from_upfronts(REF, {3: 0.01, 5: 0.0218}, 0.01, _flat()).calibration_result


def _r_sovereign():
    from pricebook.credit.sovereign_cds import bootstrap_sovereign_hazard
    return bootstrap_sovereign_hazard(REF, {1: 50, 5: 120}, _flat(), "BR").calibration_result


def _r_cpi():
    from pricebook.fixed_income.inflation import bootstrap_cpi_curve
    return bootstrap_cpi_curve(
        REF, 100.0, [(date(2027, 1, 1), 0.025), (date(2031, 1, 1), 0.03)]
    ).calibration_result


def _r_real_yield():
    from pricebook.fixed_income.inflation_bond_advanced import real_yield_curve_bootstrap
    return real_yield_curve_bootstrap(
        [98.0, 90.0], [100, 100], [0.01, 0.015], [2.0, 10.0], 100.0, 110.0
    ).calibration_result


def _r_dividend():
    from pricebook.equity.dividend_advanced import dividend_curve_bootstrap
    return dividend_curve_bootstrap(100.0, 0.03, [1.0, 5.0], [2.0, 9.0]).calibration_result


# Registry: name in COVERED -> thunk returning its calibration_result.
REGISTRY = {
    "bootstrap": _r_bootstrap,
    "bootstrap_forward_curve": _r_forward_curve,
    "bootstrap_ois": _r_ois,
    "bootstrap_spread_curve": _r_spread_curve,
    "bootstrap_rfr": _r_rfr,
    "bootstrap_ibor": _r_ibor,
    "bootstrap_basis_curve": _r_basis_curve,
    "aad_bootstrap": _r_aad,
    "bootstrap_curve_from_bonds": _r_bond_curve,
    "bootstrap_tenor_basis": _r_tenor_basis,
    "bootstrap_credit_curve": _r_credit_curve,
    "bootstrap_from_upfronts": _r_upfronts,
    "bootstrap_sovereign_hazard": _r_sovereign,
    "bootstrap_cpi_curve": _r_cpi,
    "real_yield_curve_bootstrap": _r_real_yield,
    "dividend_curve_bootstrap": _r_dividend,
}

# Covered names exercised behaviourally elsewhere (their own test file).
_BEHAVIOURAL_ELSEWHERE = {
    "bootstrap_hazard_from_bonds",
    "bootstrap_hazard_mixed",
    "bootstrap_hazard_adaptive",
}


def test_registry_covers_all_covered_names():
    """Every COVERED producer is either in the runtime registry or tested elsewhere."""
    missing = COVERED - set(REGISTRY) - _BEHAVIOURAL_ELSEWHERE
    assert not missing, f"COVERED producers with no behavioural check: {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_covered_bootstrapper_attaches_record(name):
    cr = REGISTRY[name]()
    assert cr is not None, f"{name} returned a curve with no calibration_result"
    assert cr.fit.model_class  # non-empty snake_case model class
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
