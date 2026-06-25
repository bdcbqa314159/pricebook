"""The ProvenanceCarrier read-interface — the one concept unifying both sides.

A curve carries its `CalibrationResult` in a field; a model-calibration result
lazily builds one via the mixin; a raw `CalibrationResult` is its own record.
All three satisfy `ProvenanceCarrier` structurally, so `save_calibration` accepts
any of them and a curve is substitutable with a calibration result at that call.

Two guarantees:

1. **Behavioural** — each realisation satisfies the Protocol, persists, and is
   substitutable; the not-a-carrier and uncalibrated-curve paths raise clearly.
2. **Structural sweep** — every class in the package that *carries* a
   `calibration_result` (field, property, or `self.x =` in init) must also expose
   `to_calibration_result()` (directly or via the `CanonicalCalibrationResult`
   base). A new carrier that forgets the accessor — silently breaking
   substitutability — fails this test.
"""

from __future__ import annotations

import ast
import pathlib
from datetime import date

import pytest

from pricebook.calibration import CalibrationResult, ProvenanceCarrier
from pricebook.core.discount_curve import DiscountCurve
from pricebook.credit.cds import bootstrap_credit_curve
from pricebook.curves.bond_curve import BondQuote, bootstrap_curve_from_bonds
from pricebook.db.db import PricebookDB
from pricebook.options.sabr import SABRCalibrationResult

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "pricebook"
REF = date(2026, 1, 1)


# ── Behavioural ─────────────────────────────────────────────────────────────

def _bootstrapped_curve():
    return bootstrap_credit_curve(
        REF, [(date(2027, 1, 1), 0.005), (date(2031, 1, 1), 0.012)],
        DiscountCurve.flat(REF, 0.03),
    )


def test_all_three_realisations_are_carriers():
    curve = _bootstrapped_curve()                                   # field carrier
    wrapper = bootstrap_curve_from_bonds(                           # forwarding wrapper
        REF, [BondQuote(date(2027, 6, 1), 0.0, 97.5, frequency=2),
              BondQuote(date(2031, 6, 1), 0.045, 97.0, frequency=2)],
        method="sequential")
    result = SABRCalibrationResult(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4, rmse=0.001,
                                   reprice_errors_bp=[1.0, -0.5], max_error_bp=1.0)  # mixin
    record = curve.to_calibration_result()                         # raw record

    for carrier in (curve, wrapper, result, record):
        assert isinstance(carrier, ProvenanceCarrier)
        assert isinstance(carrier.to_calibration_result(), CalibrationResult)
    assert record.to_calibration_result() is record               # a record is its own record


def test_curve_is_substitutable_with_its_record_at_save():
    curve = _bootstrapped_curve()
    with PricebookDB(":memory:") as db:
        via_record = db.save_calibration(curve.to_calibration_result())
        via_curve = db.save_calibration(curve)   # the curve itself — substitutable
        assert via_curve == via_record           # same id, idempotent


def test_uncalibrated_curve_is_a_carrier_but_yields_none():
    flat = DiscountCurve.flat(REF, 0.03)
    assert isinstance(flat, ProvenanceCarrier)       # structurally a carrier
    assert flat.to_calibration_result() is None      # …but carries nothing
    with PricebookDB(":memory:") as db:
        with pytest.raises(TypeError, match="None"):
            db.save_calibration(flat)


def test_non_carrier_is_rejected():
    with PricebookDB(":memory:") as db:
        with pytest.raises(TypeError, match="ProvenanceCarrier"):
            db.save_calibration(42)


# ── Structural sweep ────────────────────────────────────────────────────────

def _carries_calibration_result(node: ast.ClassDef) -> bool:
    # annotated field or a `calibration_result` property in the class body
    for n in node.body:
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) \
                and n.target.id == "calibration_result":
            return True
        if isinstance(n, ast.FunctionDef) and n.name == "calibration_result":
            return True
    # `self.calibration_result = ...` anywhere in the class (e.g. __init__)
    for n in ast.walk(node):
        if isinstance(n, ast.Attribute) and n.attr == "calibration_result" \
                and isinstance(n.ctx, ast.Store):
            return True
    return False


def _exposes_accessor(node: ast.ClassDef) -> bool:
    if any(isinstance(n, ast.FunctionDef) and n.name == "to_calibration_result"
           for n in node.body):
        return True
    base_names = [b.id for b in node.bases if isinstance(b, ast.Name)]
    base_names += [b.attr for b in node.bases if isinstance(b, ast.Attribute)]
    return "CanonicalCalibrationResult" in base_names


def test_every_carrier_class_exposes_the_accessor():
    """Any class carrying calibration_result must also expose to_calibration_result."""
    violations: list[str] = []
    for path in PKG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _carries_calibration_result(node) \
                    and not _exposes_accessor(node):
                violations.append(f"{node.name}  @ {path.relative_to(PKG_ROOT)}")
    assert not violations, (
        "Classes that carry calibration_result but do not expose "
        "to_calibration_result() (add the one-line accessor or inherit "
        "CanonicalCalibrationResult):\n" + "\n".join(f"  {v}" for v in sorted(violations))
    )
