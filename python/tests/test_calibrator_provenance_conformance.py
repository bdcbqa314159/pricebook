"""Conformance gate for the calibrator (mixin) side of the provenance system.

Counterpart to test_bootstrapper_provenance_conformance.py. Where curves carry
their own ``calibration_result``, *model* calibrators expose theirs through the
``CanonicalCalibrationResult`` ABC mixin: each family-result subclass declares a
``calibration_result`` field and implements ``_build_calibration_record()``; the
mixin provides ``to_calibration_result()`` (lazy build + cache) and
``calibration_id``.

Three layers:

1. **Discovery guard** — AST-scan the package for classes that subclass
   ``CanonicalCalibrationResult`` and assert each is classified COVERED (or
   ALLOWLISTED). A new family-result nobody classified fails this test.
2. **Structural contract** — for each COVERED subclass assert it actually
   satisfies the mixin contract: a ``calibration_result`` dataclass field and an
   own override of ``_build_calibration_record`` (not the ABC's abstract one).
3. **Behavioural gate** — construct each cheaply-buildable family result and
   assert ``to_calibration_result()`` yields a valid, DB-round-tripping record.
   Heavy results (carrying live model / curve / leverage objects) are exercised
   in their own dedicated tests, listed in ``_BEHAVIOURAL_ELSEWHERE``.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import pathlib

import pytest

from pricebook.calibration import (
    CalibrationResult,
    CanonicalCalibrationResult,
    ProvenanceCarrier,
)
from pricebook.db.db import PricebookDB

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "pricebook"

# name -> "module:ClassName" for every CanonicalCalibrationResult subclass.
CLASSES = {
    "SABRCalibrationResult": "pricebook.options.sabr",
    "JointCalibrationResult": "pricebook.credit.joint_equity_credit",
    "HazardBootstrapResult": "pricebook.credit.bond_hazard_bootstrap",
    "ParticleCalibrationResult": "pricebook.fx.fx_slv_calibration",
    "HWCalibrationResult": "pricebook.models.hw_calibration",
    "JumpCalibrationResult": "pricebook.models.jump_calibration",
    "LMMCalibrationResult": "pricebook.models.lmm_calibration",
    "DispersionCalibrationResult": "pricebook.models.stochastic_correlation",
    "G2PPCalibrationResult": "pricebook.models.g2pp_calibration",
    "RebonatoLMMCalibrationResult": "pricebook.models.lmm_advanced",
    "JYCalibrationResult": "pricebook.fixed_income.jarrow_yildirim",
    "DividendCalibrationResult": "pricebook.equity.dividend_calibration",
    "MultiCurveResult": "pricebook.curves.multicurve_solver",
}

COVERED = set(CLASSES)
ALLOWLIST: dict[str, str] = {}  # no deliberate exclusions on the calibrator side

# COVERED results exercised behaviourally in their own test files (they carry a
# live model / curve / leverage object that is expensive to fabricate here).
_BEHAVIOURAL_ELSEWHERE = {
    "HazardBootstrapResult": "test_bond_hazard_calibration_result.py",
    "HWCalibrationResult": "test_calibration_result_g2pp_hw.py",
    "G2PPCalibrationResult": "test_calibration_result_g2pp_hw.py",
    "JYCalibrationResult": "test_jarrow_yildirim.py",
    "ParticleCalibrationResult": "test_fx_slv_calibration.py",
}


def _resolve(name: str):
    mod = importlib.import_module(CLASSES[name])
    return getattr(mod, name)


def _discover_subclasses() -> dict[str, str]:
    """AST-find class names whose bases mention CanonicalCalibrationResult."""
    found: dict[str, str] = {}
    for path in PKG_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    base_names.append(b.id)
                elif isinstance(b, ast.Attribute):
                    base_names.append(b.attr)
            if "CanonicalCalibrationResult" in base_names:
                found[node.name] = str(path.relative_to(PKG_ROOT))
    return found


# ── Layer 1: discovery ──────────────────────────────────────────────────────

def test_every_calibrator_subclass_is_classified():
    discovered = _discover_subclasses()
    classified = COVERED | set(ALLOWLIST)
    unclassified = {n: f for n, f in discovered.items() if n not in classified}
    assert not unclassified, (
        "Unclassified CanonicalCalibrationResult subclasses (add to CLASSES/COVERED "
        "or ALLOWLIST):\n" + "\n".join(f"  {n}  @ {f}" for n, f in sorted(unclassified.items()))
    )


def test_classified_set_matches_discovery():
    discovered = set(_discover_subclasses())
    stale = (COVERED | set(ALLOWLIST)) - discovered
    assert not stale, f"Classified names no longer found in the package: {sorted(stale)}"


# ── Layer 2: structural contract ────────────────────────────────────────────

@pytest.mark.parametrize("name", sorted(COVERED))
def test_subclass_satisfies_mixin_contract(name):
    cls = _resolve(name)
    assert issubclass(cls, CanonicalCalibrationResult)
    # Declares the calibration_result field (mixin __init_subclass__ enforces it;
    # assert explicitly so a regression is legible).
    fields = {f.name for f in dataclasses.fields(cls)}
    assert "calibration_result" in fields, f"{name} missing calibration_result field"
    # Overrides _build_calibration_record (not the ABC's abstract version).
    assert "_build_calibration_record" in cls.__dict__ or any(
        "_build_calibration_record" in base.__dict__
        for base in cls.__mro__[1:-1]  # below the subclass, above ABC/object
        if base is not CanonicalCalibrationResult
    ), f"{name} does not override _build_calibration_record"
    assert getattr(cls._build_calibration_record, "__isabstractmethod__", False) is False
    # Must be non-frozen: the mixin's `to_calibration_result()` lazy-caches by
    # mutating `self.calibration_result`. A frozen subclass passes class creation
    # (`__init_subclass__` runs before `@dataclass`) then raises FrozenInstanceError
    # on first use — so the guard lives here, where every subclass is checked.
    params = getattr(cls, "__dataclass_params__", None)
    assert params is not None and not params.frozen, (
        f"{name} must be a non-frozen @dataclass — the mixin lazy-caches the record"
    )


# ── Layer 3: behavioural ────────────────────────────────────────────────────

def _b_sabr():
    cls = _resolve("SABRCalibrationResult")
    return cls(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4, rmse=0.001,
               reprice_errors_bp=[1.0, -2.0], max_error_bp=2.0)


def _b_joint():
    cls = _resolve("JointCalibrationResult")
    return cls(asset_vol=0.2, leverage=0.5, recovery_mean=0.4, recovery_vol=0.1,
               equity_vol_model=0.30, equity_vol_market=0.31,
               cds_spread_model_bp=120.0, cds_spread_market_bp=118.0,
               equity_vol_error_pct=1.0, cds_spread_error_bp=2.0, fit_quality=0.001)


def _b_jump():
    cls = _resolve("JumpCalibrationResult")
    return cls(model_type="merton", params={"lambda": 0.1, "mu_j": -0.05, "sigma_j": 0.1},
               rmse_vol=0.005, market_vols=[0.20, 0.22], model_vols=[0.21, 0.225],
               strikes=[90.0, 100.0], n_params=3)


def _b_lmm():
    cls = _resolve("LMMCalibrationResult")
    return cls(calibrated_vols=[0.20, 0.18], target_swaption_vols={(1, 1): 0.20},
               fitted_swaption_vols={(1, 1): 0.21}, rmse=0.001)


def _b_dispersion():
    cls = _resolve("DispersionCalibrationResult")
    return cls(kappa=1.0, theta=0.04, sigma=0.3, residual=0.001,
               index_variance_model=0.040, index_variance_target=0.041)


def _b_rebonato():
    import numpy as np
    cls = _resolve("RebonatoLMMCalibrationResult")
    return cls(vols=np.array([0.20, 0.18, 0.16]), residual=0.001,
               n_swaptions=5, method="rebonato")


def _b_dividend():
    from pricebook.equity.dividend_advanced import dividend_curve_bootstrap
    cls = _resolve("DividendCalibrationResult")
    curve = dividend_curve_bootstrap(100.0, 0.03, [1.0, 2.0], [2.0, 3.8])
    return cls(curve=curve, rmse=0.01, fitted_futures=[2.0, 3.8],
               market_futures=[2.01, 3.79], method="bootstrap")


def _b_multicurve():
    from datetime import date
    from pricebook.core.discount_curve import DiscountCurve
    cls = _resolve("MultiCurveResult")
    ref = date(2026, 1, 1)
    return cls(ois_curve=DiscountCurve.flat(ref, 0.03),
               projection_curve=DiscountCurve.flat(ref, 0.032),
               residual=1e-10, n_iterations=4, jacobian=None, converged=True)


BUILDERS = {
    "SABRCalibrationResult": _b_sabr,
    "JointCalibrationResult": _b_joint,
    "JumpCalibrationResult": _b_jump,
    "LMMCalibrationResult": _b_lmm,
    "DispersionCalibrationResult": _b_dispersion,
    "RebonatoLMMCalibrationResult": _b_rebonato,
    "DividendCalibrationResult": _b_dividend,
    "MultiCurveResult": _b_multicurve,
}


def test_every_covered_result_is_behaviourally_checked():
    """Each COVERED result is either built here or exercised in a named test."""
    unchecked = COVERED - set(BUILDERS) - set(_BEHAVIOURAL_ELSEWHERE)
    assert not unchecked, f"COVERED results with no behavioural check: {sorted(unchecked)}"


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_calibrator_builds_valid_record(name):
    obj = BUILDERS[name]()
    assert isinstance(obj, ProvenanceCarrier)  # satisfies the read interface
    cr = obj.to_calibration_result()
    assert isinstance(cr, CalibrationResult)
    assert cr.fit.model_class  # non-empty snake_case key (CalibrationFit validates)
    # Lazy build is cached / idempotent.
    assert obj.to_calibration_result() is cr
    with PricebookDB(":memory:") as db:
        cid = db.save_calibration(cr)
        assert db.load_calibration(cid) == cr
        # Substitutable: the carrier itself can be saved (same id, idempotent).
        assert db.save_calibration(obj) == cid
