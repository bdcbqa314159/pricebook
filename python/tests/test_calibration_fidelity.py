"""Fidelity gate — the record must tell the truth, not merely exist.

The conformance gates prove a calibration record is *present* and round-trips.
This gate proves it is *honest*, closing the G1/G6/G7 design gaps:

* **G1** — no record may carry an empty residual vector. An empty `residuals`
  makes `rms_residual` read as `0.0` — "no data" masquerading as a perfect fit.
* **G6** — every `algorithm` is a canonical snake_case audit key (never the old
  `"unknown"` sentinel), so records group cleanly by optimiser.
* **G7** — `model_class` is globally unique across the ~30 producer families, so
  `list_calibrations(model_class=…)` never silently merges two of them.

It draws every real record from the two conformance registries (curve
bootstrappers + model calibrators), so it covers the whole producer surface.
"""

from __future__ import annotations

import re

import pytest

from tests.test_bootstrapper_provenance_conformance import REGISTRY as _BOOT
from tests.test_calibrator_provenance_conformance import BUILDERS as _CALIB

_ALGO_RE = re.compile(r"[a-z][a-z0-9_]*")


def _all_records() -> dict[str, object]:
    """family-label -> CalibrationResult, across both producer registries."""
    records: dict[str, object] = {}
    for name, thunk in _BOOT.items():
        records[f"bootstrap:{name}"] = thunk()
    for name, builder in _CALIB.items():
        records[f"calibrator:{name}"] = builder().to_calibration_result()
    return records


RECORDS = _all_records()


@pytest.mark.parametrize("label", sorted(RECORDS))
def test_record_residuals_are_nonempty_and_attributable(label):
    """G1 — a record must carry real, labelled residuals (no false-perfect 0)."""
    rec = RECORDS[label]
    n = len(rec.fit.residuals)
    assert n > 0, f"{label}: empty residual vector — rms_residual would read as a false 0.0"
    assert len(rec.fit.quotes_fitted) == n, f"{label}: residuals not 1:1 with quotes"


@pytest.mark.parametrize("label", sorted(RECORDS))
def test_record_algorithm_is_canonical(label):
    """G6 — algorithm is a non-empty snake_case key, never the 'unknown' sentinel."""
    algo = RECORDS[label].optimiser_run.spec.algorithm
    assert _ALGO_RE.fullmatch(algo), f"{label}: non-canonical algorithm {algo!r}"
    assert algo != "unknown", f"{label}: placeholder algorithm 'unknown'"


@pytest.mark.parametrize("label", sorted(RECORDS))
def test_converged_record_residuals_are_consistent(label):
    """G1 — a record that claims converged with rms 0 really is exact (max ~0)."""
    rec = RECORDS[label]
    if rec.optimiser_run.converged and rec.fit.rms_residual == 0.0:
        assert rec.fit.max_residual == 0.0, (
            f"{label}: claims converged + rms 0 but max_residual is "
            f"{rec.fit.max_residual} — inconsistent fit story"
        )


# model_class keys legitimately shared because one producer *delegates* to
# another (a single underlying calibration exposed via two entry points).
_DELEGATION_GROUPS = {
    "discount_curve_bootstrap": {"bootstrap:bootstrap", "bootstrap:bootstrap_rfr"},
    "projection_curve_bootstrap": {
        "bootstrap:bootstrap_forward_curve", "bootstrap:bootstrap_ibor",
    },
}


def test_model_class_is_globally_unique():
    """G7 — no two *unrelated* families share a model_class audit key.

    A key may map to several labels only when they form a known delegation group
    (same calibration, different entry point); any other sharing is a collision
    that would silently merge two families under `list_calibrations(model_class=…)`.
    """
    by_mc: dict[str, set[str]] = {}
    for label, rec in RECORDS.items():
        by_mc.setdefault(rec.fit.model_class, set()).add(label)
    collisions = [
        f"{mc!r}: {sorted(labels)}"
        for mc, labels in by_mc.items()
        if len(labels) > 1 and labels != _DELEGATION_GROUPS.get(mc)
    ]
    assert not collisions, "model_class collisions across families:\n" + "\n".join(collisions)
