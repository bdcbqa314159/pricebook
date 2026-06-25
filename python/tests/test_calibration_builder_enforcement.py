"""Phase 4 gate — records are built only through the two factories.

The capture-not-reconstruct migration (OPEN.md §0c) routed every producer through
`curve_calibration_record` (curves) or `model_calibration_record` (model
calibrators). This locks that invariant: a producer may NOT hand-roll the record
components (`CalibrationResult` / `CalibrationFit` / `OptimiserRun` /
`OptimiserSpec`). Hand-rolling is exactly how the eager/lazy duality and the
fabricated-convergence debt (G1–G9) crept in.

A new calibrator that constructs a record by hand instead of calling a builder
fails here — it must capture a `SolveReport` and go through the factory.
"""

from __future__ import annotations

import ast
import pathlib

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "pricebook"

# Only the type definitions and the two builders may construct the raw components.
_ALLOWED = {
    "calibration/_types.py",        # the dataclass definitions themselves
    "calibration/_model_record.py", # model_calibration_record (the builder)
    "calibration/_curve_record.py", # curve_calibration_record (the builder)
}
_RAW_COMPONENTS = {"CalibrationResult", "CalibrationFit", "OptimiserRun", "OptimiserSpec"}


def test_records_are_built_only_through_the_factories():
    violations: list[str] = []
    for path in PKG_ROOT.rglob("*.py"):
        rel = path.relative_to(PKG_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in _RAW_COMPONENTS):
                violations.append(f"  {rel}:{node.lineno} constructs {node.func.id}(...)")
    assert not violations, (
        "Calibration records must be assembled via curve_calibration_record / "
        "model_calibration_record (capture a SolveReport; don't hand-roll the "
        "components):\n" + "\n".join(sorted(violations))
    )
