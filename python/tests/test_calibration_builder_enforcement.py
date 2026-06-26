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


def _local_aliases(tree: ast.Module) -> set[str]:
    """Local names that bind a raw component, incl. `import ... as X` aliases —
    so `from pricebook.calibration._types import OptimiserSpec as Spec` makes
    `Spec` a tracked name, closing the aliased-import evasion."""
    names = set(_RAW_COMPONENTS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name in _RAW_COMPONENTS:
                    names.add(a.asname or a.name)
    return names


def test_records_are_built_only_through_the_factories():
    violations: list[str] = []
    for path in PKG_ROOT.rglob("*.py"):
        rel = path.relative_to(PKG_ROOT).as_posix()
        if rel in _ALLOWED:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        local = _local_aliases(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            # bare/aliased name: `OptimiserSpec(...)` / `Spec(...)`
            hit = (isinstance(f, ast.Name) and f.id in local)
            # attribute access: `t.OptimiserSpec(...)` / `_types.CalibrationFit(...)`
            hit = hit or (isinstance(f, ast.Attribute) and f.attr in _RAW_COMPONENTS)
            if hit:
                name = f.id if isinstance(f, ast.Name) else f.attr
                violations.append(f"  {rel}:{node.lineno} constructs {name}(...)")
    assert not violations, (
        "Calibration records must be assembled via curve_calibration_record / "
        "model_calibration_record (capture a SolveReport; don't hand-roll the "
        "components):\n" + "\n".join(sorted(violations))
    )
