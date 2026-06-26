"""Single builder for model-calibrator provenance records.

The Family-B mirror of `curve_calibration_record`. Where curve bootstrappers
attach their record to the curve, model calibrators (SABR, Hull-White, G2++, …)
expose a per-family result; this factory assembles their canonical
`CalibrationResult` from one place, so no calibrator hand-rolls the four-component
skeleton.

The optimiser facts come in as a `SolveReport` — a **required** argument the
calibrator captures from its optimiser (`SolveReport.external` / `.analytic`). A
calibrator therefore cannot omit or fabricate `converged` / `iterations` / `seed`:
it passes through what the optimiser actually reported (or `converged=None` when
no optimiser ran). One truthful build path, captured at fit time.

Cross-cutting behaviour lives here, not in each calibrator: a non-convergence
`warning` is appended automatically, so "it didn't converge" is always visible.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence
from uuid import UUID

from pricebook.calibration._solve import SolveReport
from pricebook.calibration._types import (
    CalibrationDiagnostics,
    CalibrationFit,
    CalibrationProvenance,
    CalibrationResult,
    ObjectiveKind,
    OptimiserRun,
    OptimiserSpec,
)


def model_calibration_record(
    *,
    model_class: str,
    parameters: Mapping[str, float],
    residuals: Sequence[float],
    quotes_fitted: Sequence[str],
    solve: SolveReport,
    objective: ObjectiveKind = ObjectiveKind.SSE,
    weights: Sequence[float] = (),
    market_snapshot_id: UUID | None = None,
    optimiser_extra: Mapping[str, Any] | None = None,
    diagnostics: CalibrationDiagnostics | None = None,
    reconstructed: bool = False,
) -> CalibrationResult:
    """Assemble a model calibrator's canonical `CalibrationResult`.

    The caller supplies the fit-specific data — `model_class` (snake_case audit
    key), the fitted `parameters`, the model-minus-market `residuals` with their
    parallel `quotes_fitted` (required; `CalibrationFit` rejects unlabelled
    residuals), and the `solve` report captured from the optimiser. The optimiser
    metadata (algorithm, iterations, converged, tolerance, seed) is read straight
    off `solve` — never re-derived.

    `solve.algorithm` is canonicalised to the snake_case audit vocabulary by
    `OptimiserSpec`. A non-convergence warning is appended to `diagnostics`
    automatically.
    """
    diag = diagnostics or CalibrationDiagnostics()
    # Warn only on a definite non-convergence (False) — not on None ("not
    # captured"), which is the honest state of a reconstructed record.
    if solve.converged is False and not any("converge" in w.lower() for w in diag.warnings):
        diag = dataclasses.replace(
            diag, warnings=tuple(diag.warnings) + (f"{model_class} calibration did not converge",)
        )
    if reconstructed and not diag.reconstructed:
        diag = dataclasses.replace(diag, reconstructed=True)
    return CalibrationResult(
        provenance=CalibrationProvenance.stamp(market_snapshot_id=market_snapshot_id),
        fit=CalibrationFit(
            model_class=model_class,
            parameters=dict(parameters),
            residuals=list(residuals),
            objective=objective,
            quotes_fitted=list(quotes_fitted),
            weights=list(weights),
        ),
        optimiser_run=OptimiserRun(
            spec=OptimiserSpec(
                algorithm=solve.algorithm,
                tolerance=solve.tolerance or 0.0,
                max_iterations=solve.max_iterations or solve.iterations,
                seed=solve.seed,
                extra=dict(optimiser_extra or {}),
            ),
            iterations=solve.iterations,
            converged=solve.converged,
        ),
        diagnostics=diag,
    )
