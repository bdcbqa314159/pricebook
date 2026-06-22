"""Shared factory for curve/term-structure bootstrapper provenance records.

Every curve or survival bootstrapper computes its own model-minus-market
`residuals` (instrument-specific), then attaches a canonical `CalibrationResult`
to the curve it returns (`curve.calibration_result`). The *assembly* of that
record — provenance stamp, the `CalibrationFit`, the optimiser run, diagnostics —
is identical across bootstrappers; `curve_calibration_record` factors it out so
the record shape is uniform and no bootstrapper hand-rolls it.

This lives in the calibration layer (L0) so every producer — `curves`,
`fixed_income`, `credit`, `equity` — can import it without pulling in any
concrete curve type.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence
from uuid import UUID

from pricebook.calibration._types import (
    CalibrationDiagnostics,
    CalibrationFit,
    CalibrationProvenance,
    CalibrationResult,
    ObjectiveKind,
    OptimiserRun,
    OptimiserSpec,
)


def pillar_parameters(
    pillar_dates: Sequence[date],
    pillar_values: Sequence[float],
    *,
    label: str = "df",
) -> dict[str, float]:
    """`{label(2026-06-22): value, ...}` — the calibrated per-pillar quantity.

    `label` names what the pillar value is: ``df`` (discount factor),
    ``survival``, ``hazard``, ``real_df``, ``cum_div``, …
    """
    return {
        f"{label}({d.isoformat()})": float(v)
        for d, v in zip(pillar_dates, pillar_values)
    }


def curve_calibration_record(
    *,
    model_class: str,
    parameters: Mapping[str, float],
    residuals: Sequence[float],
    quotes_fitted: Sequence[str],
    algorithm: str,
    iterations: int,
    converged: bool = True,
    tolerance: float = 0.0,
    objective: ObjectiveKind = ObjectiveKind.SSE,
    market_snapshot_id: UUID | None = None,
    optimiser_extra: Mapping[str, Any] | None = None,
    diagnostics_extra: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Assemble the canonical `CalibrationResult` for a bootstrapped curve.

    The caller supplies the curve-specific data — `model_class` (snake_case audit
    key), the pillar `parameters` (see `pillar_parameters`), the model-minus-market
    `residuals` with their parallel `quotes_fitted`, and the optimiser metadata.
    This builds the four components uniformly. `CalibrationFit` enforces the
    snake_case / length-agreement conventions at construction.
    """
    return CalibrationResult(
        provenance=CalibrationProvenance.stamp(market_snapshot_id=market_snapshot_id),
        fit=CalibrationFit(
            model_class=model_class,
            parameters=dict(parameters),
            residuals=list(residuals),
            objective=objective,
            quotes_fitted=list(quotes_fitted),
        ),
        optimiser_run=OptimiserRun(
            spec=OptimiserSpec(
                algorithm=algorithm,
                tolerance=tolerance,
                max_iterations=iterations,
                extra=dict(optimiser_extra or {}),
            ),
            iterations=iterations,
            converged=converged,
        ),
        diagnostics=CalibrationDiagnostics(extra=dict(diagnostics_extra or {})),
    )
