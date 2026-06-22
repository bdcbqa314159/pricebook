"""Shared test fixtures and helpers."""

from datetime import date

from pricebook.calibration import (
    CalibrationDiagnostics,
    CalibrationFit,
    CalibrationProvenance,
    CalibrationResult,
    ObjectiveKind,
    OptimiserRun,
    OptimiserSpec,
)
from pricebook.core.discount_curve import DiscountCurve
from pricebook.core.survival_curve import SurvivalCurve


def build_calibration_result(
    *,
    model_class: str,
    parameters,
    residuals,
    optimiser: OptimiserSpec,
    iterations: int,
    converged: bool,
    objective: ObjectiveKind = ObjectiveKind.SSE,
    quotes_fitted=(),
    weights=(),
    diagnostics: CalibrationDiagnostics | None = None,
    market_snapshot_id=None,
    code_version=None,
    id=None,
    timestamp=None,
) -> CalibrationResult:
    """Test-only convenience: assemble a `CalibrationResult` from flat kwargs.

    Production code constructs the three components directly; tests use this to
    keep fixtures concise. Mirrors the component-injection the producers do.
    """
    return CalibrationResult(
        provenance=CalibrationProvenance.stamp(
            market_snapshot_id=market_snapshot_id, code_version=code_version,
            id=id, timestamp=timestamp,
        ),
        fit=CalibrationFit(
            model_class=model_class, parameters=parameters, residuals=residuals,
            objective=objective, quotes_fitted=quotes_fitted, weights=weights,
        ),
        optimiser_run=OptimiserRun(spec=optimiser, iterations=iterations, converged=converged),
        diagnostics=diagnostics or CalibrationDiagnostics(),
    )


def make_flat_curve(ref: date, rate: float) -> DiscountCurve:
    """Build a flat discount curve at the given continuously compounded rate."""
    return DiscountCurve.flat(ref, rate)


def make_flat_survival(ref: date, hazard: float) -> SurvivalCurve:
    """Build a flat survival curve at the given constant hazard rate."""
    return SurvivalCurve.flat(ref, hazard)
