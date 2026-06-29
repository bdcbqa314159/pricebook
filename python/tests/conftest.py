"""Shared test fixtures and helpers."""

from datetime import date

# Reproducible Hypothesis profile: derandomize so a failing property always
# replays with the same inputs (no flaky CI), and drop the per-example deadline
# (numerical properties can be slow without being broken). Guarded so the suite
# still collects if Hypothesis isn't installed.
try:
    from hypothesis import settings as _hyp_settings

    _hyp_settings.register_profile("pricebook", derandomize=True, deadline=None)
    _hyp_settings.load_profile("pricebook")
except ImportError:
    pass

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

    Since `CalibrationFit` now requires `quotes_fitted` whenever residuals are
    present, this helper auto-labels them (`quote_i`) when a fixture supplies
    residuals but no quotes — keeping concise test fixtures valid without
    weakening the production contract.
    """
    if residuals and not quotes_fitted:
        quotes_fitted = [f"quote_{i}" for i in range(len(residuals))]
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
