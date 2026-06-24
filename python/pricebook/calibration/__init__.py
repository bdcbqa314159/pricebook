"""Pricebook calibration layer (DESIGN.md §2.2, §5.1, §6 G1 P1).

Calibration is its own layer — separate from models, curves, instruments —
because a calibration takes market quotes (which the model layer does not
know about), produces parameters (which the model needs), and chooses which
quotes to fit (a business decision).

Every calibration produces a `CalibrationResult` — a first-class artefact
carrying parameters, residuals, convergence story, and provenance. Pricing
that uses calibrated parameters carries the `CalibrationResult.id` so the
audit chain `(price) -> (calibration) -> (market snapshot, code version)`
is reconstructable.

This module defines the *types* for the calibration layer. Per-family
calibrators (hazard bootstrap, G2++, Hull-White, SABR, LMM, curve
bootstrap, etc.) migrate to producing `CalibrationResult` in subsequent
slices of G1 P1.

Public API:

    from pricebook.calibration import (
        CalibrationResult,
        OptimiserSpec,
        CalibrationDiagnostics,
        ObjectiveKind,
    )
"""

from pricebook.calibration._types import (
    CalibrationDiagnostics,
    CalibrationFit,
    CalibrationProvenance,
    CalibrationResult,
    CanonicalCalibrationResult,
    ObjectiveKind,
    OptimiserRun,
    OptimiserSpec,
    ProvenanceCarrier,
)
from pricebook.calibration._curve_record import (
    curve_calibration_record,
    pillar_parameters,
)
from pricebook.calibration._solve import (
    SolveReport,
    brentq_solve,
    global_local_solve,
    least_squares_solve,
    minimize_solve,
    particle_solve,
)
from pricebook.calibration._model_record import model_calibration_record

__all__ = [
    "CalibrationDiagnostics",
    "CalibrationFit",
    "CalibrationProvenance",
    "CalibrationResult",
    "CanonicalCalibrationResult",
    "ObjectiveKind",
    "OptimiserRun",
    "OptimiserSpec",
    "ProvenanceCarrier",
    "SolveReport",
    "brentq_solve",
    "curve_calibration_record",
    "global_local_solve",
    "least_squares_solve",
    "minimize_solve",
    "model_calibration_record",
    "particle_solve",
    "pillar_parameters",
]
