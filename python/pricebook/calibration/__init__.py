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
)
from pricebook.calibration._curve_record import (
    curve_calibration_record,
    pillar_parameters,
)

__all__ = [
    "CalibrationDiagnostics",
    "CalibrationFit",
    "CalibrationProvenance",
    "CalibrationResult",
    "CanonicalCalibrationResult",
    "ObjectiveKind",
    "OptimiserRun",
    "OptimiserSpec",
    "curve_calibration_record",
    "pillar_parameters",
]
