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

Every per-family calibrator and curve bootstrapper now produces a
`CalibrationResult` (the G1 P1 migration is complete). Two builders assemble
it from one place each — `curve_calibration_record` for curve/term-structure
bootstrappers (Family A), and `model_calibration_record` for model calibrators
(SABR, Hull-White, G2++, …, Family B), the latter capturing the optimiser facts
in a `SolveReport` so convergence is recorded, never re-derived.

Public API (see `__all__`):

    the record types     — `CalibrationResult` and its components
                           (`CalibrationProvenance`, `CalibrationFit`,
                           `OptimiserRun`, `OptimiserSpec`,
                           `CalibrationDiagnostics`, `ObjectiveKind`)
    the family mixin     — `CanonicalCalibrationResult`, `ProvenanceCarrier`
    the builders         — `curve_calibration_record`, `pillar_parameters`,
                           `model_calibration_record`, `SolveReport`
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
from pricebook.calibration._solve import SolveReport
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
    "curve_calibration_record",
    "model_calibration_record",
    "pillar_parameters",
]
