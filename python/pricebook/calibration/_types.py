"""Calibration layer types — `CalibrationResult`, `OptimiserSpec`, ...

See `DESIGN.md` §3.3 for the rationale on the shape of `CalibrationResult`.
The short version: every consumer (pricing, audit, debug, regret analysis)
needs a different slice of the calibration story; expose all the slices,
not an opaque blob.

This module sits at L0 in the empirical dependency graph (see AUDIT_PLAN.md
§1). Its only load-time pricebook dependency is `core.serialisable` (also
L0), pulled in to make the result types first-class serialisable artefacts;
that edge is acyclic — `core.serialisable` imports nothing from calibration.
The lone in-function `import pricebook` in the factory below is lazy and
does not create a runtime cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from pricebook.core.serialisable import serialisable_convention


class ObjectiveKind(str, Enum):
    """How the calibration objective combines per-quote residuals.

    The objective is what the optimiser actually minimises. It is recorded
    on the `CalibrationResult` so a later debugger knows whether a low
    `rms_residual` reflects an unweighted SSE, a weighted SSE, or a
    robust loss (Huber, L1) that suppresses outliers.
    """

    SSE = "sse"
    WEIGHTED_SSE = "weighted_sse"
    RMSE = "rmse"
    MAX_ERROR = "max_error"
    L1 = "l1"
    HUBER = "huber"


@serialisable_convention("optimiser_spec")
@dataclass(frozen=True)
class OptimiserSpec:
    """Description of the optimiser used in the calibration.

    Carried on the result so that a re-run with the same inputs reproduces
    the same parameters bit-for-bit (assuming a deterministic seed when
    the algorithm is stochastic).
    """

    algorithm: str  # "L-BFGS-B", "differential_evolution", "least_squares", ...
    tolerance: float
    max_iterations: int
    seed: int | None = None  # required for stochastic optimisers
    extra: Mapping[str, Any] = field(default_factory=dict)


@serialisable_convention("calibration_diagnostics")
@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Auxiliary structured diagnostics for a calibration run.

    Each field is optional — calibrators populate what they have access to.
    A calibration that just records the objective history (one float per
    iteration) is already enormously useful for post-mortem debugging.
    """

    objective_history: Sequence[float] = ()
    parameter_history: Sequence[Mapping[str, float]] = ()
    timing_ms: float | None = None
    warnings: Sequence[str] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Canonicalise the sequence fields to tuples: a frozen record should
        # hold immutable sequences, and it makes a serialise→deserialise
        # round-trip exact (JSON arrays come back as lists, not tuples).
        object.__setattr__(self, "objective_history", tuple(self.objective_history))
        object.__setattr__(self, "parameter_history", tuple(self.parameter_history))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@serialisable_convention("calibration_result")
@dataclass(frozen=True)
class CalibrationResult:
    """Result of a calibration run.

    A first-class artefact — versioned, serialisable in subsequent slices,
    intended to be stored. Every consumer of calibrated parameters
    references this result by `id` to make the provenance chain auditable.

    Field categories:
        identity      — `id`, `timestamp`, `code_version`
        what was fit  — `model_class`, `parameters`, `quotes_fitted`,
                        `weights`, `objective`
        fit quality   — `residuals`, `rms_residual`, `max_residual`
        optimiser     — `optimiser`, `iterations`, `converged`
        extras        — `diagnostics`, `market_snapshot_id`

    `market_snapshot_id` is `None` until G1 P2 lands a `MarketSnapshot`
    type at L1 (DESIGN.md §5.1 A2). Existing calibrators may set it later
    when that piece exists.

    Construct manually only if you need to override an auto-derived
    invariant (e.g. you compute residuals from a non-standard space).
    Use the `CalibrationResult.new(...)` factory in the normal case.
    """

    # Identity (every run has a unique id)
    id: UUID
    timestamp: datetime
    code_version: str

    # What was calibrated
    model_class: str
    parameters: Mapping[str, float]

    # What was fit
    quotes_fitted: Sequence[str]
    weights: Sequence[float]
    objective: ObjectiveKind

    # Fit quality
    residuals: Sequence[float]
    rms_residual: float
    max_residual: float

    # Optimiser story
    iterations: int
    optimiser: OptimiserSpec
    converged: bool

    # Optional structured diagnostics
    diagnostics: CalibrationDiagnostics = field(default_factory=CalibrationDiagnostics)

    # Links to other artefacts (filled in as the layer grows)
    market_snapshot_id: UUID | None = None

    @classmethod
    def new(
        cls,
        *,
        model_class: str,
        parameters: Mapping[str, float],
        residuals: Sequence[float],
        optimiser: OptimiserSpec,
        iterations: int,
        converged: bool,
        objective: ObjectiveKind = ObjectiveKind.SSE,
        quotes_fitted: Sequence[str] = (),
        weights: Sequence[float] = (),
        diagnostics: CalibrationDiagnostics | None = None,
        market_snapshot_id: UUID | None = None,
        code_version: str | None = None,
        id: UUID | None = None,
        timestamp: datetime | None = None,
    ) -> "CalibrationResult":
        """Factory: generate `id` and `timestamp`, derive RMSE / max-error.

        Keyword-only on purpose — calibrators historically have many
        positional parameters and this constructor mustn't continue that.

        `id` and `timestamp` are auto-generated when omitted (the normal
        case). Pass them explicitly to reproduce a stored result or to make
        a test deterministic. The auto-stamped `timestamp` is timezone-aware
        UTC so the provenance record is unambiguous across machines.
        """
        residuals_list = list(residuals)
        if residuals_list:
            squared = [r * r for r in residuals_list]
            rms = math.sqrt(sum(squared) / len(squared))
            mx = max(abs(r) for r in residuals_list)
        else:
            rms = 0.0
            mx = 0.0

        if weights:
            weights_list = list(weights)
        else:
            weights_list = [1.0] * len(residuals_list)

        if code_version is None:
            import pricebook  # lazy: avoids a circular import at module load
            code_version = pricebook.__version__

        return cls(
            id=id if id is not None else uuid4(),
            timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
            code_version=code_version,
            model_class=model_class,
            parameters=dict(parameters),
            quotes_fitted=list(quotes_fitted),
            weights=weights_list,
            objective=objective,
            residuals=residuals_list,
            rms_residual=rms,
            max_residual=mx,
            iterations=iterations,
            optimiser=optimiser,
            converged=converged,
            diagnostics=diagnostics or CalibrationDiagnostics(),
            market_snapshot_id=market_snapshot_id,
        )


class CanonicalCalibrationResult:
    """Mixin for per-family calibration results that expose a canonical
    `CalibrationResult` provenance artefact.

    A subclass (a non-frozen ``@dataclass``) must:
        * declare the field ``calibration_result: CalibrationResult | None = None``;
        * implement ``_build_calibration_record() -> CalibrationResult``, mapping
          its model-specific fields onto the canonical record.

    A calibrator may populate ``calibration_result`` eagerly (richest
    provenance — iterations, convergence, weights captured at fit time);
    otherwise ``to_calibration_result()`` builds it lazily from the instance and
    caches it. Either way a caller gets one stable record per instance.

    This is the abstraction the (deleted) ``Calibrator`` Protocol failed to be:
    it has real implementers and factors out the field/accessor scaffolding
    every family was duplicating, without dictating the model-specific mapping.
    """

    calibration_result: "CalibrationResult | None"

    def to_calibration_result(self) -> "CalibrationResult":
        """Return the canonical record — the stored one, or a lazily-built+cached one."""
        if self.calibration_result is None:
            self.calibration_result = self._build_calibration_record()
        return self.calibration_result

    def _build_calibration_record(self) -> "CalibrationResult":
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_calibration_record()"
        )

    @property
    def calibration_id(self) -> str | None:
        """The canonical record's id once built/stored, else None (no build side-effect)."""
        return str(self.calibration_result.id) if self.calibration_result else None
