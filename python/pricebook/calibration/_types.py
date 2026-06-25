"""Calibration layer types.

A `CalibrationResult` is a composite of small value objects — each consumer
(pricing, audit, debug) reads the slice it needs; the record is not an opaque
blob:

    `CalibrationProvenance`   — where it came from (id, timestamp, snapshot)
    `CalibrationFit`          — what was fitted and how well
    `OptimiserRun`            — how the solver behaved (wraps `OptimiserSpec`)
    `CalibrationDiagnostics`  — optional structured extras

`CanonicalCalibrationResult` is the mixin a per-family result type uses to
expose one of these records.

This module sits at L0 in the empirical dependency graph (see AUDIT_PLAN.md
§1). Its only load-time pricebook dependency is `core.serialisable` (also
L0), pulled in to make the result types first-class serialisable artefacts;
that edge is acyclic — `core.serialisable` imports nothing from calibration.
The lone in-function `import pricebook` (in `CalibrationProvenance.stamp`,
to read `__version__`) is lazy and does not create a runtime cycle.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
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

    algorithm: str  # canonicalised to snake_case: "l_bfgs_b", "differential_evolution", …
    tolerance: float
    max_iterations: int
    seed: int | None = None  # required for stochastic optimisers (reproducibility)
    extra: Mapping[str, Any] = field(default_factory=dict)

    # `algorithm` is an audit dimension (you group/filter by optimiser), so it is
    # canonicalised to one queryable vocabulary the same way `model_class` is —
    # lowercase with runs of spaces/hyphens collapsed to underscores. This makes
    # "Nelder-Mead", "nelder_mead" and "L-BFGS-B" → "nelder_mead" / "l_bfgs_b".
    _ALGORITHM_RE = re.compile(r"[a-z][a-z0-9_]*")

    def __post_init__(self) -> None:
        # Any run of non-alphanumeric characters (spaces, hyphens, '+', '/', '.')
        # collapses to a single underscore, so compound / vendor names like
        # "differential_evolution+L-BFGS-B" or "brentq-per-bond" become one key.
        canon = re.sub(r"[^a-z0-9]+", "_", str(self.algorithm).lower()).strip("_")
        if not self._ALGORITHM_RE.fullmatch(canon):
            raise ValueError(
                f"algorithm must canonicalise to non-empty snake_case (audit key); "
                f"got {self.algorithm!r}"
            )
        object.__setattr__(self, "algorithm", canon)


@serialisable_convention("optimiser_run")
@dataclass(frozen=True)
class OptimiserRun:
    """The optimiser's configuration (`spec`) and its outcome — the
    "how the solver behaved" component of a calibration.

    Bundles the plan (`spec`) with what actually happened (`iterations` taken,
    whether it `converged`) so the optimiser story lives in one place rather
    than split across the parent record.
    """

    spec: OptimiserSpec
    iterations: int
    converged: bool


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
    # True when the record was reconstructed by a result's `_build` fallback
    # (hand-built instance) rather than captured at fit time — a first-class
    # flag rather than a magic `extra["record_source"]` string.
    reconstructed: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Canonicalise the sequence fields to tuples: a frozen record should
        # hold immutable sequences, and it makes a serialise→deserialise
        # round-trip exact (JSON arrays come back as lists, not tuples).
        object.__setattr__(self, "objective_history", tuple(self.objective_history))
        object.__setattr__(self, "parameter_history", tuple(self.parameter_history))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@serialisable_convention("calibration_provenance")
@dataclass(frozen=True)
class CalibrationProvenance:
    """Where a calibration record came from — the audit stamp.

    Assigned once at creation and frozen. `id` is the key every consumer
    references; `timestamp` is timezone-aware UTC (unambiguous across
    machines); `market_snapshot_id` links to the `MarketSnapshot` the quotes
    came from (`None` until set).
    """

    id: UUID
    timestamp: datetime
    code_version: str
    market_snapshot_id: UUID | None = None

    @classmethod
    def stamp(
        cls,
        *,
        market_snapshot_id: UUID | None = None,
        id: UUID | None = None,
        timestamp: datetime | None = None,
        code_version: str | None = None,
    ) -> "CalibrationProvenance":
        """Build an audit stamp, auto-filling `id` / `timestamp` / `code_version`.

        The normal way a calibrator stamps provenance: `id` is a fresh UUID,
        `timestamp` is tz-aware UTC now, `code_version` is the running pricebook
        version. Pass any of them explicitly to reproduce a stored record or
        make a test deterministic.
        """
        if code_version is None:
            import pricebook  # lazy: avoids a circular import at module load
            code_version = pricebook.__version__
        return cls(
            id=id if id is not None else uuid4(),
            timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
            code_version=code_version,
            market_snapshot_id=market_snapshot_id,
        )


@serialisable_convention("calibration_fit")
@dataclass(frozen=True)
class CalibrationFit:
    """What was fitted, and how well — the numerical result of a calibration.

    `rms_residual` / `max_residual` are derived `@property`s over `residuals`
    (single source of truth — they can never drift). `rms_residual` is
    deliberately *unweighted* regardless of `objective`/`weights`: those record
    how the optimiser combined residuals; this is a plain magnitude summary. A
    consumer wanting a weighted RMS computes it from `residuals` + `weights`.
    """

    model_class: str
    parameters: Mapping[str, float]
    residuals: Sequence[float]
    objective: ObjectiveKind = ObjectiveKind.SSE
    quotes_fitted: Sequence[str] = ()
    weights: Sequence[float] = ()

    # model_class is an audit key — a controlled snake_case vocabulary so that
    # records from one calibrator group under one stable tag.
    _MODEL_CLASS_RE = re.compile(r"[a-z][a-z0-9_]*")

    def __post_init__(self) -> None:
        # Canonicalise the sequence fields to tuples: a frozen record should
        # hold immutable sequences (same convention as CalibrationDiagnostics),
        # and it makes a serialise→deserialise round trip exact (JSON arrays
        # come back as lists, not tuples). `parameters` stays a dict.
        object.__setattr__(self, "residuals", tuple(self.residuals))
        object.__setattr__(self, "quotes_fitted", tuple(self.quotes_fitted))
        object.__setattr__(self, "weights", tuple(self.weights))

        # Contract checks — turn the per-quote conventions from "documented" to
        # "enforced at construction", so a calibrator can't ship a structurally
        # valid but inconsistent record.
        if not self._MODEL_CLASS_RE.fullmatch(self.model_class):
            raise ValueError(
                f"model_class must be non-empty snake_case (audit key); "
                f"got {self.model_class!r}"
            )
        # A fit with no targets is not a fit. An empty residual vector makes the
        # derived `rms_residual` read as 0.0 — "no data" masquerading as a
        # perfect fit. Reject it at construction (the type-level G1 guarantee).
        n = len(self.residuals)
        if n == 0:
            raise ValueError(
                "residuals must be non-empty — a CalibrationFit with no targets "
                "is not a fit (an empty residual vector reads as a false perfect)"
            )
        if self.weights and len(self.weights) != n:
            raise ValueError(
                f"weights length {len(self.weights)} must match residuals length {n} "
                f"(parallel per-quote arrays)"
            )
        # Every residual must be attributable to a quote: a record carrying
        # residuals with no `quotes_fitted` is an unlabelled magnitude no auditor
        # can trace back to an instrument. Scalar/aggregate fits pass a single
        # label (e.g. "aggregate_objective").
        if n and not self.quotes_fitted:
            raise ValueError(
                f"quotes_fitted is required when residuals are present "
                f"({n} residuals, no quotes) — label even an aggregate objective"
            )
        if self.quotes_fitted and len(self.quotes_fitted) != n:
            raise ValueError(
                f"quotes_fitted length {len(self.quotes_fitted)} must match residuals "
                f"length {n} (parallel per-quote arrays)"
            )

    @property
    def rms_residual(self) -> float:
        if not self.residuals:
            return 0.0
        return math.sqrt(sum(r * r for r in self.residuals) / len(self.residuals))

    @property
    def max_residual(self) -> float:
        return max((abs(r) for r in self.residuals), default=0.0)


@serialisable_convention("calibration_result", schema_version=3)
@dataclass(frozen=True)
class CalibrationResult:
    """A calibration record — three components plus diagnostics:

        provenance     — where it came from (`CalibrationProvenance`)
        fit            — what was fitted and how well (`CalibrationFit`)
        optimiser_run  — how the solver behaved (`OptimiserRun`)
        diagnostics    — optional structured extras (`CalibrationDiagnostics`)

    Each component is a small self-describing value object, so the record is
    graspable at a glance and each concern reads on its own. All four are set
    in one act (a calibration runs) and frozen together.

    Construct directly from the components — that *is* the interface:

        CalibrationResult(
            provenance=CalibrationProvenance.stamp(market_snapshot_id=...),
            fit=CalibrationFit(model_class=..., parameters=..., residuals=...),
            optimiser_run=OptimiserRun(spec=OptimiserSpec(...), iterations=..., converged=...),
        )

    `CalibrationProvenance.stamp()` auto-fills the id/timestamp/code_version
    boilerplate; `diagnostics` defaults to empty.
    """

    provenance: CalibrationProvenance
    fit: CalibrationFit
    optimiser_run: OptimiserRun
    diagnostics: CalibrationDiagnostics = field(default_factory=CalibrationDiagnostics)

    def to_calibration_result(self) -> "CalibrationResult":
        """A record *is* its own canonical record — satisfies `ProvenanceCarrier`."""
        return self


@runtime_checkable
class ProvenanceCarrier(Protocol):
    """The read-interface for anything that can yield a `CalibrationResult`.

    One concept — "I carry calibration provenance" — with three structural
    realisations (no shared base required, this is a `Protocol`):

        * `CalibrationResult` itself                       → returns ``self``;
        * a *curve* (or a wrapper forwarding to one) that stores its record in a
          ``calibration_result`` field                     → returns the field, or
          ``None`` if the curve was not produced by a calibration (a flat or
          loaded curve legitimately has no provenance);
        * a `CanonicalCalibrationResult` family result      → lazily builds + caches.

    The point of naming it: `PricebookDB.save_calibration` accepts any
    `ProvenanceCarrier`, so a bootstrapped curve and a model-calibration result
    are *substitutable* at the persistence call — the storage mechanism differs
    (field vs lazy build) but the way you read the record does not.
    """

    def to_calibration_result(self) -> "CalibrationResult | None": ...


class CanonicalCalibrationResult(ABC):
    """Abstract mixin for per-family calibration results that expose a canonical
    `CalibrationResult` provenance artefact.

    A subclass (a non-frozen ``@dataclass``) must:
        * declare the field ``calibration_result: CalibrationResult | None = None``;
        * implement ``_build_calibration_record() -> CalibrationResult``, mapping
          its model-specific fields onto the canonical record.

    The second requirement is enforced: ``_build_calibration_record`` is an
    ``@abstractmethod``, so a family that forgets it fails at *instantiation*
    (``TypeError: Can't instantiate abstract class``) rather than on first use.

    A calibrator may populate ``calibration_result`` eagerly (richest
    provenance — iterations, convergence, weights captured at fit time);
    otherwise ``to_calibration_result()`` builds it lazily from the instance and
    caches it. Either way a caller gets one stable record per instance.

    This is the abstraction the (deleted) ``Calibrator`` Protocol failed to be:
    it has real implementers and factors out the field/accessor scaffolding
    every family was duplicating, without dictating the model-specific mapping.
    """

    calibration_result: "CalibrationResult | None"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Enforce the field half of the contract at class-definition time
        # (the ABC enforces the method half at instantiation). A family that
        # inherits the mixin but forgets the `calibration_result` field fails
        # here — fast, with a clear message — rather than with a stray
        # AttributeError the first time `to_calibration_result()` runs.
        super().__init_subclass__(**kwargs)
        if "calibration_result" not in cls.__dict__.get("__annotations__", {}):
            raise TypeError(
                f"{cls.__name__} is a CanonicalCalibrationResult but does not "
                f"declare the required field "
                f"'calibration_result: CalibrationResult | None = None'."
            )

    def to_calibration_result(self) -> "CalibrationResult":
        """Return the canonical record — the stored one, or a lazily-built+cached one."""
        if self.calibration_result is None:
            self.calibration_result = self._build_calibration_record()
        return self.calibration_result

    @abstractmethod
    def _build_calibration_record(self) -> "CalibrationResult":
        """Map this family's fields onto a canonical `CalibrationResult`.

        Abstract — every subclass implements it; the ABC enforces this at
        instantiation, so a family that forgets it fails fast.
        """

    @property
    def calibration_id(self) -> str | None:
        """The canonical record's id once built/stored, else None (no build side-effect)."""
        return str(self.calibration_result.provenance.id) if self.calibration_result else None
