# Calibration provenance — architecture

How every calibration in the library — bootstrapping a curve, fitting a model to
a vol surface, solving a coupled OIS/projection system — produces an *auditable,
reproducible, persistable* record, uniformly.

This document is the map. The authoritative detail lives in the docstrings of
`_types.py`, `_solve.py`, `_curve_record.py` and `_model_record.py`; this explains
how the pieces fit and *why* they are shaped the way they are.

---

## 1. The problem

Every calibration yields *parameters* plus a story: which market quotes went in,
how well they were fit, which optimiser ran, did it converge, against what market
snapshot. Historically that story was computed and dropped. The goal of this
layer is to make it a **first-class artefact** attached to everything that
calibrates, so any result can be audited and reproduced after the fact — and to
do so **uniformly**, so a new pricer's calibrator inherits the same shape for
free instead of inventing its own.

---

## 2. The core record — `CalibrationResult`

A frozen composite of four small, self-describing value objects. Deliberately
*not* an opaque blob: each consumer (pricing, audit, debugger) reads only the
slice it needs.

```
CalibrationResult  (frozen)
├── provenance     : CalibrationProvenance   — id (UUID), timestamp (tz-aware UTC),
│                                               code_version, market_snapshot_id
├── fit            : CalibrationFit           — model_class (snake_case audit key),
│                                               parameters{}, residuals[], quotes_fitted[],
│                                               objective (ObjectiveKind);
│                                               rms_residual / max_residual properties
├── optimiser_run  : OptimiserRun             — wraps OptimiserSpec(algorithm, tolerance,
│                                               max_iterations, seed, extra{}); iterations, converged (bool|None)
└── diagnostics    : CalibrationDiagnostics   — all optional: objective_history,
                                                parameter_history, timing_ms, warnings,
                                                reconstructed (bool), extra{}
```

Two invariants are enforced **in the constructor**, not by convention:

- **`CalibrationFit`** rejects an empty residual vector, a `model_class` that is
  not snake_case (regex `[a-z][a-z0-9_]*`), residuals with no `quotes_fitted`,
  and parallel arrays that disagree in length. A malformed or no-target record
  cannot be built.
- **`CalibrationProvenance.stamp()`** auto-fills id / timestamp (tz-aware UTC) /
  code_version, so producers never hand-roll that boilerplate.

All four components are set in one act and frozen together — "a calibration ran"
is a single immutable event.

**`parameters` holds fitted *scalars*; surfaces are fingerprinted.** When a
calibration fits something too large for the scalar `parameters` dict — FX-SLV
fits a whole **leverage surface** — the surface is not stuffed into the record.
Instead `_surface_digest(values)` puts a **shape + sha256 fingerprint** into
`diagnostics.extra` (`leverage_surface_sha256` / `..._shape`). This is the
designer's `ParamDigest` idea, realised in the open `extra` bag rather than by
widening `parameters` to a per-family `float | ParamDigest` union. The record
stays small and queryable; a re-run can be *verified* to have produced the same
surface; and the digest keys the surface in a side store if the blob is kept.

---

## 3. The read interface — `ProvenanceCarrier`

The system has **one concept** — "I carry calibration provenance" — with three
structural realisations. It is named by a `@runtime_checkable` `Protocol`:

```python
@runtime_checkable
class ProvenanceCarrier(Protocol):
    def to_calibration_result(self) -> CalibrationResult | None: ...
```

The three realisations (no shared base — this is structural typing):

| Realisation | `to_calibration_result()` returns |
|---|---|
| `CalibrationResult` itself | `self` |
| a **curve** (or a wrapper forwarding to one) carrying a `calibration_result` field | the stored record — or `None` if it was never calibrated (a flat / loaded / bumped curve legitimately has no provenance) |
| a **`CanonicalCalibrationResult`** family result | a lazily-built, cached record |

The point of naming it: a bootstrapped curve and a model-calibration result are
**substitutable** wherever provenance is read — most importantly at
`PricebookDB.save_calibration`, which accepts any `ProvenanceCarrier`. The
*storage mechanism* differs (field vs lazy build); the *read interface* does not.

---

## 4. The two storage mechanisms — and why two

The split is forced by **what the calibrating function returns**, not by two
design philosophies.

### Pattern A — curve-carries-provenance (bootstrappers)

A bootstrapper's output *is a curve* — and the curve is a shared, pre-existing
value type (`DiscountCurve`, `SurvivalCurve`, `AADDiscountCurve`, `CPICurve`, …)
used across the whole library. So the curve **is** the carrier: it holds a plain
nullable field, filled by the producer before returning.

```python
# in the curve's __init__
self.calibration_result: CalibrationResult | None = None   # None until a bootstrapper sets it

# the producer injects directly
curve.calibration_result = curve_calibration_record(model_class="credit_curve_bootstrap", …)
return curve
```

A nullable field is the *right* model because provenance on a curve is
**optional** — a flat curve, a curve loaded from disk, a bumped curve all exist
with no calibration. Each carrier also exposes the one-line
`to_calibration_result()` that returns the field, satisfying `ProvenanceCarrier`.

**Wrappers forward.** When a bootstrapper returns something richer than a bare
curve — `BondCurveResult`, `RFRCurveResult`, `SovereignHazardResult`, or an
`(IBORCurve, TenorBasis)` tuple — the wrapper exposes `calibration_result` as a
**property forwarding to the underlying curve**. One carrier, one record, no
duplication. (This is why `bootstrap_tenor_basis` works for free: it attaches the
record to the long-tenor projection curve and `IBORCurve` forwards.)

### Pattern B — the ABC mixin (model calibrators)

A model calibrator's output *is parameters*, returned in a **bespoke per-family
type** (`SABRCalibrationResult` = α/β/ρ/ν; `HWCalibrationResult` = a/σ + model).
There is no shared curve to hang the record on, and every family's fields differ.
So these inherit the mixin, which factors out the scaffolding while letting each
family own the *mapping*:

```python
class CanonicalCalibrationResult(ABC):
    calibration_result: CalibrationResult | None          # the field (the contract)

    def __init_subclass__(cls):                           # ① field enforced at CLASS-DEF time
        if "calibration_result" not in cls.__annotations__:
            raise TypeError(...)

    @abstractmethod
    def _build_calibration_record(self) -> CalibrationResult:   # ② mapping enforced at INSTANTIATION
        ...

    def to_calibration_result(self):                      # lazy build + cache → one stable record
        if self.calibration_result is None:
            self.calibration_result = self._build_calibration_record()
        return self.calibration_result

    @property
    def calibration_id(self): ...                         # the record's UUID once built
```

The contract is enforced **structurally at two moments**: `__init_subclass__`
rejects (at import) a subclass that forgets the field; `@abstractmethod` rejects
(at instantiation) one that forgets the mapping.

**Eager populate + reconstruct fallback.** The calibrate function populates
`calibration_result` **eagerly** at fit time with the real `SolveReport` captured
from the optimiser (§6). `to_calibration_result()` returns that stored record; it
only falls back to `_build_calibration_record()` for a *hand-built* instance,
which reconstructs from stored fields, marks `reconstructed=True`, and reports
`converged=None` (no optimiser ran — not a guess). Both paths go through the same
builder, so there is no eager/lazy *drift* — just captured vs not-captured.

> This mixin is explicitly the abstraction the deleted `Calibrator` Protocol
> failed to be: it has real implementers and removes duplicated field/accessor
> code without dictating the model-specific mapping.

### The layering, in one picture

```
ProvenanceCarrier  (Protocol — the read interface; what save_calibration accepts)
   ├── curves / wrappers          → field (Pattern A): to_calibration_result() returns it
   ├── CanonicalCalibrationResult → lazy build (Pattern B)
   └── CalibrationResult          → returns self
```

---

## 5. Why there is no `Bootstrapper` or `Calibrator` class

A natural question: calibrators have `CanonicalCalibrationResult` — why no
top-level class for bootstrappers? Three answers, all "correct as-is":

1. **Neither a bootstrapper nor a calibrator is an object.** Both are *functions*
   (`bootstrap_credit_curve(...)`, `sabr_calibrate(...)`). Only their *results*
   are objects. `CanonicalCalibrationResult` is not "the calibrator base class" —
   it is the mixin for calibrator *result types*. There is no class for either
   *process*.

2. **No dispatch site.** Nobody holds a `list[Bootstrapper]` and calls
   `.bootstrap()` polymorphically; callers invoke a specific function with
   specific market data. A process class would be ceremony with no consumer — the
   exact reason the `Calibrator` Protocol was deleted. Shared machinery (Newton
   in `curves/global_solver.py`, `brentq`, the record helper) is shared by
   **composition, not inheritance**.

3. **Curves must not go under the ABC.** Forcing every curve to implement
   `_build_calibration_record()` and pass the field-enforcing `__init_subclass__`
   would be wrong: a curve has a life outside calibration, and its provenance is
   optional. A nullable field models that; an ABC would lie.

The genuine shared concept was never the *process* — it was the **read interface
on the result**, which `ProvenanceCarrier` now names.

**When would a `Bootstrapper` class earn its place?** If a multi-currency
curve-building *service* ever needs to select and run bootstrappers
polymorphically (a registry / strategy by instrument or currency), a
`Bootstrapper` Protocol with `bootstrap(market) -> ProvenanceCarrier` would pay
for itself. Not before.

---

## 6. Capture-not-reconstruct — the solver layer + two builders

Records are **assembled only through two factories**, never hand-rolled. This is
the load-bearing rule (enforced — see §8): hand-rolling is how the eager/lazy
duality and the fabricated-convergence debt crept in historically.

**The capture layer (`_solve.py`) — `SolveReport`.** A calibration's optimiser
facts (`converged / iterations / tolerance / seed`) are *captured from the
optimiser that actually ran*, never re-derived. The capture vehicle is
`SolveReport`. In practice each calibrator wraps the result of its existing
optimiser call:

```python
SolveReport.external(algorithm=…, converged=result.success, iterations=result.nit, …)
                                       # wraps scipy / pb_minimize / differential_evolution
SolveReport.analytic()                 # closed-form: no iteration, converged=True honestly
```

`_solve.py` also ships five ready-made primitives — `minimize_solve`,
`least_squares_solve`, `global_local_solve`, `brentq_solve`, `particle_solve` —
that run the solve *and* return the `SolveReport` for you (`-> (solution,
SolveReport)`). They're a convenience for **new** calibrators; the existing 13
capture via `.external()` around their own optimiser calls, so the primitives are
currently exercised only by their tests.

`SolveReport.converged` is **tri-state `bool | None`**: the optimiser's real
verdict, or `None` = "not captured" (a reconstructed, hand-built result). It is
*never* guessed from a threshold.

**Two builders, one per family.** Each takes the captured data and assembles the
four components uniformly:

```python
# Pattern A — curves (_curve_record.py)
curve_calibration_record(*, model_class, parameters, residuals, quotes_fitted,
                         algorithm, iterations, converged=True, …) -> CalibrationResult
# Pattern B — model calibrators (_model_record.py)
model_calibration_record(*, model_class, parameters, residuals, quotes_fitted,
                         solve: SolveReport, objective=SSE, weights=(),
                         reconstructed=False, …) -> CalibrationResult
pillar_parameters(dates, values, *, label="df") -> {"df(2027-01-01)": 0.97, …}
```

`model_calibration_record` *requires* a `SolveReport`, so a calibrator can
neither omit nor invent convergence; cross-cutting behaviour (the
non-convergence warning, the `reconstructed` flag) lives here, not copy-pasted
into each calibrator. Both builders live at **L0** so every producer imports
them without pulling in a concrete curve type.

The pipeline every producer follows:

```
inputs → solver primitive → SolveReport → {curve,model}_calibration_record → CalibrationResult
                              (captured)         (the one assembler)
```

---

## 6b. The two producer structures, side by side

Both families end at the same `CalibrationResult`, read through the same
Protocol, persisted through the same sink, held by the same gates. They differ in
**what they return** and **how they get the optimiser facts** — everything
downstream is shared.

```
        FAMILY A — bootstrappers                 FAMILY B — model calibrators
        (curve is the artefact)                  (parameters are the artefact)
  ───────────────────────────────────────  ───────────────────────────────────────
  returns   a shared curve                  a bespoke result type
  carrier   field on the curve              CanonicalCalibrationResult mixin
            (+ wrapper forwarders)           (calibration_result field + _build)
  builder   curve_calibration_record        model_calibration_record
  facts     direct args                     captured via SolveReport.external
            (algorithm/iterations/converged) (wraps the optimiser's result)
  why       convergence honest by           a real optimiser ran — capture its
            construction (closed-form        verdict so it's never guessed
            exact / brentq raises / Newton)
  params    per-pillar values               fitted scalars (+ surface digest)
            (pillar_parameters)
```

**Anatomy — one of each:**

```python
# FAMILY A — bootstrap_credit_curve(...) -> SurvivalCurve
curve = SurvivalCurve(ref, pillar_dates, pillar_survivals)          # ① solve (per-pillar brentq)
residuals = _verify_credit_round_trip(curve, cds_spreads, …)       # ② model − market reprice
curve.calibration_result = curve_calibration_record(               # ③ assemble + attach to curve
    model_class="credit_curve_bootstrap",
    parameters=pillar_parameters(pillar_dates, pillar_survivals, label="survival"),
    residuals=residuals, quotes_fitted=[...],
    algorithm="bootstrap", iterations=len(pillar_dates), converged=True)  # facts direct
return curve                                                       # the curve IS the carrier

# FAMILY B — sabr_calibrate(...) -> SABRCalibrationResult
result = pb_minimize(objective, x0, method="nelder_mead")          # ① fit (existing optimiser)
solve = SolveReport.external(algorithm="nelder_mead",              #   CAPTURE its verdict
                             converged=result.success, iterations=result.nit, …)
cr = model_calibration_record(model_class="sabr", parameters={α,β,ρ,ν},  # ② assemble (report required)
                              residuals=[...], quotes_fitted=[...], solve=solve)
return SABRCalibrationResult(..., calibration_result=cr)           # ③ store eagerly on the result
```

**Pipelines** (they converge after the builder):

```
A:  quotes → solve (brentq/Newton/closed-form) → reprice residuals ─┐
B:  quotes → optimiser → SolveReport.external(result) ──────────────┤
                                                                     ▼
                              {curve,model}_calibration_record → CalibrationResult
                                       │
              ProvenanceCarrier.to_calibration_result()  →  db.save_calibration  →  table ⇄ load
```

| | **Family A — bootstrappers** | **Family B — calibrators** |
|---|---|---|
| Producers | ~17 (`*_bootstrap`, `global`/`coupled`) | 13 (`CanonicalCalibrationResult` subclasses) |
| Returns | shared curve (`DiscountCurve`, `SurvivalCurve`, …) | bespoke result (`SABRCalibrationResult`, …) |
| Carrier | field on curve + forwarding wrappers | `CanonicalCalibrationResult` mixin |
| Builder | `curve_calibration_record` | `model_calibration_record` |
| Optimiser facts | **direct args** (honest by construction) | **captured** via `SolveReport` |
| `converged` | a known `bool` | `bool` (captured) or `None` (not captured) |
| `parameters` | per-pillar values | fitted scalars (+ surface digest) |
| Conformance gate | `test_bootstrapper_provenance_conformance` | `test_calibrator_provenance_conformance` |
| Read · persist · 3 shared gates | **shared** (provenance-carrier · fidelity · builder-enforcement) | **shared** |

The split is **only** in the producer surface (left two rows); from the builder
onward — the record, the read Protocol, the sink, and three of the five gates
(provenance-carrier, fidelity, builder-enforcement) — the two families are one
system. Only the two conformance gates are family-specific.

---

## 7. Persistence — one polymorphic sink

`PricebookDB.save_calibration(carrier: ProvenanceCarrier)` is the single consumer
that closes the build → store → read loop for **both** sides. It calls
`to_calibration_result()`, writes a row keyed on the `calibration_id` (UUID) into
the `calibration_results` table — full record as JSON, identity/quality fields
denormalised into columns so the audit chain is queryable
(`list_calibrations(model_class="hull_white")`, or by `market_snapshot_id`)
without rehydrating every blob. Idempotent on the id. `load_calibration(id)`
round-trips via the `@serialisable_convention` machinery (flat-dict serialisation
with `schema_version`; UUID / datetime / tuple atom support).

It raises legibly for the two failure modes: a non-carrier, and a carrier whose
`to_calibration_result()` is `None` (e.g. an uncalibrated curve — nothing to
persist).

```
        Pattern A (curve / wrapper)              Pattern B (mixin result)
           curve.calibration_result            result.to_calibration_result()
                          \                          /
                           →   CalibrationResult   ←
                                      │
                       db.save_calibration(carrier)
                                      │
                     calibration_results table  ⇄  load_calibration(id)
```

---

## 8. Enforcement — the gates

Convention rots; five test gates turn it into CI-enforced invariants.

- **`test_bootstrapper_provenance_conformance.py`** — AST-discovers every public
  `bootstrap*` / `*_bootstrap` function and asserts each is classified COVERED
  (and behaviourally produces a non-None, DB-round-tripping record) or
  ALLOWLISTED with a reason (only `bootstrap_ci`, a statistical resampler, is).

- **`test_calibrator_provenance_conformance.py`** — AST-discovers every
  `CanonicalCalibrationResult` subclass; asserts each is classified, satisfies
  the mixin contract (field + own `_build_calibration_record`), and (where
  cheaply constructible) builds a valid, persistable record.

- **`test_provenance_carrier.py`** — the read abstraction: all three realisations
  satisfy the Protocol and are substitutable at `save_calibration`; the error
  paths raise; **plus an AST sweep** that fails if any class carrying a
  `calibration_result` forgets `to_calibration_result()`.

- **`test_calibration_fidelity.py`** — the record must be *honest*: across every
  produced record, residuals are non-empty and 1:1 with quotes, `algorithm` is a
  canonical snake_case key (never `"unknown"`), and `model_class` is globally
  unique across families (delegation aliases excepted).

- **`test_calibration_builder_enforcement.py`** — the single-builder lock:
  AST-asserts `CalibrationResult` / `CalibrationFit` / `OptimiserRun` /
  `OptimiserSpec` are constructed **only** in the type-definitions module and the
  two builders. A producer that hand-rolls a record — reintroducing the
  eager/lazy duality or fabricated convergence — fails CI.

Plus type-level invariants in the constructors themselves (`CalibrationFit`
rejects empty residuals, unlabelled residuals, and non-snake_case keys;
`OptimiserSpec` canonicalises `algorithm`) — caught at construction, not just by
a gate.

---

## 9. Layer & dependency notes

This package sits at **L0** in the empirical dependency graph. Its only load-time
pricebook dependency is `core.serialisable` (also L0), pulled in to make the
record types first-class serialisable artefacts; that edge is acyclic
(`core.serialisable` imports nothing from `calibration`). The lone in-function
`import pricebook` in `CalibrationProvenance.stamp` (to read `__version__`) is
lazy and creates no runtime cycle.

Core curve types (`core/discount_curve.py`, `core/survival_curve.py`) stay
**free of any runtime calibration import**: the `calibration_result` field is a
string-annotated attribute, and `to_calibration_result()` merely returns it — so
no curve module imports the calibration layer. `ProvenanceCarrier` is satisfied
structurally, not by inheritance, precisely to keep that edge absent.

---

## 10. How to add provenance to a new …

**… bootstrapper** (returns a curve):
1. Ensure the returned curve type has a `calibration_result` field and a
   `to_calibration_result()` accessor (the structural sweep will tell you if
   not).
2. After solving, compute model-minus-market `residuals` with parallel
   `quotes_fitted`, then `curve.calibration_result = curve_calibration_record(...)`
   with a snake_case `model_class`.
3. If you return a wrapper, forward `calibration_result` to the inner curve.
4. Add the function to the COVERED registry in the bootstrapper gate.

**… calibrator** (returns parameters) — the whole job is ~6 lines:
1. Make the family-result a `@dataclass` subclassing `CanonicalCalibrationResult`,
   declaring `calibration_result: CalibrationResult | None = None`.
2. In the calibrate function: run your optimiser, then **capture** its verdict —
   `SolveReport.external(algorithm=…, converged=result.success, iterations=…)`
   (or use a `_solve.py` primitive, which returns the report for you; or
   `SolveReport.analytic()` for closed-form). Then
   `model_calibration_record(model_class=…, parameters=…, residuals=…,
   quotes_fitted=…, solve=report)`, stored eagerly on the result. Never write
   `converged=` from a threshold — capture it or pass `None`.
3. Implement `_build_calibration_record()` as the hand-built fallback: same
   builder, with `SolveReport.external(converged=None, …)` (no optimiser ran) and
   `reconstructed=True`.
4. Add the class to `CLASSES` / `COVERED` in the calibrator gate.

---

## One-sentence shape

One immutable four-part record (`CalibrationResult`), **captured** from the
optimiser via a `SolveReport` and **assembled** through one of two L0 builders,
read through one Protocol (`ProvenanceCarrier`) realised two ways — a **field on
the curve** when the curve is the artefact, a **mixin method** when parameters
are — persisted through one polymorphic DB sink, and held to the contract by
five gates plus the constructors' own type-level invariants.
