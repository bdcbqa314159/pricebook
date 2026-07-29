# Artifact #22 — F2: the Model + Calibrator foundation (cross-asset) · rev 3

**Status:** RATIFIED (rev 3, 2026-07-29), after adversarial re-review — Q2 worked-example formula
corrected (`PV = N·(par_rate − K)·RPV01`) and §0 ground 2 re-grounded (a clean (B) collides with
pricing@L4 / calibration@L3, not §3d; it is deferred for want of a consumer).
**Design only; code lands with F2's first consumer (T1's swap).** The capability model (rev 1) is
right and stands. This revision does one thing: **rev 1 stated three claims that are true for F2's
actual remit — linear + closed-form-analytic — as if they were universal, cross-asset, settled-once
laws. The universalisation is where each broke.** Rev 2 re-scopes them, names the limitations with
re-open triggers, and fixes one API bug. The review's four attacks and their resolutions are logged
in §R.

**Rev 3 (this pass) changes nothing in the capability model or the (A) fork; it *grounds* two
already-made arguments in the now-ratified CLAUDE.md §3d ("Building-block discipline — shared
primitives across the spine"):** Q2's safety is restated as **sharing the building-block primitives**
(drift-proof by construction), with the reprice-through-engine test demoted to a *backstop*; and §0's
second ground for (A) is softened to "a further reason to defer," leaving the fork resting on the
airtight ground 1. Nothing new is decided. See the §R note.

**The load-bearing idea, now stated WITH its precondition:**

> A model is a bundle of **capabilities** (closed-form building blocks), not a pricer and not a
> parameter bag. Calibration (L3) and the engine (L4) both COMPOSE those capabilities, from opposite
> sides, and neither imports the other — **so long as every calibrating instrument has a closed-form
> price under the trial model.** The moment a calibration target is itself *numerically* priced, that
> precondition fails, and the claim needs the fork below.

---

## §0 — The fork (decided first; it drives Q2 and Q3, which are the same crack)

**The crack.** "Calibration reprices through the model's own capabilities, never L4" holds only when
the calibrating instrument has a closed-form model price (a swap from `df`s, a European from a
characteristic function). It **fails** when the target is numerically priced — a Bermudan swaption, a
barrier, a convexity-adjusted future — because the residual then needs **state evolution + payoff
logic**, which is L4 engine work. That is the *same* missing primitive as Q3's deferred path-state:
one crack, seen from two sides.

**Two ways to close it:**
- **(A) Name the limitation.** Ratify: *a calibration target must be closed-form-priceable under the
  trial model.* Re-open trigger: the first model calibrated to a numerically-priced instrument.
- **(B) Hoist.** Introduce shared L3 primitives — `StateProcess` (evolve state) + `Payoff(state_history)
  → cashflows` — composed by *both* calibration (L3→L3) and the engine (L4→L3), so exotic-target
  calibration stays L3.

**Chosen: (A), on two grounds — ground 1 is airtight alone; ground 2 is a further reason to defer.**

1. **Rule of two forbids (B) now.** `StateProcess` and `Payoff` have **zero present consumers** —
   nothing in F2's remit through T1 calibrates to an exotic. Building them now is speculative
   infrastructure ahead of a need (§6b), the exact class of thing Phase 4 deleted (`NumericalConfig`,
   the `Underlying` siblings). A foundation earns its abstractions from consumers, not from foresight.
   This ground stands on its own; the fork rests on it.

2. **Even a clean (B) is deferrable for want of a present consumer — and its collision is not with §3d.**
   Rev 2 over-stated this ground by claiming (B) *cannot* be made clean. It can — in two shapes, each of
   which *satisfies* §3d ("define the atom once; every stage composes it"): (i) one shared L3
   exotic-pricing primitive that both calibration and the engine compose, or (ii) the calibration
   *driver* placed above the engine. §3d is not what forecloses either. What does is that each shape
   re-opens a *different* ratified guardrail: shape (i) puts PV aggregation at L3 (colliding with
   **"pricing lives in L4"**), shape (ii) moves the driver off L3 (colliding with **"calibration lands
   at L3"**, CLAUDE.md §1). A real cost — but a design question with **no present consumer** to force it,
   settled by the first exotic-calibration consumer under §3d, not now. So (B) is *deferrable, not
   wrong*; the decision does not lean here, it leans on ground 1.

**Why (A) is not a retreat.** F2 being a *foundation* does not mean "prices everything now" — it means
**the shape absorbs extensions additively.** The `CalibratedModel` protocol (`.market` + opt-in
capabilities) absorbs a future `StateProcess` capability without changing. So the *type contract* is
settled once; the *limitation* is on which calibration targets are expressible today, and it is
written down with a trigger rather than discovered later. That is the honest cross-asset answer under
our own rules. **(B) is pre-named as the known future amendment** (§Q3), so the MC/PDE topic *extends*
the ratified boundary instead of breaking it.

---

## Q1 — What a model IS (stands; one correction)

Unchanged from rev 1 and still right: a `CalibratedModel` is **frozen**, carries `model.market` (the
`MarketSnapshot` it was calibrated to — A1), and satisfies one or more **capability protocols** (the
closed-form building blocks of its dynamics). No fat base class; capability is opt-in; higher layers
depend on the capability, not the concrete type (F1's `CurveHandle` one layer down). `DiscountingModel`
is the degenerate case — no free parameters, no dynamics, *built* (by F1 curve construction) rather
than *solved*; single-curve is the further degenerate (`projection(index)` → discount curve).

**Correction carried in from Q4:** rev 1 said a calibrated model "materialises once, at convergence."
That is true for a global solve but **false for a sequential bootstrap**, which builds the curve
incrementally (§Q4). The model is the *output* of calibration in both cases; *when* it materialises is
method-dependent, not a universal.

**Residual risk:** capability proliferation → a disguised type ladder. Guard (rev 1, kept): capabilities
key to *what the math is* (discounting, implied vol, characteristic function), not to *which model*
provides it; a single-implementer capability fails rule of two and folds back concrete. §Q3(a) adds the
sharper guard — a capability must pin its *semantics*, not just its signature.

---

## Q2 — The calibrator contract (restated with precondition; the cost owned)

**The contract (unchanged in shape):** `calibrate(spec) → (CalibratedModel, CalibrationResult)`, a
**free function** at L3; the model is the *output* (a model that calibrates itself holds state and
reaches for market — forbidden). `CalibrationSpec` is **data** (reproducible, serialisable); it names
the residual — reprice each calibrating instrument, compare to its quote. (API fixed — quotes now live
*inside* the spec, §5.)

**The load-bearing claim, correctly conditioned:**

> Calibration composes the model's building blocks and never imports L4 — **when the calibrating
> instruments are closed-form-priceable under the trial model.** Outside that precondition, §0(A)'s
> limitation applies: such a target is not calibratable in F2's foundation, pending the §0(B) amendment.

**Owning the cost of "two paths, one answer."** Rev 1 sold this as a free independent oracle. The
review is right that it is not. Repricing a calibrating instrument composes model capabilities with
*product structure* — that **is** pricing. Naively, keeping L3⊥L4 would mean **re-hosting that
composition inside L3, a second implementation beside the L4 engine's**, and if the two drift — a
day-count, a discounting convention — calibration converges to a curve that reprices to par under the
L3 residual but **not** under the engine. A silent bug, and exactly the doc 19 §6 mis-resolution smell
in a new place.

**The primary safety is SHARING the building blocks, not testing the two ends against each other**
(CLAUDE.md §3d, now ratified). The engine and the calibrator do not stand up two annuity loops to be
reconciled after the fact; they **compose the same `df` / `forward` / `RPV01` primitives** — one
definition, the closed-form building blocks §3 already ratifies (`df`, `RPV01`, `B(t,T)`,
zero-bond-option), reached through the model's capabilities. Because the calibrator's linear
composition (`par_rate = (df₀ − dfₙ)/Στᵢdfᵢ`) and the engine's (`PV = N·(par_rate − K)·RPV01`, K the fixed coupon) are
*built from the same shared atoms*, they **cannot drift by construction** — there is no second
day-count, no second interpolation, **and the schedule/accrual dates flow from the one L2 `Product`
description (a shared `Schedule`), never re-derived per layer** — for them to disagree on. That third
clause matters: sharing the *function* `rpv01()` fixes the loop internals, but argument-identity is
what closes the third axis §3d's worked example names — **accrual-vs-payment dates** — and it holds
only because the calibrating swap and the repriced swap are the *same* product with the *same*
schedule, not because `rpv01` is shared. Collapsing further to literally one
composition would force PV-aggregation up to L3 (the §0(B) smear); sharing the *atoms* while each layer
composes its own tiny expression is the clean middle, and it is precisely what §3d mandates ("an atom
is defined once; every stage composes it").

**The reprice-through-engine test is now a BACKSTOP, not the proof.** T1's requirement — *the
calibrated instrument reprices to zero NPV through the full L4 engine, not merely through the
calibrator's residual* — is retained but demoted: it **confirms the primitives were in fact shared**,
it does not *prove* the two ends equivalent. And it cannot prove that: being a reprice-to-par oracle,
it is **blind to exactly the class of assumption error it was meant to catch** — blind **both** to
mis-resolution (doc 19 §6: a par instrument reprices to par even off a mis-resolved-but-self-consistent
curve) **and**, for the calibrating par instrument itself, to the annuity/schedule divergence above —
because `(par_rate − K) = 0` makes `PV = N·(par_rate − K)·RPV01` insensitive to *which* `RPV01` the
engine composes (par swaps telescope, §3d). That second blindness is the point, not a footnote: it is
*why* the schedule must itself be a shared atom, not merely why the test is a backstop. A test that
shares the blind spot of the thing it guards cannot be the guarantee; the guarantee lives in the
**sharing**, and the test only backstops it. If duplication ever grows beyond the linear/analytic
trivial case, the shared composition is hoisted to a primitive both layers call (§3d) — not needed for
F2's remit, so not built now.

**Consequence, written into the contract:** F2's *correctness* is **structural** — the shared-atom
guarantee is checkable at code review, independent of L4. But its *empirical confirmation* still
requires L4: **F2 cannot be *proven* in isolation**, because the backstop that confirms the sharing
actually happened is an L4 operation. Therefore **F2 is proven by T1's swap, not on its own** —
consistent with the whole "contract now, code with the first consumer" philosophy, and stated here so
no one mistakes a green F2 unit for a proven foundation.

**Residual risk:** the backstop is a test, and a test can be forgotten. Mitigation: it is named as a T1
gate in §Q4's oracle and in doc 18's C2 close condition ("a par swap prices to zero NPV off the built
curve, end to end") — the backstop already exists as a ratified T1 requirement; F2 only points at it.
The *primary* safety, note, does not depend on anyone remembering the test — it is structural (shared
atoms), per §3d.

---

## Q3 — The engine boundary (down-scoped; extension pre-named; auto-selection corrected)

**The boundary statement, now scoped.** Rev 1's symmetry — *the model knows nothing about the product;
the engine knows nothing about the dynamics* — is true for **scalar / closed-form capabilities** and
**false under path-dependence.** A `paths(...)` capability is not semantically self-contained: the
engine would have to know which state variable is which and under what measure (dynamics leaking into
L4), and the escape — the model evaluates the payoff — leaks the product into L3. So:

> **Q3's boundary is ratified for composable scalar / closed-form capabilities only.** Within that
> scope the symmetry holds exactly and the engine composes building blocks to price.

**The extension is pre-named, not left to be discovered.** Path-dependent / callable pricing arrives
via the §0(B) `StateProcess` + `Payoff` split, as a **known future amendment** tied to the same trigger
as §0(A). Because `CalibratedModel` is `.market` + opt-in capabilities, that amendment **adds a
capability additively** — the MC/PDE topic *extends* this boundary rather than breaking it. The
protocol stays; only the *boundary statement's scope* widens when the primitive lands.

**The concrete signal that fires the build is CLAUDE.md §3d's exception-count gauge.** When
`isinstance` / special-case exceptions **cluster at the exotic boundary** — the engine and the
calibrator both forced to special-case a numerically-priced target instead of composing shared atoms —
that cluster (a reusable relationship recurring) is the ratified trigger to **name and build the
`StateProcess` + `Payoff` block**, exactly as §3d prescribes for "a building block was missing." Until
that cluster appears there is no consumer, and §6b holds the build.

**Auto-selection corrected (review 3b).** Rev 1's wording implied the registry *dispatches on capability
satisfaction* — i.e. auto-picks the engine. Wrong. When a model satisfies several capabilities (Heston
has both a characteristic function and a path evolution; a European can price by Fourier *or* Monte
Carlo), **choosing which is an explicit numerics decision**, carried in the numerical config — not an
automatic lookup. The registry's job is to **validate** that the chosen engine's required capability is
satisfied by the model, and to fail loudly if not. It validates; it does not pick.

**Residual risk:** the scope line ("scalar/closed-form") is a judgement call at the margin — a
convexity-adjusted future is "closed-form" only given a convexity model. Such borderline targets are
resolved by §0(A)'s test (is there a closed-form price under *this* trial model?), not by taxonomy.

---

## Q3′ — Two ratification conditions for FUTURE capability protocols

Nothing changes for the only ratified capability, `Discounting` (`df(t)→float` is self-specifying).
These gate every capability added later:

**(a) A capability must pin its SEMANTIC contract, not just its type signature.** `df(t)→float` is
self-specifying; `characteristic_function(u, t)→complex` is *not* — cf of what, under which measure,
which numeraire, what argument convention? Without the semantic contract, "shared capability" degrades
to *shared signature + per-model semantics* = `isinstance` with extra steps. A capability protocol
ships with its measure/numeraire/units/argument conventions stated, or it is not ratified.

**(b) Engine selection among multiply-satisfiable capabilities is a NUMERICS choice, not a lookup**
(the corrected §Q3 rule, stated as a standing condition): the config picks the method; the registry
validates the requirement is met.

---

## Q4 — The contract survives; the implementation narrative is fixed

**What survives (the contract):** one `calibrate` signature; one residual definition (reprice-vs-quote);
`method` selects between approaches **as data** — *including the global multi-curve case.* Rev 1's
"build a trial model from a parameter vector" (old line 99) read single-curve; made explicit now: in a
**global solve the trial "model" is the entire jointly-solved `CurveSet`** — discount + every
projection + basis — spanning all pillars at once, and the residual vector is every calibrating
instrument across all curves. The contract does not change; only its instantiation is wider.

**Over-claims corrected:**
- Rev 1: "no per-family loop; the model materialises once at convergence." **False for sequential
  bootstrap** — that *is* a loop of 1-D root-finds against a **partially-built** curve, each pillar
  solved using the pillars already fixed. It materialises incrementally.
- **Jacobian provenance differs by method** (same result field, different means): a **simultaneous**
  solve gets the Jacobian from the solver's own iteration; a **bootstrap** forms it **post-hoc**
  (analytic, or bump-and-rebuild). Rev 1 implied one uniform source.

**Reframed:** `calibrate` dispatches, by `spec.method`, to one of **two orchestrations** —
an **incremental 1-D sweep** (partial curves, post-hoc Jacobian) or a **global N-D solve**
(solver-supplied Jacobian). **Signature and residual are shared; orchestration and Jacobian provenance
are not.** That is the precise sense in which "the difference is data": the *interface* is one, the
*inside* legitimately forks, and the fork is declared (`method`) rather than hidden behind model type.

**Residual risk:** two orchestrations behind one function could drift in what "residual" means (a
bootstrap's 1-D residual vs a global vector residual). Guard: both are defined as *reprice-vs-quote for
the calibrating set*; the bootstrap's is the degenerate 1-element case of the global vector. Oracle:
**single-curve == multi-curve degenerate, exactly** (doc 18 §8) — the test that the two orchestrations
agree where they overlap.

---

## Minimal type sketch (rev 2; API bug fixed)

```python
# ---- L3: capabilities (only Discounting is ratified; others gated by Q3′) ------------------
class CalibratedModel(Protocol):
    @property
    def market(self) -> MarketSnapshot: ...                     # A1

class Discounting(Protocol):                                     # 2 consumers today (every linear leg)
    def df(self, t: date) -> float: ...
    def forward(self, index: RateIndex, start: date, end: date) -> float: ...
# Future capabilities ship WITH their semantic contract (measure/numeraire/units) — Q3′(a).

@dataclass(frozen=True)
class DiscountingModel:                                          # degenerate: built, not solved
    market: MarketSnapshot                                       # df/forward delegate to market.curves

# ---- L3: calibration -----------------------------------------------------------------------
class CalibrationMethod(Enum):
    SEQUENTIAL   = auto()    # incremental 1-D sweep; partial curves; post-hoc Jacobian
    SIMULTANEOUS = auto()    # global N-D solve; solver-supplied Jacobian

@dataclass(frozen=True)
class CalibrationSpec:                                           # DATA — self-describing, serialisable (F1 3.4)
    targets:  TargetSet                                          # what parameters are solved
    quotes:   QuoteSet                                           # what pins them — SINGLE source of truth (fix §5)
    method:   CalibrationMethod
    numerics: SolverConfig                                       # explicit knobs; concrete type lands w/ 1st calibrator
    #  4 fields — ≤5 ✓

@dataclass(frozen=True)
class CalibrationResult:                                         # BESIDE the model, never an attribute (F1 §5)
    quotes_used: tuple[QuoteId, ...]
    residuals:   tuple[float, ...]
    iterations:  int
    converged:   bool
    jacobian:    Jacobian | None                                # solver-supplied (global) or post-hoc (bootstrap)
    #  5 fields — ≤5 ✓

def calibrate(spec: CalibrationSpec) -> tuple[CalibratedModel, CalibrationResult]:
    ...   # free function; the spec is fully self-describing (quotes inside it), so a serialised
          # spec reproduces the calibration exactly. Reprices calibrating instruments through the
          # model's OWN capabilities (never L4) — VALID ONLY when they are closed-form-priceable (§0A).

# ---- L4: the engine composes capabilities; numerics picks the method, registry validates ----
def price(instrument: Product, model: CalibratedModel, numerics: SolverConfig) -> PricingResult:
    ...   # composes the capability the chosen method requires; registry VALIDATES satisfaction,
          # does not auto-select (Q3′b). Dispatch is structural, never isinstance.
```

`calibrate(spec)` — one argument, one source of truth. `CalibrationSpec` 4 fields, `CalibrationResult`
5; both obey §3b.

---

## What I did NOT decide (rev 2)

- **`StateProcess` + `Payoff`** — the §0(B) primitive for numerically-priced targets. Pre-named as a
  known amendment; **built by the first MC/PDE topic**, never before (rule of two).
- **The concrete `SolverConfig`** — lands with T1's bootstrap, decomposed by method family, ≤5 fields
  (the deleted-`NumericalConfig` lesson).
- **Every capability beyond `Discounting`** — each arrives with its second model *and* its semantic
  contract (Q3′a).
- **`jacobian: Jacobian | None` — the `None` branch** — decided by the first calibrator that forms
  none; T1's curve solve always forms one.
- **The two orchestrations as concrete algorithms** — T1 builds both and proves single==multi degenerate.
- **AAD / adjoint Jacobians** — later topic (doc 18 §6); F2's Jacobian is the bump-and-rebuild / analytic
  result AAD must later reproduce.

---

## Oracle summary (where provable vs merely self-consistent)
| capability | pricing oracle | strength |
|---|---|---|
| `Discounting` (linear) | reprice-to-par **through the L4 engine** (§Q2 **backstop**; the drift guard is the shared `df`/`forward`/`RPV01` atoms, §3d) · DV01 analytic vs finite-diff | self-consistent for par; closed-form for a flat-curve PV |
| credit `survival` | CDS reprices to quoted spread | **self-consistency only** unless flat-hazard closed form / QuantLib |
| equity `cf` (Heston) | European reprices to quoted vol | closed-form via cf; **external** cross-check (QuantLib) |

Weak spot, carried from doc 19 §6 into calibration: **assert the resolved identity against the
curve/model being built; never infer it** — reprice-to-par/spread is blind to mis-resolution.

---

## §R — Review resolutions (this pass)
| # | attack | resolution |
|---|---|---|
| fork | Q2 claim fails for numerically-priced targets (= Q3's path-state; one crack) | **§0**: chose (A) name-the-limitation, leaning on **ground 1** (rule-of-two forbids (B) now — zero present consumers, airtight alone); **ground 2 softened (rev 3)** — (B) *can* be made clean and §3d is satisfiable; a clean (B) merely re-opens a *different* guardrail (pricing@L4 or calibration@L3) and, decisively, has no present consumer — a further reason to defer, not the load-bearing one. (B) pre-named as a known amendment. |
| 1 | "two paths, one answer" is duplication risk, not a free oracle; and F2 self-validation needs L4 | **Q2 (regrounded rev 3)**: primary safety is **sharing the `df`/`forward`/`RPV01` atoms *and* the one L2 `Schedule`** (§3d) — drift-proof by construction across all three axes (day-count, interpolation, accrual/payment dates); the reprice-**through-the-engine** test is demoted to a **backstop**, admitted blind (doc 19 §6, *and* to telescoping annuity drift on the par instrument itself); correctness is structural, **empirical proof is T1's swap, not F2 in isolation** — kept. |
| 2 | Q3 symmetry false under path-dependence | **Q3**: scope the boundary to scalar/closed-form; pre-name the `StateProcess`+`Payoff` extension; protocol stays, only the boundary statement's scope widens later. |
| 3 | capabilities need semantics not signatures; engine selection isn't auto | **Q3′**: two ratification conditions for future capabilities — pin the semantic contract; selection is a numerics choice, registry validates not picks. |
| 4 | Q4 implementation over-claims (single materialise; uniform Jacobian; single-curve trial) | **Q4**: two orchestrations by `spec.method` (incremental sweep vs global solve); shared signature+residual, forked orchestration+Jacobian provenance; global trial model = whole `CurveSet`. |
| 5 | `calibrate(quotes, spec)` duplicates `spec.quotes` | **type sketch**: `calibrate(spec)`; quotes fold into the spec; field counts rechecked (4 / 5). |

**Rev 3 note (grounding, not new decisions).** Rev 3 lands after CLAUDE.md §3d ("Building-block
discipline — shared primitives across the spine") was ratified, and uses it to *ground* two arguments
this doc had already made from first principles. **Q2** no longer rests its safety on a test policing
two duplicated compositions; it rests on §3d's rule that the engine and calibrator **compose the same
`df`/`forward`/`RPV01` atoms *and* the one L2 `Schedule`** — one definition — so the linear composition
cannot drift by construction on any of its three axes (day-count, interpolation, accrual/payment
dates), with the reprice-through-engine test explicitly demoted to a backstop (and noted, per
doc 19 §6, to be blind to the very error class it once claimed to catch). **§0's ground 2** is softened:
(B) can be made clean — §3d is *satisfiable* — so the fork no longer claims otherwise; a clean (B) merely
re-opens a *different* guardrail (pricing@L4 or calibration@L3) and has no present consumer, so it is
deferred, the fork leaning on the airtight ground 1 (rule of two). The capability model, the (A) fork outcome, and every type in the sketch are unchanged.
The §3d **exception-count gauge** is now cited (Q3) as the concrete trigger that will fire the deferred
§0(B) `StateProcess`+`Payoff` build.
