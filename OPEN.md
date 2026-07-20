# Open items — single tracker

> **This file is TRACKED, not gitignored.** `.gitignore:52` ignores root `/*.md`, but `:58` carries an
> explicit `!/OPEN.md` exception (CI needs it as a `verify.py` input — `CLAUDE.md §7c`). **Every edit
> must land in a commit** — a local-only edit silently diverges the ledger from CI. (V1, fixed
> 2026-07-19: the former "edit freely, no commits needed" note was a stale quarry-era leftover.)
>
> **This document spans two trees. Read the section header for its subject:**
> - **§"NG / Foundation audit closure …"** and everything under the `AC-*` / `NG-*` ids describe the
>   **new tree** (`pricebook_ng`, `src/`, `tests_ng/`) — currently `v0.82.0`, 149 tests.
> - **§"HOT TOPIC"** and the legacy-debt items below it describe the **quarry** (`python/pricebook/`) —
>   last reviewed 2026-06-19 at `v1.119` (8268/8268 L≤3, 12,797 full suite). Those figures are quarry
>   figures; they say nothing about the ng tree. (V2, 2026-07-19.)

---

## NG audit-closure — deferred-scope ledger (load-bearing temporaries)

*(empty — NG-DEFER-1 discharged.)*

**NG-DEFER-1 — ACT/ACT ICMA long-stub refusal — DISCHARGED (Phase 3b increment 2).** The raise in
`_act_act_icma` is deleted; long coupons now compute Rule 251.2 (summation over notional periods),
pinned against the ISDA 2006 §4.16 published long-first-coupon example (0.9157608695652174). The
re-open trigger fired as planned; no residual debt.

---

## Foundation audit closure — Tier-4 & deferred-scope ledger (Phase 5, 2026-07-19)

Closes the three independent foundation audits (`redesign/independent_audits/closed_*.md`). Every
finding is either fixed-with-a-test (Phases 0–4, see `CHANGELOG.md` v0.75.0–v0.80.0 + the per-finding
disposition blocks in the closed reports) or ledgered below with a **named re-open trigger**. These
are *deferred scope*, not hidden wrongness — none blocks building the next layer. (Format note: these
are deliberately **not** `- [NG-…]` entries; they do not offset a suppression, so they stay out of the
`verify.py debt` balance.)

**Phase-5 reclassification (2026-07-19).** Every ledgered item was re-tested against one question — *does
it make the library give a WRONG ANSWER or accept INVALID INPUT today?* Result: **only AC-T4.2 was YES**
(zero/negative `Tenor` accepted) — promoted to a fix (v0.82.0), removed from the ledger. **EURIBOR_6M**
was fixed alongside (not wrongness — a gap in a set we present as complete). Everything else is NO —
absent capability or clean rejection, correctly deferred (building now = speculative infra ahead of its
consumer, §6b). Notably: T4.10/T4.11 **raise** on bad input (no silent NaN); T4.7/T4.8/T4.9 have **no
triggering consumer** among the registered calendars/indices today; T4.17 (`Weekend` time-invariant)
matches the current Sun–Thu Israeli convention in code — the audit's "Israel Mon–Fri from 2026" is an
unverified external claim, not a demonstrated wrong answer.

**Trigger-reality audit.** Most triggers name a real roadmap topic (Topic 1, models, credit, FX, L6).
**Six do not** — they are condition/event/usage-driven, not topic-driven, and are flagged here so they
are not mistaken for scheduled work: **AC-T4.7 / AC-T4.8 / AC-T4.9** fire only if a *future calendar or
`NEAREST` consumer* introduces the pattern (guarded-by-convention today); **AC-T4.11** is error-message
quality on economically-invalid input; **AC-T4.17** fires on a *real-world weekend-rule change* (external
event); **AC-T4.18** is DRY debt (fires on a 4th consumer, rule of three). These stay ledgered as latent
traps / debt with condition-based triggers — honest, but not roadmap items.

### Deferred sub-parts of otherwise-fixed findings

| id | finding | what shipped | what is deferred | re-open trigger |
|---|---|---|---|---|
| AC-2.2b | USD calendars (audit 2.2) | `US_GOVERNMENT_SECURITIES` (SIFMA + Good Friday, `SUNDAY_ONLY`); SOFR bound to it | separate **NYSE** and **Fed-bank/EFFR** calendars (different Good-Friday/half-day/observance) | first **equity (NYSE)** or **EFFR** consumer lands |
| AC-2.4b | Tokyo calendar (audit 2.4) | astronomical `equinox(3/9)`; Emperor's-Birthday moves (`2,23 since 2020` / `12,23 until 2018`) | **Silver Week** sandwiched-holiday (*kokumin no kyūjitsu*) + **2020/2021 Olympic** one-off shifts | JPY-calendar completeness / **equity-JP** topic |
| AC-3.6b | FX spot (audit 3.6) | correct `fx_spot_date` (joint-calendar count + USD-holiday-for-cross); `spot_lag` out of `CurrencyPair` equality | (1) **FX pair-conventions registry** (quote order, cross triangulation) — L1 market-data scope, not L0. (2) **B3 intermediate-day rule**: the count pauses on *any* closed centre (joint); the asymmetric ACI rule (a USD holiday on an intermediate day does not pause a USD pair's count) is **not** implemented — no citable source with a verifiable worked example, and the green-oracle gate forbids coding an unverifiable convention. `fx_spot_date` is thus a *joint-count* spot algorithm, not the full ACI algorithm. | **FX market-data** topic (L1) — supplies both the registry and a citable ACI oracle |
| AC-C1 | Half-day / early-close calendar classification (POST_CLOSURE C; S5 redirected) | **nothing** — the unused half-day table (`day_type`, `DayType`, `HolidaySet.half_days`, `_half_days_of`, the 3 US half-day rules) was **deleted** at v0.83.0 (an unread table is worse than an absent one: nothing tests it, so the first consumer trusts it unverified) | the **concept** — a day's trading status (BUSINESS/HALF/HOLIDAY/WEEKEND) and per-market early-close rules. Rebuild it *with* its consumer, shaped by the actual cut-off need, and oracle it against published early-close dates | first **fixing-cutoff / early-close consumer** (e.g. a SOFR/SONIA fixing time, an option expiry cut, an equity half-day settlement) |

### AUDIT.md Tier-4 — rides with its asset-class topic (18 items, all "scheduled, not discovered")

| id | item | where | re-open trigger |
|---|---|---|---|
| AC-T4.1 | Index registry too thin — remaining: EFFR, BBSW, AONIA, CORRA, TIIE 28D, SELIC, WIBOR/PRIBOR/BUBOR/JIBAR. *(**EURIBOR_6M fixed** v0.82.0 — was a gap in a set we call complete.)* | `rate_index.py` | each index's currency/product topic (rates/EM/credit) |
| ~~AC-T4.2~~ | ~~Zero/negative tenor accepted~~ — **FIXED v0.82.0** (reclassified YES: accepted invalid input). `Tenor.__post_init__` rejects non-positive counts; test `test_t4_2_non_positive_tenor_is_rejected`. Removed from ledger. | `tenor.py` | — (closed) |
| AC-T4.3 | `distributions.py` too thin (bivariate normal CDF, non-central χ²) | `distributions.py` | first model/engine that needs them (Topic 2+) |
| AC-T4.4 | `least_squares` cannot bound (`method="lm"` hardcoded; need `trf` for Feller, \|ρ\|<1) | `solvers.py` | first stoch-vol calibration |
| AC-T4.5 | No `TimeMeasure(anchor, day_count)` concept | absent (invariant at `rate_basis.py`) | **RULED (A1, redesign/20 Part A addendum): `TimeMeasure` is the only sanctioned `date→t` map; build as an L0 module in Topic 1's first slice, with its first curve consumer.** Promoted "someday"→"before Topic 1". |
| AC-T4.6 | CDS maturity roll pre/post-2015 (`standard_cds_maturity`) | `schedule.py` | credit layer |
| AC-T4.7 | `is_holiday` forward year-spill checks `year`/`year+1` but not `year−1` (Dec→Jan observed) | `calendars.py` | next calendar whose Dec holiday observes into January |
| AC-T4.8 | `observe()` hardcodes Sat/Sun (wrong for a future mondayising FRI_SAT market) | `calendars.py` | first FRI_SAT calendar that mondayises |
| AC-T4.9 | `NEAREST` tie-break rolls backward (QuantLib/practice roll forward) — decide + document | `calendars.py` | first `NEAREST` consumer |
| AC-T4.10 | `log(y≤0)` unguarded (negative rate/spread or underflowed DF → bare `math domain error`) | `interpolation.py` | curve layer stores negative-carrying series (Topic 1) |
| AC-T4.11 | `convert_rate` on negative growth factor (`log(−x)` crash / NaN) — reject-vs-define | `rate_basis.py` | first negative-growth path |
| AC-T4.12 | CDI rate-application trap (returns annualized exp rate; `r·yf` consumer silently wrong by convexity) | `rate_index.py` | BR curve / CDI-swap product |
| AC-T4.13 | Interpolators rebuilt per call (O(N·M)) — also `PONYTAIL-DEBT` marker | `interpolation.py` | Topic 1 (hot curve caches the interpolator) |
| AC-T4.14 | `Money` unrounded float; `minor_units` decorative — document so no ledger/settlement assumes rounding | `money.py` | booking/settlement (L6) |
| AC-T4.15 | Two `frequency` concepts (`Frequency` tenor-step vs ICMA `frequency: int`/yr), no bridge — decide 28D/TIIE | `schedule.py` vs `day_count.py` | **RULED (A2, redesign/20 Part A addendum): `Frequency.per_year()` raises for non-integer tenors (28D/daily/bullet); BUS-period products (TIIE/CDI) do not enter ICMA contexts. Implement with the fixed-leg builder.** |
| AC-T4.16 | No time-of-day/timezone story (`datetime.time` + IANA zone for expiry cuts, equity closes) | (was `underlying.py`) | FX options / equity topic |
| AC-T4.17 | `Weekend` time-invariant (Saudi 2013; Israel Mon–Fri from 2026 → `TEL_AVIV` wrong forward) — record or add `since=` | `calendars.py` | when a weekend-rule change enters scope |
| AC-T4.18 | Month-arithmetic triplication (3 near-duplicate add-months/EOM helpers) — DRY debt, fine to leave | `tenor.py`, `schedule.py`, `day_count.py` | when a fourth consumer appears (rule of three) |

### PONYTAIL-DEBT.md — ponytail markers (1, tracked)

| id | marker | ceiling | upgrade trigger |
|---|---|---|---|
| AC-PD.1 | scipy spline rebuilt per `interpolate()` call (`interpolation.py`) | O(N) rebuild/eval → O(N·M) for M queries; `_boundary_slope` doubles it on extrapolation | Topic 1 — a hot curve caches the interpolator itself (= AC-T4.13) |

---

## 🔥 HOT TOPIC (parked, revisit) — serialisation typing debt (mypy)

**Surfaced 2026-07-02** while hardening `data_registry.py` (the `from_dict` typing error, now fixed locally with a `Convention` Protocol). The same pattern recurs repo-wide: **118 mypy errors across ~50 files**, all the serialisation contract.

**Diagnosis:** `core/serialisable.py`'s `Serialisable` base IS correctly typed, but usage is **~20/21 via the `@serialisable` / `@serialisable_convention` decorators** (19 convention + 2 instrument), which inject `to_dict`/`from_dict`/`_SERIAL_*` at runtime via `cls.from_dict = …`. **mypy cannot see decorator-injected methods** → ~110 `type[X] has no attribute from_dict/to_dict/_SERIAL_TYPE` errors at consumers. The rest are self-contained in `serialisable.py` (`classmethod used with a non-method`, `__init__ unsound`). Pure type-hygiene — **zero runtime value**; the library works (13k tests green).

**STATUS: Phase 1 DONE (v1.238.0)** — option (A) chosen. Built the typed `SerialisableConvention` base (inheritance replaces `@serialisable_convention`; type checkers now see `to_dict`/`from_dict`), behaviour byte-identical, base is mypy-clean. Migrated 6 conventions (EquityIndexSpec + 5 composite_convention). Consumer errors **115 → 113**.

**PHASE 2 (remaining ~20 convention classes) — the sweep.** Per-class pattern (verified):
```
@serialisable_convention("key")          →   @dataclass(frozen=True)
@dataclass(frozen=True)                       class Foo(SerialisableConvention):
class Foo:                                         _SERIAL_TYPE = "key"   # AFTER the docstring!
    """doc"""                                      """doc"""  ...
```
- **Gotcha:** insert `_SERIAL_TYPE` AFTER the class docstring (before it demotes the docstring to a bare string). `schema_version=N` → also set `_SERIAL_SCHEMA_VERSION = N` (e.g. `calibration_result`). Update each file's import `serialisable_convention` → `SerialisableConvention`.
- **Remaining files/classes:** `credit/sovereign_cds.py` (sovereign_cds_conventions); `credit/cds_conventions.py` (cds_index_spec, cds_settlement_convention); `core/market_conventions.py` (commodity_contract_spec, linker_convention); `core/rate_index.py` (rate_index); `calibration/_types.py` (optimiser_spec, optimiser_run, calibration_diagnostics, calibration_provenance, calibration_fit, calibration_result [schema_version=3]); `fixed_income/` esg_bonds, sovereign_bonds, ois, sukuk, repo_specialness, inflation_indices, supranational; `curves/` curve_builder (currency_conventions), em_curve_builder.
- **Per file:** migrate → `.venv/bin/python -m pytest` that module's serialisation test + a `to_dict`/round-trip check → measure mypy consumer count drop. Golden net = the existing per-type serialisation tests + full suite.
- **After all migrated:** remove the `@serialisable_convention` decorator (0 users) + the dead `@serialisable` envelope decorator (already 0 users). Then the 476/621 "classmethod used with a non-method" errors vanish too.
- **NOT covered by this:** the non-convention errors (DiscountCurve/Trade/CDS/vol_surface `_SERIAL_TYPE`/from_dict) come from the `Serialisable` base / custom paths, a separate sub-problem — assess after conventions are done.

**Options (A–D) [historical, A chosen]:**
- **(A) Root-cause refactor** — give conventions a typed `SerialisableConvention` **base** (declares the methods with types) and migrate the 19 `@serialisable_convention` classes to inherit it. Genuinely mypy-clean, zero runtime change, but ~20-file migration → do it **golden-net-guarded, phase by phase** like the calendar refactor. ← the real fix.
- **(B) `if TYPE_CHECKING:` stubs** — per-class static method stubs so mypy sees them. Additive, but ~21 files of stubs.
- **(C) Bounded** — fix only `serialisable.py`'s own ~6 internal errors + document the decorator-path errors as a known pattern; leave the 110 consumer errors.
- **(D) Scattered `# type: ignore`** — argued **against** (noise across 50 files).

**Recommendation when we reach it:** (A) if we want the codebase mypy-clean (best long-term), else (C) bounded + documented. NOT (D). Needs a golden/behaviour net first (serialisation round-trip snapshot) since it touches the serialisation of ~20 instrument/convention types.

**Audit findings (2026-07-04, `serialisable.py` content audit — separate from the typing debt but same module/golden-net):**
- **Triplicated serialisation logic (consolidation).** The `to_dict` loop and the `from_dict` "build kwargs from params via hints + `_deserialise_atom`" loop each appear **3×** — `Serialisable` base (envelope), `@serialisable` decorator (envelope, near-exact copy), `@serialisable_convention` decorator (flat, 3rd copy of the kwargs loop). The "Fix A.11 B6" schema/structured-error patches had to be applied across copies and can drift. Fix: extract `_auto_to_dict(instance, fields, flat=...)` + `_auto_from_dict(cls, payload, fields)` shared helpers; base + both decorators call them (3→1). **Do this BEFORE the typing fix** — typing one implementation is far easier than three, and it's the natural precursor to option (A). Golden-net-guarded (round-trip snapshot).
- **Latent dict round-trip gaps (2 asymmetries).** `_serialise_atom` recurses dict *values* but leaves non-str *keys* as-is → a `dict[date,·]`/`dict[Enum,·]` field serialises to non-JSON keys (`json.dump` fails). `_deserialise_atom` has `list[X]` handling (incl. `list[Serialisable]`) but **no `dict[K,V]` branch** → a `dict[str, Serialisable]` field serialises to nested dicts but deserialises to raw dicts (objects not rebuilt). ~12 serialisable classes have dict fields (mostly `dict[str,float]`, safe) — check those when consolidating; add symmetric dict handling to both atoms.
- **Minor:** CurrencyPair is special-cased by duck-typing (`hasattr base/quote/value`) in `_serialise_atom` — fragile if it ever gains a `to_dict`.

**Sub-item (2026-07-04):** the `Convention` Protocol + `T = TypeVar("T", bound=Convention)` I added to `data_registry.py` at v1.218 surfaces a **type-var error at a load_registry consumer** — `cds_conventions.py:168: Value of type variable "T" of "load_registry" cannot be "CDSIndexSpec" [type-var]` (CDSIndexSpec doesn't structurally satisfy `Convention` — its `from_dict`/`to_dict` shape differs, same decorator-injection root cause). Typing-only, runtime fine, tests green. Check when tackling the debt whether it's CDSIndexSpec-specific or several consumers; fixing the decorator path (option A) should clear it. Do NOT loosen the `Convention` bound to silence it — the bound is what made `data_registry` itself mypy-clean.

---

## 🏝️ DECIDE — orphaned-but-hardened core modules (0 production consumers)

**Surfaced across the core/ pass (2026-06-28 → 07-05).** These modules are coherent, tested, and now hardened, but **nothing in production imports them** — complete infrastructure waiting to be wired in (or removed):
- `core/dependency_graph.py` (v1.221) — DAG for incremental risk. **User said KEEP** (believes it's used inside the library; grep finds no consumer — keep as infra).
- `core/mandate.py` (v1.230) — buy-side IPS compliance engine. Needs a `Portfolio → PortfolioHolding` adapter to be usable (EOD + pre-trade checks).
- `core/settlement.py` (v1.235) — physical/cash/auction settlement framework. Needs wiring into instruments' settlement flows.
- `core/caching.py` (v1.212–214) — NullCache/DictCache/LRUCache/LazyValue catalogue; the `TestNoCacheClones` guard tracks whether it earns its place. (Related: dead `mc_greeks_auto.PathCache` already removed v1.213.)

**Decision needed (per module):** on a near-term roadmap → keep + note the intended consumer; else → flag for removal (dead code is debt). Not urgent, but shouldn't sit as silent orphans. `dependency_graph` = keep (decided).

---

## 🔧 MINOR — pyright not pointed at `.venv` (repo-wide)

`npx pyright <file>` reports `Import "numpy"/"scipy" could not be resolved` on every file — pyright isn't using the project venv, so the editor shows spurious `reportMissingImports` repo-wide. Fix: add a `pyrightconfig.json` (`{"venvPath": ".", "venv": ".venv"}`) or point the editor's Python interpreter at `.venv`. Environmental, not a code bug; deferred (config choice for the user).

---

## 📐 QUEUED (L1 pass) — multi-D interpolation review + 3D consolidation

**Surfaced 2026-07-04** while reviewing `core/interpolation.py` (1D, L0 — reviewed + hardened at v1.228, non-finite guard added). The multi-D story is unreviewed and uneven:

- **2D** — `numerical/_interpolation.py` (L1): `bilinear`, `bicubic` (scipy `RectBivariateSpline`), `interpolate_2d`, `rbf_interpolate` (RBF scattered-data). **Not yet reviewed.** Almost certainly carries the same **non-finite-input gap** just fixed in 1D (and possibly more — a quick probe hit an arg-signature that needs a proper read). Natural next file when the pass reaches L1. Review for the same clarity/consolidation + guards lens.
- **3D — no central primitive.** Vol cubes (swaption vol cube, FX smile cube, IR vol surface, inflation smile) each handle the third axis themselves (nested 1D/2D or scipy `RegularGridInterpolator` directly). So any hardening (finite guards, extrapolation conventions) is repeated per-consumer or absent. **Consolidation question:** introduce a single tested/guarded 3D interpolator in `numerical`, or leave distributed? Decide during the L1 pass.

**When reached:** review `numerical/_interpolation.py` first (2D), then take the 3D-consolidation decision. Both fit the "clarity and consolidation" round.

---

## 0d · ACTIVE PLAN — approximation module "perfect" hardening (12 phases)

**Goal:** take `core/approximation.py` + `numerical/_spectral.py` from "right architecture, accepted warts" to bulletproof — verification that makes silent misuse and silent drift impossible. Born from a multi-pass audit (v1.188 guards → v1.189 dedup → v1.190 correctness fixes → v1.191–1.194 structural cleanup) + a testing-strategist failure-modes-first plan.

**Root principle:** build the verification net FIRST, then make risky changes against it. Cost is mostly tests, not source (~12 slices ≈ 36 commits). Each phase = one slice, full-suite-gated, phase-by-phase with a stop after each.

**Bug-classes to make impossible** (all four historical bugs were "test-shaped-wrong"): symmetric-interval, midpoint-only eval, sign-invariant checks, loose tolerances, defining-contract-never-tested. Every new test breaks symmetry on every axis: asymmetric `[a,b]` (a≠−b), non-even/non-odd f, off-center off-node query, tolerance pinned to true method accuracy.

### Stage 1 — Verification net (additive tests, ~zero risk)
- **P1 ✅ v1.195.0** — add `hypothesis` (dev extra, derandomized CI profile) + Chebyshev asymmetric reproduce-at-nodes property + `SpectralResult.evaluate` mirror property. Kills symmetric/midpoint class.
- **P2 ✅ v1.196.0** — Padé **order-(L+M) matching** property + closed-form `exp` diagonal golden coeffs + near-singular threshold (just-above/below). Kills silent-truncation class.
- **P3 ✅ v1.197.0** — BVP **manufactured solution**, variable-coeff operator `L = D2 + p(x)D + q(x)I`, tol ≤ 1e-8, off-center eval. Kills operator-specific class.
- **P4a ✅ v1.198.0** — Richardson `order`-drives-cancellation both-directions + `best_estimate==corner`; B-spline partition-of-unity **sweep** + right-endpoint behavior + cubic golden vs scipy.
- **P4b ✅ v1.199.0** — Gauss exactness boundary (2n−1 exact / 2n not), `D2@x²=2`, clamp idempotency, `legendre` weight-sum invariant. **← Stage 1 (verification net) COMPLETE.**

### Stage 2 — Contracts (code+tests, low risk)
- **P5 ✅ v1.200.0** — pin scalar-return contract (scalar→`float` across all 4 `evaluate`s; kills the `np.atleast_1d` smell) + cross-check `chebyshev_interpolate` vs `chebyshev_expand`.
- **P6 ✅ v1.201.0** — `Approximant` Protocol + `to_dict` mixin + one conformance test over all result types (surfaces lossy `SpectralResult.to_dict`). **← Stage 2 (contracts) COMPLETE.**

### Stage 3 — Engineering (med risk, against the Stage-1 net)
- **P7 ✅ v1.202.0** — B-spline iterative **de Boor** (kills O(2^degree) recursion).
- **P8 ✅ v1.203.0** — Richardson explicit per-column **orders** API (geometric-series assumption → opt-in).
- **P9 ✅ v1.204.0** — edge-case + **complexity docstrings** on every function.
- **P10 ✅ v1.205.0** — **re-divergence CI guard**: test fails if a duplicate `chebyshev_*`/diff-matrix def reappears or an upward import lands. **← Stage 3 (engineering) COMPLETE.**

### Stage 4 — Risky perf (highest blast radius; last, against full net)
- **P11 ✅ SKIPPED (2026-06-30, evaluated)** — **FFT-based DCT** gate FAILED. `scipy.fft.dct(type=1)` is the exact analytic equivalent but NOT bit-identical: across 1,400 random cases (n≤80) max abs diff 5.9e-15, max **rel diff 2.8e-10** (cancellation on near-zero coeffs). Deliberately not done: perf win irrelevant at n≤50 (microseconds, no pricer bottlenecked), and a ~6e-15 perturbation across ~20 downstream modules + a new `scipy.fft` coupling on the core hot path is not worth it. No code change. The O(n²) loop stays; the docstring already notes the FFT option.

### Stage 5 — Cleanup
- **P12 ✅ v1.206.0** — delete/replace misleading existing tests (`test_scalar_evaluation` symmetric-midpoint, `test_arbitrary_interval` abs=0.5 on a kink) — AFTER replacements land so coverage never dips.

### ✅ PLAN COMPLETE (v1.195–v1.206, 2026-06-30). All phases done (P11 deliberately skipped). Full suite 13,062 pass. The approximation/Chebyshev cluster is hardened end-to-end.

### Post-plan follow-ons (2026-06-30 → 07-01):
- **v1.207** — quadrature de-dup: `spectral_integrate` delegates to canonical `_integrate.integrate` (Gauss-Legendre single-sourced).
- **v1.208** — new technique: **barycentric Lagrange** (arbitrary-node interpolation).
- **v1.209** — new technique: **Remez minimax** (best polynomial; fail-loud on non-convergence; Chebyshev-basis result).
- **v1.210** — new technique: **Hermite** (values + derivatives, confluent divided differences).
- **v1.211** — **quadrature re-divergence guard** (P10 analogue for Gauss-Legendre). → module now exposes **8 techniques** under one hardened contract; each new one dropped in with zero friction.

### ⬜ WATCH / TRIGGERED — split `core/approximation.py` into an `approximation/` subpackage
- **Trigger:** file is at **686 LOC / 8 techniques** as of v1.211 — right at the ~500 LOC / ~8-technique split threshold. Adding **1–2 more** techniques flat is fine; **past that, split.**
- **Action when triggered:** `approximation/{chebyshev,interpolation,rational,extrapolation,basis}.py` (+ the kernel), with `__init__.py` re-exporting the current public names so no caller changes. Move `Approximant` protocol + `_ResultToDict` mixin to `approximation/_base.py`.
- **Safety net:** the structural guards (single-home defs, no-upward-import, conformance) + full suite carry the refactor; the single-home guard's expected paths (`core/approximation.py`) must be updated to the new module paths.
- **Not urgent:** nothing is wrong today; the file is navigable (clean section markers). This is the only architectural decision on the horizon for the module.

**Out of scope (Tier C — confirmed counterproductive):** swap to `numpy.polynomial.chebyshev` (different node convention, wide blast radius, no correctness gain); exhaustive adversarial fuzz on every primitive.

**Tests to strengthen (fold into the family phases):** `test_polynomial_exact` / `test_exp_accurate` (→ asymmetric), `test_interpolate_sin`/`_exponential` (tol 0.01→1e-10), `test_diff_matrix_shape` (demote to smoke), `test_partition_of_unity` (single-point→sweep), `test_diagonal_improves` (endpoints→full diagonal), `test_exp_pade_22` (rel=1e-3→golden coeffs).

---

## 0e · ACTIVE — caching abstraction (contract + no-cache baseline)

**Done — Phase 0 (v1.212.0):** `core/caching.py` reshaped from 3 unused concrete classes into a **`Cache` protocol** (`get_or_compute(key, compute)`) + **`NullCache`** (the "without caching" baseline) + reference impls `LRUCache`/`DictCache` (+ `LazyValue`). A confrontation test (cache result == NullCache result) ships as a reusable cache-honesty check. Additive; no production imported the old `CurveCache`/`CalibrationCache`.

**Why:** caching in the codebase is heterogeneous *by design* (lru_cache for pure fns, dict for immutable memo, bespoke OrderedDict LRU for MC paths) — a survey showed forcing one class would be over-engineering. The value is the **isolation seam**: any component can take a `Cache`, and injecting `NullCache` proves the cache isn't hiding a staleness/aliasing bug.

**Phase 1 — client migration: REASSESSED & effectively closed (v1.213).** The premise (a live MC path cache) was false:
- `models/mc_greeks_auto.PathCache` — was **DEAD** (never wired into `auto_greeks`, 0 callers, misleading docstring). **Removed in v1.213.0** — you can't inject `NullCache` into a cache nothing uses.
- `pricing/market_data_provider._cache` — real opt-in cache but the provider has ~0 external callers (notebook/test-level); low value.
- `core/calendar._holiday_cache` — genuinely live, but a correct minimal dict over **immutable** per-year keys → a stale/aliasing bug is impossible by construction, so the `NullCache` confrontation is moot. Leave as-is.
- `curves/curve_bumper` — `functools.lru_cache`, pure fn — stdlib is right. Leave.
- **Verdict:** no existing cache needs retro-migration.

**v1.214 — design decision: caching is a CATALOGUE, not an abstraction.** The v1.212 `Cache` Protocol had zero consumers (nothing type-hinted it) → stripped. `core/caching.py` is now an à-la-carte catalogue of utilities (`NullCache`/`DictCache`/`LRUCache`/`LazyValue`) sharing the duck-typed `get_or_compute(key, compute)` convention — no base class/Protocol. Functional-first (stdlib `functools` for pure-fn memo). A guard (`TestNoCacheClones`) flags any `*Cache` class defined outside the catalogue = the "clones appeared → reconsider abstraction" trigger. **§0e done.** Reconsider a shared class/type only if that guard starts firing (real clones).

---

## 0c · ACTIVE PLAN — calibration "capture-not-reconstruct" migration (healthy structure for N future calibrations)

**Goal:** a structure that survives many new *kinds* of calibration with minimal per-calibrator code and records that are **impossible to lie** (no fabricated convergence, no empty-residuals-as-perfect, no eager/lazy drift). Born from the G1–G9 fidelity audit + an independent clean-slate design (app-designer) + the head-to-head comparison.

**Root diagnosis:** the record is currently built *at a distance* from the optimiser, so convergence/residuals get re-derived and drift (the eager `calibrate()` path and the lazy `_build()` fallback are two builders that disagree). The fix closes that distance: capture a `SolveReport` *from the solver*, pass it to *one* builder.

**Strategy decision (locked): in-place, additive-layers-first — NOT parallel `new_`.**
- One record type, one sink (`save_calibration`), one gate set → forking with `new_` doubles schema/serialisation/gates for a single-user single-DB system and tends to ossify. The existing gates ARE the migration safety net.
- The highest-value change (capturing `SolveReport`) is purely *additive new layers*, so no fork is needed. Add new layers, migrate producers onto them in place, then restructure the shared core.
- **Payoff comes after Phase 0–1, not after all 15 migrate:** once the solve layer + builder exist, every NEW calibration uses the clean pattern from day one; the existing 15 migrate opportunistically.

**Adopt (from clean-slate design):** solver-primitive `SolveReport` layer; single `model_calibration_record()` builder; fused `Residuals` value object (empty unconstructible); typed per-family `Diagnostics` (kill the `extra{}` junk drawer); `ParamDigest` for fitted surfaces (FX-SLV).
**Reject:** the `Provenanced[T]` wrapper — keep field-on-curve + `ProvenanceCarrier` protocol for Family A (the Optional leak is cheaper than `.split()` churn across hundreds of curve call sites). **So Family A's 18 bootstrappers stay nearly untouched; work concentrates on the solve layer + Family B's 15 calibrators.**
**Validated as already-correct:** the 4-component record shape (provenance/fit/optimiser/diagnostics) and `model_class` as a string key — an independent designer kept both.

### Phases (each phase = green, full-suite-gated; slice discipline per phase)
- **Phase 0 ✅ v1.157.0 — `calibration/_solve.py` (additive, zero risk).** `SolveReport` (algorithm/converged/iterations/tolerance/seed) + `analytic()`/`external()` + the 5 primitives the Family-B calibrators actually use: `minimize_solve` (SABR/HW/joint/dividend/dispersion), `least_squares_solve`, `global_local_solve` (DE+polish, captures seed: G2++/jump), `brentq_solve` (JY), `particle_solve` (seeded MC: FX-SLV). 7 tests pin "captured, not invented". ✅
- **Phase 1 ✅ v1.158.0 — `model_calibration_record()` builder (additive).** Family-B mirror of `curve_calibration_record`. Both families now assemble through one factory each; no hand-rolled skeletons. ✅
  - **→ After Phase 1 the "16th calibration" is clean** (primitive → SolveReport → builder, ~6 lines). Structure now survives new implementations.
- **Phase 2 ✅ v1.159-1.161 — migrated all 15 calibrators (SABR ref + Group A + Group B).** Route each through a primitive, populate the record once from the captured `SolveReport`, **delete the lazy `_build` fallback**. Retires the eager/lazy duality + fabricated-converged per-producer. Eager+lazy both through the builder; dispersion→analytic(), multicurve faithful. ✅
- **Phase 3 ✅ v1.162.0 — type-level empty-fit rejection + typed reconstructed.** Fused `Residuals` (real G1 fix); kill `diagnostics.extra` → typed `Diagnostics`. Contained because every producer already routes through the builder. ⬜
- **Phase 4 ✅ v1.163.0 — builder-enforcement gate; multicurve eager site closed. MIGRATION COMPLETE.** Delete lazy fallbacks / magic thresholds / `record_source` strings; add "`SolveReport` only constructed by primitives" grep-gate. ⬜

**Two assumptions to confirm while building:** (1) every fit has ≥1 real target — FX-SLV's residual is a placeholder `0.0`, so its "target" is fake (digest + typed ParticleDiagnostics is what actually saves it); (2) the ~15 calibrators can route through ~5 primitives — today SABR/HW/G2PP/jump/fx_slv call scipy/MC directly, so wrapping them IS the Phase-0/2 work.

**Prereqs/no-conflict:** builds on the completed §0b (gates exist) and v1.156 (fidelity sweep). Independent of §0 (refactoring pass).

### ✅ MIGRATION COMPLETE through v1.168.0 (13,012 tests pass; calibration is **L1**, on core/L0)
Phases 0–4 done + tri-state `converged` (v1.164) + FX-SLV `ParamDigest` (v1.165) + primitive deletion (v1.166) + 2 review passes that found/fixed real bugs (v1.167 fx_slv converged; v1.168 tuple-`extra` round-trip + `list_calibrations(None)` + 3 gate hardenings). `ARCHITECTURE.md` written, accuracy-checked, layer-reconciled. Reading order: `L1_DEPS.md`.

### Optional / remaining (NOTHING structural — none block anything)
**A — optional do-now (clear, bounded):**
1. **Human-eye reading pass** for calibration (the §0 activity, now unblocked) — order in `L1_DEPS.md`: `_solve` & `_types` → `_curve_record` & `_model_record` → `__init__`.
2. ✅ DONE — **`L0_DEPS.md … L7_DEPS.md` all generated** (whole tree, dependency order). Regenerate via `tools/layer_deps.py --layer N --write`.
3. **Populate the unused `CalibrationDiagnostics` fields** — `objective_history` / `parameter_history` / `timing_ms` are filled by nobody (only `warnings`/`reconstructed`/`extra` used). `timing_ms` is a cheap win; else accept as optional-by-design.

**B — deferred, do ONLY if the trigger fires (rejected as gold-plating; triggers recorded):**
- `Provenanced[T]` wrapper for curves → trigger: Optional-handling churn at curve call sites becomes painful.
- Fused `Residuals` value object → trigger: a 4th parallel per-quote array appears, or read sites desync.
- Per-family typed `Diagnostics` subclasses → trigger: ≥5 families need the *same* diagnostic field.
- `Bootstrapper`/`Calibrator` process class → trigger: a multi-currency curve-building *service* needs polymorphic select/run.

**C — larger separate effort (market-data flow, NOT a provenance gap):**
- **G3 full market-snapshot wiring.** `market_snapshot_id` is complete + threaded by the 9 producers that receive a `MarketSnapshot`; the rest take raw quotes, not a snapshot. Threading snapshots into those callers is a market-data-flow effort — do only if audit-by-snapshot coverage matters more than today.

---

## 0b · ✅ COMPLETE (v1.152.0) — calibration provenance full sweep (curves + calibrators, both gated)

**DONE — both sides conformance-gated.**

- **Bootstrapper side** (`test_bootstrapper_provenance_conformance.py`, v1.150–v1.151): every public `bootstrap*`/`*_bootstrap` curve producer attaches a canonical `CalibrationResult`. 18 COVERED in the behavioural registry + 3 bond-hazard tested elsewhere; ALLOWLIST holds only `bootstrap_ci` (statistical, not a curve). v1.151 closed the last two `global_solver` producers: wired `coupled_bootstrap` (was a genuine gap — dual-curve solve returned both curves with no record) and normalised `global_bootstrap` off its hand-rolled `CalibrationResult` onto the helper.
- **Calibrator side** (`test_calibrator_provenance_conformance.py`, v1.152): all 13 `CanonicalCalibrationResult` mixin subclasses are discovery-classified + structurally checked (field + `_build_calibration_record` override) + behaviourally built (8 here, 5 heavy deferred to their dedicated tests).

A new bootstrapper or calibrator added without provenance now fails CI on both sides. Tier history below retained for reference.

**Goal:** every public curve/survival bootstrapper attaches a canonical `CalibrationResult` to the curve it returns (`curve.calibration_result`), so all curve calibrations are auditable/persistable — uniform with the 2 already wired. NOT a mixin job (curves are the artefact; the mixin is for family-result *types*).

**Current state (verified 2026-06-22):**
- ✅ Producing records: `curves/bootstrap.bootstrap` (discount_curve_bootstrap), `curves/global_solver.global_bootstrap` (discount_curve_global), `credit/bond_hazard_bootstrap` (on the mixin). These 2 curve ones are the "correctly separate" pattern — keep as-is.
- ⬜ ~13 produce **no** provenance. Delegation map:
  - **Delegate to a wired core** (inherit for free, verify only): `curves/rfr_bootstrap.bootstrap_rfr` → `bootstrap`.
  - **Delegate to an UN-wired core**: `fixed_income/ibor_curve` → `bootstrap_forward_curve` (which itself attaches nothing).
  - **Build their own curve** (need wiring): `fixed_income/ois`, `fixed_income/tenor_basis`, `fixed_income/xccy_basis`, `fixed_income/rfr.bootstrap_spread_curve`, `curves/bond_curve`, `curves/aad_curves`, `credit/cds`, `credit/cds_market`, `credit/sovereign_cds`, `fixed_income/inflation.bootstrap_cpi_curve`, `fixed_income/inflation_bond_advanced.real_yield_curve_bootstrap`, `equity/dividend_advanced.dividend_curve_bootstrap`.

**Foundation first (do before any waterfall):**
- **F1 — shared helper.** Generalise the existing `curves/bootstrap._build_bootstrap_calibration_result` into a reusable `curve_calibration_record(*, reference_date, pillar_dates, model_class, residuals, quotes, algorithm, iterations, converged, extra=…)` → `CalibrationResult`. Every bootstrapper calls it; avoids 13× hand-rolled construction. (1 slice.)
- **F2 — curves can carry provenance.** Confirm each curve type holds `calibration_result: CalibrationResult | None = None`: `DiscountCurve` ✓ (already). Check/add for `SurvivalCurve` (credit) and `AADDiscountCurve` (aad). (1 slice if either is missing.)

**Waterfall (target core files first, then cascade by asset class — 1 slice each):**
- **Tier 1 — core curve primitives:** `bootstrap_forward_curve` (covers `ibor_curve` by delegation); verify `bootstrap_rfr` inherits.
- **Tier 2 — foundational rates:** `bootstrap_ois`, `bootstrap_spread_curve`.
- **Tier 3 — basis / specialised rates:** `bootstrap_tenor_basis`, `bootstrap_basis_curve` (xccy), `bond_curve`, `aad_curves`.
- **Tier 4 — credit:** `cds` (bootstrap_credit_curve), `cds_market` (from_upfronts), `sovereign_cds`.
- **Tier 5 — inflation/equity:** `bootstrap_cpi_curve`, `real_yield_curve_bootstrap`, `dividend_curve_bootstrap`.

**Per-slice shape:** compute per-pillar round-trip residuals, build the record via `curve_calibration_record(...)`, attach to the returned curve, add a test that `curve.calibration_result` round-trips + persists via `db.save_calibration(curve.calibration_result)`.

**Closing gate (last slice):** a conformance test that imports every public bootstrapper, runs a small fixture, and asserts the returned curve carries a non-None `calibration_result` (with an explicit allowlist for any deliberately-excluded ones) — turns "all curves auditable" from convention into an enforced invariant, mirroring the mixin's ABC/field guard.

**Scope:** F1+F2 (≤2) + Tiers 1–5 (~13) + gate (1) ≈ **16 slices**. Residual-unit/model_class conventions inherit the `CalibrationFit.__post_init__` enforcement already in place (snake_case + length agreement), so no new validation needed.

**Open questions to resolve at execution:** does `bootstrap_rfr` return the `bootstrap()`'d curve verbatim (true inherit) or rebuild it? does `SurvivalCurve`/`AADDiscountCurve` already have the field? do `cpi`/`real_yield`/`dividend` return curve objects that can carry the attribute?

**Bugs surfaced by the campaign:**
- ✅ **B-rfr-global** (found Tier 1, fixed v1.143.0): `bootstrap_rfr(method="global")` passed `deposit_day_count`/`fixed_day_count`/`fixed_frequency` to `global_bootstrap`, whose params are `deposit_dc`/`swap_dc`/`swap_frequency` → `TypeError` on any global call. Fixed the kwarg names; xfail removed (test now passes normally).

**Progress:** F1 ✅ v1.140.0 (helpers). F2 ✅ v1.141.0 (SurvivalCurve/AADDiscountCurve fields). Tier 1 ✅: `bootstrap_forward_curve` attaches `projection_curve_bootstrap` record; `IBORCurve` + `RFRCurveResult` forward `.calibration_result` from their inner curve (ibor + rfr-sequential inherit). Tier 2 ✅: `bootstrap_ois` (v1.144) + `bootstrap_spread_curve` (v1.145). Tier 3a ✅ v1.146.0: `bootstrap_basis_curve` (xccy, `xccy_basis_bootstrap`) + `aad_bootstrap` (`aad_discount_curve_bootstrap`). Tier 3b ✅ v1.147.0: `bond_curve` (`BondCurveResult.__post_init__` injects `bond_curve_bootstrap` onto the discount curve, all 5 methods; wrapper forwards `.calibration_result`) + `tenor_basis` (`tenor_basis_bootstrap` on long-tenor projection, forwarded by `IBORCurve`). Tier 4 ✅ v1.148.0: `bootstrap_credit_curve` (`credit_curve_bootstrap`; `build_cds_curve` inherits by delegation), `bootstrap_from_upfronts` (`cds_upfront_bootstrap`), `bootstrap_sovereign_hazard` (`sovereign_hazard_bootstrap`, `SovereignHazardResult` forwards; tracked `fitted_tenors` to keep parallel arrays aligned past the `dt<=0` skip). Tier 5 ✅ v1.149.0: added the `calibration_result` field to `CPICurve`/`RealYieldCurveResult`/`DividendCurve` (F2-equivalent; the two dataclasses' `to_dict` now drop the field) and attached records — `cpi_curve_bootstrap`, `real_yield_curve_bootstrap`, `dividend_curve_bootstrap`. **Next: the closing conformance gate — a test that imports every public bootstrapper, runs a fixture, and asserts the returned curve carries a non-None `calibration_result` (with an explicit allowlist for any deliberate exclusions).**

---

## 0 · ACTIVE WORK ITEM — layer-by-layer refactoring pass

**Status (2026-06-19):** L0 dependency tree generated (`L0_DEPS.md` at repo root), 113 modules verified complete. **Reading not yet started.**

**What:** User-driven, file-by-file code-quality review of the library, walking each layer from L0 to L6 in depth order (leaves first → roots last) per the `L<N>_DEPS.md` ordering. Focus is clean-code review (SOLID, Fowler smells, readability) — **NOT** correctness (the audit chain closed that at v1.119).

**Why human-driven:** earlier agent passes were observed to skip parts. User wants to read every file themselves.

**Where to resume:**
1. Open `L0_DEPS.md` at repo root.
2. Start with `calibration` sub-package (2 modules — trivial warm-up).
3. Then proceed in the recommended order: market_data → db → pe → core → statistics → numerical → ts → viz.
4. Within each sub-package, read modules in the order they appear (depth 0 leaves first, then depth 1, etc.).

**Tool to regenerate any layer's tree:**
```bash
.venv/bin/python tools/layer_deps.py --layer 0 --write    # → L0_DEPS.md
.venv/bin/python tools/layer_deps.py --layer 1 --write    # → L1_DEPS.md  (not yet generated)
# ... up through --layer 6
```

**Findings during the read** → become slices via the standard 3-commit flow (code+test / stamp / release notes). Anything caught that's not slice-shaped (a question, a held-as-is decision, a smell to revisit) goes here under sections 1-4 below.

When all layers L0-L6 are read and slices landed, this section closes.

### 0a · Calibration-result unification (surfaced reading L0/calibration, 2026-06-20)

Reading `calibration/_types.py` surfaced that the canonical `CalibrationResult` is half-unified and **build-and-drop**: ~10 producers, but production reads only `.id`; never serialised/persisted; `to_calibration_result()` has zero production callers; 8 bespoke `*CalibrationResult` types never adopted it; 2 files (`credit/rating_models.py:34`, `models/calibration_utils.py:19`) shadow the name; the `Calibrator` Protocol has zero implementers.

**Decision (user):** finish the unification, **consumer axis first** (make the record load-bearing before widening producers).

Roadmap:
- **Phase 0 — make it load-bearing**
  - ✅ Slice 1 (v1.120.0): `CalibrationResult`/`OptimiserSpec`/`CalibrationDiagnostics` serialisable + tz-aware injectable clock; UUID/datetime/dict/tuple atoms added to `core.serialisable`.
  - ✅ Slice 2 (v1.121.0): `calibration_results` table + `save_calibration`/`load_calibration`/`load_calibration_raw`/`list_calibrations` on `PricebookDB`. Denormalised audit-chain columns consume model_class/timestamp/objective/converged/iterations/rms/max/market_snapshot_id. Phase 0 COMPLETE.
- **Phase 1 — kill contradictions:**
  - ✅ 1a (v1.122.0): deleted the dead `Calibrator` Protocol (zero implementers; only its own test used it).
  - ✅ 1b (v1.123.0): renamed the 2 name-shadows → `GeneratorCalibrationResult` (credit/rating_models), `RobustCalibrationResult` (models/calibration_utils). `class CalibrationResult` is now unique. Phase 1 COMPLETE.
- **Phase 3 — wire shut:** ✅ 3a (v1.124.0): `db.save_calibration` polymorphic (accepts family results via `to_calibration_result()`); loop build→store→read closed & proven; `to_calibration_result()` now has a real consumer. *(Done before Phase 2 by design.)*
- **Phase 2 — widen producers** (~1 slice each onto the proven `to_calibration_result()` pattern):
  - ✅ 1/6 (v1.125.0): `lmm_advanced` — renamed `LMMCalibrationResult`→`RebonatoLMMCalibrationResult` (2nd name-shadow) + widened. Builders populate per-swaption residuals.
  - ✅ 2/6 (v1.126.0): `jarrow_yildirim.JYCalibrationResult` (builder-populate, per-tenor residuals).
  - ✅ 3/6 (v1.127.0): `dividend_calibration.DividendCalibrationResult` (lazy-cache, faithful residuals).
  - ✅ 4/6 (v1.128.0): `joint_equity_credit.JointCalibrationResult` (builder-populate, weighted_sse, relative residuals).
  - ✅ 5/6 (v1.129.0): `fx_slv_calibration.ParticleCalibrationResult` (builder-populate, SV-config params; placeholder residual flagged → C.4).
  - ✅ 6/6 (v1.130.0): `stochastic_correlation.DispersionCalibrationResult` (lazy-cache). **Phase 2 COMPLETE — all 12 families produce the canonical record.**
  - (Excluded: `GeneratorCalibrationResult` — matrix; `RobustCalibrationResult` — internal helper.)

### Calibration unification — re-assessment follow-ups — ALL CLOSED (Phase 4, v1.131–v1.134)
- ✅ **C.1 — Phase 4 mixin (v1.131.0):** `CanonicalCalibrationResult` extracted; adopted by all 12 families (net −11 lines). Unified the two variants (builder-populate / lazy-cache) into one `to_calibration_result()` + `_build_calibration_record()`. The abstraction the deleted `Calibrator` Protocol never was.
- ✅ **C.2 — model_class overlap (v1.132.0):** `lmm_advanced` → `"lmm_rebonato"`; `lmm_calibration` keeps `"lmm"`.
- ✅ **C.3 — optional field (v1.131.0):** folded into the mixin; kept optional, the mixin handles stored-or-lazy uniformly. Consumer (persistence) wired in Phase 3a, so "optional" is now safe.
- ✅ **C.4 — fx_slv residual bug (v1.133.0):** replaced the `* 0.0` placeholder with real local-vol reproduction error. (If fx_slv becomes load-bearing, run `numerical-critic` on the residual definition.)
- ✅ **C.5 — _types.py coherence (v1.134.0):** `rms_residual`/`max_residual` now derived `@property`s (no drift); unweighted-rms semantics documented; serialisation schema bumped to v2.

**Calibration unification: COMPLETE (v1.119 → v1.134, 16 slices).** Original objection fully resolved; structure, naming, loop, and metrics all clean.

---

## 1 · Open & pickable as small slices

Each is one slice (code+tests / stamp / RELEASE_NOTES) under the existing flow. Pick in any order.

### O.1 — Remove `strict_icma` flag entirely
**Source:** `.archive/audit-v1.119/AUDIT_L0_CORE.md` LD.1 (lines 768-775).
**What:** v1.110 flipped the default `strict_icma=False → True`. The trigger condition was "the final-slice commit will rip the flag out entirely." Still pending: drop the flag from `year_fraction()` and `_act_act_icma()` signatures, remove the 3 legacy-contract `strict_icma=False` opt-ins in tests, delete the silent-fallback code paths.
**Risk:** low. No production caller passes the flag. Three legacy-contract tests document the dropped behaviour — delete them (or rewrite as "ValueError is raised").
**Effort:** 1 slice.

### O.2 — Verify or delete NS dual implementation
**Source:** memory `recurring-bug-patterns.md` pattern #4.
**What:** `curves/nelson_siegel.py` AND `curves/curve_advanced.py` both ship Nelson-Siegel formulas. Memory flags "status unclear post-sweep; verify before assuming closed." Investigate: is one dead? Is one a wrapper of the other? If duplicate, delete one; if both live with different semantics, document the split.
**Risk:** low (investigation) then low-medium (deletion if applicable).
**Effort:** 1 investigation pass + possibly 1 delete slice.

### O.3 — Verify or delete key-rate-DV01 dual implementation
**Source:** memory `recurring-bug-patterns.md` pattern #4.
**What:** Same shape as O.2 — `curves/key_rate_risk.key_rate_dv01` AND `curves/curve_bumper.key_rate_dv01s` both exist. Same investigation + decision.
**Risk:** low.
**Effort:** 1 investigation pass + possibly 1 delete slice.

### O.4 — `bond_hazard_bootstrap.py:1063` TODO
**Source:** `python/pricebook/credit/bond_hazard_bootstrap.py:1063` (in-code).
**What:** `dc_adj = discount_curve  # TODO: bump for liquidity`. Add a liquidity-spread bump to the discount curve before pricing the hazard bootstrap. Currently uses raw discount curve — fine for clean cases, slightly off for liquid issuers vs illiquid.
**Risk:** low.
**Effort:** 1 slice if specified; 1 design conversation first if the bump's parameterisation is open.

---

## 2 · Held by design — leave alone

Each carries an explicit rationale. Re-open only if the rationale stops applying.

### H.1 — A.11 B3 (registry duplicate-registration warning)
**Source:** `.archive/audit-v1.119/AUDIT_L0_CORE.md` line 28.
**Status:** HELD-AS-IS. Audit suggested adding a DeprecationWarning on re-register; current behaviour silently ignores duplicates (via `if key and key not in _REGISTRY`), which the held-as-is decision judged safer.
**Re-open if:** a real bug surfaces from a duplicate registration going un-noticed.

### H.2 — LD.3 (FixedRateBond `_ytm_time_to` fallback paths)
**Source:** `.archive/audit-v1.119/AUDIT_L0_CORE.md` LD.3 (lines 785-794).
**Status:** defensive only — none of the standard pricer paths hit the fallback. Three branches: `coupons_per_year is None` (WEEKLY), `target` not in `coupon_dates`, `settle` outside the coupon range.
**Re-open if:** a real caller actually trips one of those branches in production.

### H.3 — LD.4 (silent import-failure recording in `_ensure_loaded`)
**Source:** `.archive/audit-v1.119/AUDIT_L0_CORE.md` LD.4 (lines 796-802).
**Status:** CI test `test_auto_discovery_succeeds_with_no_import_failures` asserts the failures list is empty. Currently empty. The silent-record behaviour exists so an import failure in a peripheral module doesn't crash the entire serialisation layer.
**Re-open if:** a peripheral-module import failure actually happens — then decide if it should escalate to `warnings.warn` or hard-error.

### H.4 — LD.5 (multi-period ACT/ACT ICMA spans not handled)
**Source:** `.archive/audit-v1.119/AUDIT_L0_CORE.md` LD.5 (lines 804-809).
**Status:** `_act_act_icma` only handles single-period spans (the only case `FixedLeg` needs). `FixedRateBond._ytm_time_to` solves the multi-period case by bypassing `year_fraction` entirely.
**Re-open if:** a future caller needs `year_fraction(settle, payment, ACT_ACT_ICMA)` over a multi-period span with refs supplied. Then extend `_act_act_icma` to implement ICMA 251.2's stub-plus-full-period decomposition.

---

## 3 · Deferred multi-slice — schema v2 bundle

These are NOT individually picked. They coordinate into a single bump (call it pricebook v1.0 / Gate 1.5 / schema-v2-cut), as documented in `.archive/audit-v1.119/AUDIT_L0_CORE.md` "When is the breaking full migration worthwhile?" section (line 857+).

The bundle contains:

| Item | What |
|---|---|
| LD.1 flag removal | Rip out `strict_icma` parameter entirely (already in O.1; could be pre-bundled) |
| LD.6 | `DiscountCurve.from_dict` defaults to `LOG_LINEAR` when key absent — remove the shim, require explicit |
| LD.7 | `PricingContext.from_dict` back-compat empty-dict defaults — remove the shims, require explicit |
| LD.8 | `_check_schema_version` absent → v1 silently — require explicit version |
| LD.9 | `serialisable_convention.from_dict` accepts both flat AND envelope format — pick one canonical |
| LD.10 | ~42 hand-rolled `to_dict` overrides in options/credit/desks don't use `make_payload`/`read_payload` helpers (don't emit `schema_version`). Concrete file list in archived `AUDIT_L0_CORE.md` line 845. Per-module migration. |

**Trigger conditions to launch the bundle** (any one):
- Shim count crosses ~25 (interactions become painful — currently 10 active shims)
- Audit chain catches a real divergence the shims hide
- A new feature requires the post-migration shape
- Major release boundary (v2.0, C++ port, external publishing)

**Don't launch the bundle just because the list feels long.** Each shim has a clear trigger; until one fires there's no benefit.

---

## 4 · Optional next moves (not queued, no deadline)

These are the natural next gains if you want to keep raising the floor. None are catalogued bugs — they're the next-quality-bar steps.

### N.1 — Per-sub-package clean-code pass
**What:** Fowler-named smells per sub-package using the `clean-code-expert` agent. The audit chain's ponytail pass caught obvious over-engineering only. SOLID-named smells (Long Method, Feature Envy, Primitive Obsession, Data Clumps, etc.) were NOT systematically hunted. A senior reviewer doing a fresh pass would still find things.
**Cadence:** one sub-package at a time; the agent picks the smells, the user picks which to apply.
**Recommended order:** start with the layers that have the most lines per function or the most legacy — likely `pricing/` or `models/` first.

### N.2 — Numerical re-derivation of high-impact pricers
**What:** Run the `numerical-audit` workflow (11-lens methodology) on specific pricers. W2/W3/W5 in the warnings sweep just demonstrated three real bugs the audit chain walked past. There will be more.
**Cadence:** one pricer at a time, before any critical release that depends on it.
**Recommended starting points:** anything in `MODULE_HEALTH.md` (archived) that had Tier-1/2/3 findings — re-derive against the textbook.

### N.3 — Test gap audit
**What:** Coverage gaps not surfaced by audit findings remain un-charted. The audit added one regression test per closed bug; that's bug-specific coverage, not coverage of legitimate but un-bugged code paths.
**Cadence:** one sub-package at a time, possibly using `testing-strategist` agent for the strategy then writing tests by hand.

### N.4 — Resume pre-audit product roadmap
**What:** The archived `PLAN_FUTURE.md` (in `.archive/plans-pre-audit/`) lists product slices 89-99+. Some may have been built since v0.88, some may not. A pass through to identify what's actually missing would be useful before adding new product work.
**Cadence:** one investigation pass, then per-slice work as picked.

---

## 5 · How to keep this file alive

After each slice that touches an open item:
1. Update the item here (mark closed, edit status, link to RELEASE_NOTES entry).
2. Update the underlying source if applicable (memory file, archived AUDIT doc).
3. Don't let this file go stale — it's the single tracker; if it diverges from reality, the next session loses the map.

When a new item surfaces (in code, in a session, from a code review):
- Add it under the right section (1-4) with: source, what to do, risk, effort.
- Don't add ad-hoc TODO files anywhere else.

When in doubt: read `AUDIT_PLAN.md` §7 for campaign history, `RELEASE_NOTES.md` for closure record, this file for what's still open.
