# Changelog

All notable changes are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `pricebook.__version__` is the
single source of truth and CI asserts it matches the top entry.

## [Unreleased]

## [0.2.0] - 2026-10-31

### Added — tranche 2: the inflation spine (ZCIIS, model-free)
The zero-coupon inflation-indexed swap, priced with **no dynamics** — pure no-arbitrage from the
foreign-currency analogy (real ↔ foreign, nominal ↔ domestic, index ↔ FX spot). Source:
**`BERHAN-2012`** §1.2.2. Proposal, review and results: `docs/tranches/02_inflation_spine.md`.

- **L0** `InflationIndex` — the asset class's *entire* L0 footprint, a bare identity ·
  **`Term(start, maturity, day_count)`**, which exists because the contract needs six loose fields
  and the rule's answer is to bundle the cohesive ones, never to raise the limit. It also puts the
  compounding exponent in one place: `TM` is measured from **inception**, since the fixed payout
  was fixed at trade time and does not shrink as the trade seasons.
- **L1** `CurveRole.REAL` — **a new curve KEY, not a new snapshot field**: a second discount curve
  costs one enum member while `MarketSnapshot`, `CurveSet` and every existing pricer stay as they
  were · `MarketSnapshot.index_fixings` · **`real_curve_from_breakeven`**, eq 1.2.16.
- **L2** `ZeroCouponInflationSwap`, serialised and version-enforced.
- **L4** `price_zc_inflation_swap` + `inflation_leg_value`, composing `df` from **both** curves.

### The design call: there is exactly ONE `Pr`
`Pr` is a curve in the snapshot; eq 1.2.16 is **how it is built** from break-even quotes, not a
second way to price. The pricer never recomputes it. Two oracles guard that, with different jobs:
the **par identity** proves construction and pricing share one `Pr`, and a **pointwise** 1.2.16
check proves the constructor is right on its own.

### Oracles — 37 green
- **1.2.14** off curves the test specifies, to 1e-9 · **the exact reduction**: 1.2.13 with
  `I(t) = I0` *is* 1.2.14, so there is one formula and not two · the index enters as a **ratio**
  (double it and the real-bond term doubles while `−Pn` does not — a difference would shift by a
  constant) · the net swap equals leg **minus** fixed leg, so the net PV cannot become its own
  formula · zero-inflation degenerate case · sign (the receiver gains on higher break-even, loses
  on a higher strike) · matured → PV 0 · missing real curve / missing fixing → failure · round-trip
  with the version enforced.

### Fixed — a must-fix found a second bug, in the API rather than the test
- **`real_curve_from_breakeven` now takes quote DATES, not year fractions.** Cowork's must-fix
  (the par oracle telescopes, so the constructor needs its own pointwise oracle) made that oracle
  **fail at the 2.5-year pillar by 6.6e-6**: a caller saying "2.5 years" converts to 912 whole
  days — 2.4986 years on the curve's own measure — so the caller's `T` and the curve's `T`
  disagreed by a day. Fixed structurally: one `date → t` map, the curve's, and a caller cannot
  hold a second. **The par oracle could never have seen this**, because a one-day disagreement
  shared by construction and pricing telescopes away — which is precisely the must-fix's argument,
  demonstrated.
- **A matured swap returns PV 0, not `PricingFailure`** (must-fix 2). Conflating *historical* with
  *unpriceable* would make a settled trade look broken; checked before any market lookup, so a
  matured trade needs no curves.

### Changed
- **No lag field on `InflationIndex`, and the stock is why.** Its `inflation_indices.py` carries one
  `publication_lag_months`: 3 for most indices, **8 for UK index-linked gilts**. Those are two
  quantities conflated — UK RPI publishes ~1 month after its reference month, and the 8 months is
  the **indexation lag the gilt's contract specifies**, which two linkers on the same index can set
  differently. So the indexation lag is L2 contract data (arriving with the linker in T3), and even
  the publication lag is omitted here because **nothing reads it** — the snapshot carries the
  already-resolved `I(t)`. Ratified by Cowork, which confirmed the 8-month figure against the stock.
- **Conventions register gains C7** (Cowork's review condition): the break-even construction, the
  nominal curve's `TimeMeasure` and the contract's `Term.day_count` share **one** day count, and the
  par identity holds only under it. Recorded **OWED**, because *no test could fail on it* — the par
  oracle telescopes when the conventions match, which they always do today. Sharpened while
  recording it: with one day count in the library the convention is not merely unguarded but
  **unviolable**, so a guard now would be an imagined problem. Trigger: the first second day count.
- **`# crosscheck: quantlib …` markers** at both sites wanting external validation (thread 0014),
  so the later optional QuantLib tier has a floored worklist instead of a from-memory sweep.
- `TEST_FLOOR` 25 → 37.

## [0.1.1] - 2026-10-31

### Fixed — a defect in v0.1.0, found by Cowork's independent review
- **The schema version was written and never read.** Both `to_dict`s stamped `schema_version` and
  **nothing dereferenced it**, so a payload from a future build would have loaded silently under
  v0.1.0 code — defeating the exact guarantee the field's own docstring claims ("a format with no
  version cannot change without silently breaking a reader"). It was therefore not merely an unused
  field but a **false claim in a comment**, which is the worse of the two. The project's named
  defect shape, **resolve-don't-just-exist**, and v0.1.0's test asserted only *presence*, which
  could never have caught it.
  Fixed with **one** shared `check_schema_version(payload)` that both `from_dict`s call — one
  definition of what is readable, so the two products cannot drift — returning the validated
  version so a future migration branches on it rather than re-reading the raw field.
- **Six oracles that assert REFUSAL, not presence:** a future version, a missing one, six
  malformed ones (`"1"`, `0`, `-3`, `1.5`, `True`, `None`), the nested payload, and **the
  ordering** — the version is checked *before* `kind`, because in a format we cannot read the
  `kind` field's meaning is itself unknown, so reporting "not a fixed_cashflow" would be a guess
  about a payload we have already failed to understand.

### Changed
- **`numerics` threading dropped from `dispatch`.** Purified from the stock as-is, it inspected
  each pricer's signature to decide whether to pass an optional numerical config — and **ran in no
  test**, since the only pricer is closed-form. Three reasons agree: the config's own rule is that
  it *lands with the first engine that reads it*; there is no consumer; and an unexercised dispatch
  branch at the baseline is the same dead-code shape the debt gate caught in `zero_rate`. It
  returns with the first pricer that integrates or samples — with a consumer and a test.
- `TEST_FLOOR` 18 → 25.

### Process note
v0.1.0 merged before this review round landed, so the fix ships as a patch rather than as part of
the tranche. The sequence Cowork set is **propose → review → land**; here it ran propose → build →
land → review → patch. No harm done (the defect was caught and is fixed in the next commit on
`main`), but the review's value is highest *before* a merge, which is where the beat belongs.

## [0.1.0] - 2026-10-31

### Added — tranche 1: the minimal spine
A single fixed cashflow priced end to end through the stateless
`price(product, model) → PricingResult`. The backbone every product needs and none can avoid, so
it prejudges no product while proving the engine contract. Proposal and results:
`docs/tranches/01_minimal_spine.md`.

- **L0** `Money`/`Currency` (currency mixing is an error, never a silent addition) ·
  `DayCountConvention` + `year_fraction` (ACT/365F only) · `TimeMeasure` (the one `date → t` map) ·
  `Cashflow` · `PricingResult`/`PricingFailure`.
- **L1** `DiscountCurve` + the `CurveHandle` protocol the engine depends on · `CurveSet`/`CurveKey`
  (curves keyed, not positional) · `MarketSnapshot`, which **carries the valuation date** — there
  is no clock read anywhere in the library.
- **L2** `FixedCashflow` — pure data, with `to_dict`/`from_dict` from the first product.
- **L3** `DiscountingModel`, carrying the snapshot it was built on.
- **L4** the engine registry (structural dispatch, no `isinstance` ladder) + `price_fixed_cashflow`.

### Oracles — 18 green
- **`PV = N·e^{−r·t}` exact to 1e-12** across 48 points. The reference is arithmetic written in the
  test, independent of every path under it — which is the point: the predecessor shipped a 5.14%
  error whose oracle restated the builder's own composition and so validated it against itself.
- **Analytic vs finite-difference DV01** to 1e-6 — two different computations, so agreement is not
  tautological.
- The **valuation-date boundary** in both directions: a flow *on* the date is excluded, one *day
  after* is included and positive. Without the second, "excluded" could be implemented as "always
  zero" and the first would still pass.
- Statelessness (byte-identical reprice, snapshot untouched); failure-is-a-value on three distinct
  failures; currency mixing refused; the representation round-trips and carries a schema version.
- **Interpolation tested on a SLOPED curve.** A flat curve returns one rate everywhere and cannot
  tell linear interpolation from any other scheme — the fixture trap the predecessor hit twice.

### Stock
11 of 13 modules **purified** from `_stock/`, 2 built fresh, nothing over 100 lines. Purification
was mostly *trimming*: `money.py` 289 → 90 lines (the open-currency *insight* kept, the registry
machinery left behind until something needs named constants); `day_count.py` 332 → 45 (one
convention); `snapshot.py` 154 → 30 (curves only — its other shapes grew one asset class at a
time and return when a product pulls them).

### Changed
- **`CalibratedModel` not carried.** A protocol declares a capability, and a discounting model has
  none to declare beyond carrying its market, so carrying it would be an empty abstraction. The
  registry's model parameter is `object` until a pricer needs a real capability.
- **`DiscountCurve.zero_rate` uses `bisect` and carries no suppression.** The first draft ended in
  an unreachable `raise` needing `# pragma: no cover`, which the debt gate caught. The answer to
  dead code is to remove the deadness, not to ledger the suppression — and `bisect` is O(log n),
  which will matter at a bootstrapped curve.
- `TEST_FLOOR` 1 → 18.

## [0.0.0] - 2026-10-30

### Added
- **The clean project scaffold.** An empty, CI-green library at `v0.0.0` — no pricing code.
  Package `pricebook`, MIT, public. `pyproject.toml`, one trivial test, the ubuntu+windows CI
  matrix on Python 3.12, and `verify.py` carried from the migration tree and **trimmed to the
  gates that still apply**.
- **The stock.** The whole previous tree — the built core, the un-migrated `quarry/`, its `docs/`
  and all 77 oracle test files — shelved in `_stock/` and **gitignored**: reference material,
  mined by hand for *content*, never for *structure*, never imported. Its pushed backup is the
  archived `pricebook-migration` repo at tag `migration-archive`.
- **`docs/handoffs/phase_R_record.md`** — what Phase R did, the three parameters Bernardo ruled
  before execution, and the four `verify.py` calls taken under delegated mechanics.
- **`docs/conventions_register.md`** carried as a tracked living doc, because it describes the
  built core rather than the migration — including its one **OWED** item, the vol clock, which
  the rebuild re-lays *with* its cross-module oracle.

### Changed
- **The migration frame is retired** (`docs/handoffs/rulings_reset_to_fresh_repo.md`). The unit of
  work is now a product you want, pulling in the core it needs; `done` is no longer "the quarry is
  empty". The architecture, the oracle discipline and the conventions are kept; the obligation to
  cross every old module is dropped.
- **`verify.py` trimmed:** the `quarry` import-floor gate and the drawdown reconciliation are
  **deleted** — both measured migration progress, and there is no migration to measure. Kept:
  `acyclic`, `layers`, `fields`, `debt`, `provenance`, `version`, `slow`, `tests`.
- **Provenance header: `quarry:` → `stock:`**, same role against the new reference tree. Because
  `_stock/` is gitignored and therefore absent in CI, the resolve-check **reports** that it could
  not dereference paths rather than skipping silently — a gate that quietly passes is this
  project's named defect class. `source:` and `oracle:` are verified in both environments.
- **License corrected to MIT** in `pyproject.toml`, matching the `LICENSE` file (the old tree's
  metadata said Proprietary).
