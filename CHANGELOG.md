# Changelog

All notable changes are recorded here, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. `pricebook.__version__` is the
single source of truth and CI asserts it matches the top entry.

## [Unreleased]

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

### Oracles — 25 green
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

### Fixed — found by Cowork's independent review, before landing
- **The schema version was written and never read.** Both `to_dict`s stamped `schema_version` and
  **nothing dereferenced it**; the test asserted only that it was *present*. A payload from a
  future build would have loaded silently under today's code — defeating the purpose the field's
  own docstring claims. The project's named defect shape, **resolve-don't-just-exist**, and a
  presence check could never have caught it. Fixed with **one** shared `check_schema_version` both
  `from_dict`s call, plus six oracles asserting **refusal**: a future version, a missing one, six
  malformed ones, the nested payload, and the *ordering* (version before `kind` — in an unknown
  format the `kind` field's meaning is itself unknown). Serialisation-in-the-first-product was
  ratified *conditioned on this*: a contract that will not enforce its own version is not one yet.

### Changed
- **`numerics` threading DROPPED from `dispatch`.** Purified from the stock as-is, it inspected
  each pricer's signature to decide whether to pass an optional numerical config — and **ran in no
  test**, since the only pricer is closed-form. Three reasons agree: the config's own rule is that
  it *lands with the first engine that reads it*; there is no consumer; and an unexercised dispatch
  branch at the baseline is the same dead-code shape the debt gate caught one commit earlier.
  It returns with the first pricer that integrates or samples.
- **`CalibratedModel` not carried.** A protocol declares a capability, and a discounting model has
  none to declare beyond carrying its market, so carrying it would be an empty abstraction. The
  registry's model parameter is `object` until a pricer needs a real capability.
- **`DiscountCurve.zero_rate` uses `bisect` and carries no suppression.** The first draft ended in
  an unreachable `raise` needing `# pragma: no cover`, which the debt gate caught. The answer to
  dead code is to remove the deadness, not to ledger the suppression — and `bisect` is O(log n),
  which will matter at a bootstrapped curve.
- `TEST_FLOOR` 1 → 25.

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
