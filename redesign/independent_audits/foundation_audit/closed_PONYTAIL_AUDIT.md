# Ponytail Audit — Foundation Over-Engineering Scan

**Date:** 2026-07-19
**Scope:** `/Users/bernardocohen/work/analysis/foundation/` — 16 modules, ~3,215 lines
**Pass:** Complexity only. Correctness bugs, security, and performance are out of scope for this pass (covered separately in `AUDIT.md`).
**Method:** Whole-tree read, ranked biggest cut first. This is a report — it applies nothing.

---

## Verdict

The codebase is genuinely lean in its **algorithms** — the day-count math, the holiday-rule DSL, and schedule generation are tight, with no reinvented stdlib and no removable dependencies. The over-engineering is concentrated in one theme the code itself keeps admitting in its docstrings: **types built ahead of the layers that will consume them** ("defined now, populated later"). That is rung 1 of the ladder — does this need to exist *yet* — and since the next step is to build those very layers (market-data, models, trade, product), the re-add cost is one file-touch, versus carrying frozen guessed fields that dependents will lock in.

---

## Findings (biggest cut first)

### 1. `yagni:` underlying.py sibling identities
`ReferenceEntity`, `InflationIndex`, `FxFixing`, `EquityUnderlying`, `CommodityUnderlying`, and the `InflationInterp` enum have **zero consumers** and guessed fields (`fixing_time: str`, `grade: str`, `exchange: str` with no identifier scheme).
**Cut:** keep the `Underlying` protocol and the `AssetClass` enum (RateIndex genuinely uses both); delete the five sibling dataclasses and `InflationInterp`. Re-add one when its asset class actually ships.
**Where:** `underlying.py:47-119` (~66 lines)

### 2. `yagni:` numerical_config.py in full
Every knob — `sobol`, `brownian_bridge`, `cos_n`, `cos_l`, `tree_steps`, and the whole `LatticeConfig` / `IntegrationConfig` — configures engines that do not exist. Nothing in the tree reads any of it. Worse, serialized schema v1 **freezes the knob names before any engine has validated them**.
**Cut:** ship this file with the first engine that needs it.
**Where:** `numerical_config.py` (114 lines)

### 3. `yagni:` results.py speculative fields
`sensitivities`, `cashflow_breakdown`, and `diagnostics` are populated by an L5 model layer that does not exist; `DiscountBasis` is a frozen dataclass wrapping a single `Currency | None`.
**Cut:** keep `pv` / `accrued` / `clean`; inline `basis: Currency | None` directly on `PricingResult`; drop the three empty fields until an engine fills them.
**Where:** `results.py:26-49` (~15 lines)

### 4. `shrink:` `_ANNUAL_BASIS` dict + `_annual_basis()`
A one-entry dict `{BUS_252: 252}` behind a lookup-and-raise helper.
**Cut:** inline as `252 if dc is BUS_252 else raise`.
**Where:** `rate_index.py:121-130`

### 5. `shrink:` `RfrConvention.none()`
Returns `cls()` — byte-for-byte identical to the default `RfrConvention()`, which other call sites already use directly.
**Cut:** delete the classmethod; call `RfrConvention()`.
**Where:** `rate_index.py:80-83`

### 6. `yagni:` `exponential_growth()`
A public fixed-rate LTN/NTN-F primitive with **no caller** in the tree (`accrued_rate` handles the daily series itself).
**Cut:** add it when the bond product needs it.
**Where:** `rate_index.py:133-139`

### 7. `shrink:` `Calendar._rule_set` property
`__post_init__` guarantees `self.days` is always a `HolidaySet`, so the union-narrowing property is dead defense.
**Cut:** use `self.days` directly at the two call sites.
**Where:** `calendar.py:251-256`

### 8. `shrink:` `Currency.value` / `Unit.value` alias properties
Return `self.code` / `self.symbol` "for call sites that read `.value`".
**Cut:** grep first; if no such call site survives, delete both; if one does, rename it away.
**Where:** `money.py:83-85, 139-141`

---

## Kept deliberately (not cuts)

- **`Currency` / `Unit` `ClassVar` annotation blocks** (46 lines mirroring the declaration loop) — the only thing giving `Currency.USD` autocomplete and type-checking. The docstring already argues the tradeoff. Leave them.
- **`JointCalendar`** — no consumer yet, but a real near-term cross-currency need. That is an incompleteness bug (it can't `adjust`), not bloat.
- **`distributions` / `solvers` / `interpolation` scipy adapters** — earn their thinness as the single C++-port swap point.
- **Dependencies** — scipy and numpy are all genuinely used. Nothing removable.

---

## Net

**−220 lines possible, −0 deps.**

The theme is uniform: delete the speculative output / config / identity types now and let each layer scaffold its own when it arrives. You are about to build those layers anyway — the re-add cost is one file-touch, versus carrying frozen guessed fields that dependents will lock into place.

---

## Closure disposition (2026-07-19)

Closed on branch `fix/foundation-audit-closure`. Items 1–3 (the speculative types) landed as the ratified
**A3** cuts in Phase 4 (v0.80.0); the micro-shrinks 4/5/8 in commit `b2fba8e0`. Two are **rejected with
reason** (kept) — a rejection is a valid disposition when the finding's premise does not hold here.

| # | finding | disposition |
|---|---|---|
| 1 | `underlying.py` sibling identities | **FIXED** v0.80.0 — deleted (A3); `Underlying` protocol + `AssetClass` kept |
| 2 | `numerical_config.py` in full | **FIXED** v0.80.0 — deleted; ships with the first engine that reads it |
| 3 | `results.py` speculative fields | **FIXED** v0.80.0 — `sensitivities`/`cashflow_breakdown`/`diagnostics` + `DiscountBasis` dropped; `basis` inlined to `Currency \| None` |
| 4 | one-entry `_ANNUAL_BASIS` dict | **FIXED** `b2fba8e0` — inlined as a direct `BUS_252` conditional |
| 5 | `RfrConvention.none()` | **FIXED** `b2fba8e0` — deleted (byte-identical to default); the `_term` factory names the intent |
| 6 | `exponential_growth()` | **REJECTED (kept).** A verified closed-form primitive with its own oracle and no guessed schema — the A3 schema-freeze hazard (the actual basis for cutting 1–3) does not apply; it is the fixed-rate counterpart to the already-present CDI daily-series machinery, with a near-term EM-bond consumer. Deleting a tested pure formula to re-add it is churn. |
| 7 | `Calendar._rule_set` property | **REJECTED (kept).** Premise ("dead defense") is false under the actual pyright config: the `days` field is a declared union (`HolidaySet \| tuple[Rule,...]`, so a bare rule tuple can be passed), and the property's narrowing is load-bearing for type-checking at the two call sites. |
| 8 | `Currency.value`/`Unit.value` aliases | **FIXED** `b2fba8e0` — deleted; zero surviving call sites (grep-verified) |
