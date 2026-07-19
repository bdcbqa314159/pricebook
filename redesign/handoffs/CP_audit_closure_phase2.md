# Checkpoint — foundation-audit closure, end of Phase 2

**Branch:** `fix/foundation-audit-closure`. **Versions:** Phase 0 `0.75.0` (package) · Phase 1 `0.76.0`
(Tier-1) · Phase 2 `0.77.0` (Tier-2). **147 L0 tests, `verify.py all` + ruff + pyright green.**
Per the handoff I **stop here** — Phase 3 is structural (CalendarProtocol + retype sweep) and the plan
is below for approval before I start.

---

## What landed

**Phase 0 — package it.** `foundation/__init__.py` with explicit `__all__` (93 exports), `py.typed`,
root `pyproject.toml` (`pip install -e .` works), and `calendar.py → calendars.py` (stdlib shadow).
Unblocks the project's own suite as an importable package.

**Phase 1 — 5 Tier-1 bugs (red→green).** 1.1 AFB leap-to-leap (3.9973→4.0, direct year-shift) · 1.2
`LOG_LINEAR`+`CONTINUE_SLOPE` negative DF (−0.275→+0.209, ruling **A1** log-space) · 1.3 BUS/252 vs CDI
window mismatch (one `[start,end)` primitive, ruling **A2**, S16 withdrawn) · 1.4 backward schedule
roll-day drift + EOM-on-wrong-end (anchor-based `anchor±k·tenor`, EOM on the generation seed) · 1.5
furikae `sorted()` (defensive — the substitute union is order-invariant, so no answer changed).

**Phase 2 — 4 Tier-2 bugs (red→green).** 2.1 SOFR plain compounded (no `observation_shift`) · 2.2
SIFMA gains Good Friday + `SUNDAY_ONLY` observance (2021-12-31 now open) · 2.3 LONDON one-offs via a new
`dates()` combinator + `since/until` on `nth` · 2.4 Tokyo astronomical `equinox()` + Emperor's-Birthday
`since=2020`.

---

## Review input 1 — Oracle quality
Every Tier-1/2 fix rests on a **red test carrying the audit's hand-calculated expected value**, verified
against published sources: ISDA/AFB whole-years; log-space DF continuation; ANBIMA `[start,end)`; ISDA
§4.10 EOM; ARRC SOFR-OIS convention; SIFMA/Treasury open-days incl. Good Friday and 2021-12-31; UK
2022–23 gazette bank holidays; QuantLib astronomical-equinox approximation. No fix rests on
self-consistency alone.

## Review input 2 — Quarry-drawdown reconciliation
Unchanged (Topic 0 parked 13/793). This is a correctness/packaging pass on live L0, not a migration —
no parking delta.

## Review input 3 — "Challenge me"
1. **SIFMA observance → `SUNDAY_ONLY` and I did NOT split into NYSE/Fed-bank calendars** (the audit's
   full 2.2 recommendation). NYSE/EFFR have **zero consumers**, so adding them now is speculative (§6b);
   I made `NEW_YORK_SIFMA` correct-for-rates and deferred the split to the first equity/EFFR consumer.
   Contest if you want the empty calendars stubbed now.
2. **Three `test_calendars` tests were rewritten**, not just repaired — they encoded the buggy federal
   Sat→Fri observance for SIFMA. Corrected to the SUNDAY_ONLY truth (2021-12-31 open; Juneteenth 2021
   Sat not shifted). A reviewer should confirm SIFMA really is Sat-not-shifted (audit + 2021-12-31
   evidence say yes).
3. **1.5 furikae has no red counterexample** — I proved the substitute *union* is order-invariant
   (greedy next-free-slot), so `is_business_day` never actually flaked. The `sorted()` fix is defensive
   + matches the definition; the test is a correctness pin (2020 Golden Week). Disposition = fixed, not
   a wrong-answer.
4. **`_equinox_day` approximation** (QuantLib's) is valid for a bounded year range; far-future years
   drift. Acceptable for the markets we price; a Tier-4 note if we ever need >2100.

## Review input 4 — Smell + debt scan
`verify.py debt` green — **0 suppressions**, no `# type: ignore`. The one PLR0913 introduced (`_step_k`,
6 args) was fixed by folding `forward` into a signed `k`, not suppressed. `fields`/`layers`/`provenance`
green; ruff + pyright clean; format-stable.

## Review input 5 — Spine conformance
All fixes stay L0 finance-free (calendars/day-counts/schedules are conventions, not valuation). The new
`dates()`/`equinox()` are rule-DSL siblings; `CalendarProtocol` (Phase 3) is the one structural type and
is called out for approval below. No upward imports; DAG intact.

---

## ►► Phase 3 plan (structural — approve before the retype sweep)

Order = most-expensive-to-retrofit first. Each is red→green where a behaviour is testable.

1. **3.2 `CalendarProtocol` (FIRST).** Define `CalendarProtocol` (`is_business_day`, `adjust`,
   `add_business_days`, `identity`); implement the missing `adjust`/`add_business_days` on
   `JointCalendar` (trivial — they only call `is_business_day`); **retype every consumer slot**
   (`RollRule.calendar`, `year_fraction`, `settlement_date`, `accrued_rate`) from `Calendar` to the
   protocol. This is the retype sweep — the reason to stop and confirm first.
2. **3.3 Registries.** Public `register_calendar`/`register_rate_index`; **all** `register_*` raise on
   conflicting re-registration (`register_currency` silently overwrites today); a test-isolation helper.
3. **3.4 Serialisation convention.** Identities (Calendar, RateIndex) serialise **by name**; atoms
   (Money, Accrual, Cashflow) **by value**. Fix in foundation so each upper layer doesn't reinvent it.
4. **3.5 `Schedule` provenance.** Per-period records (start, end, `is_stub`, ICMA anchors, payment
   date); explicit stub-boundary dates on `ScheduleTerms`; **raise on long stubs** until ACT/ACT ICMA
   supports multiple quasi-coupon periods; decide where `payment_delay` applies (dead code today).
5. **3.6 FX spot.** `fx_spot_date(pair, trade_date)` with intermediate-day + joint-centre + USD checks;
   move `spot_lag` out of `CurrencyPair` equality into a pair-conventions registry.
6. **3.7 `_denominator` raise** on unsupported day counts (silent 360 fallback today).
7. **3.7 `FixingSource` protocol** — `accrued_rate` accepts a `FixingSource`; `FixingHistory` stays the
   trivial impl.

**Then Phase 4** (apply the Part A field cuts + doc-wording corrections) and **Phase 5** (ledger Tier 4
→ `OPEN.md`, close the three reports to `closed_*.md`).

**Open question for the ruling:** 3.5 is large (Schedule redesign + ScheduleTerms fields + ICMA long-stub
raise). Confirm you want it in this pass, or split it to a follow-up so Phase 3 stays a focused
protocol/registry/serialisation sweep.
