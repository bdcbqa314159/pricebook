# Checkpoint — foundation-audit closure, end of Phase 3

**Branch:** `fix/foundation-audit-closure`. **Version:** `0.78.0`. **154 L0 tests, `verify.py all` +
ruff + pyright green.** Phase 3 (Tier-3 structural) done in the approved order. Next per the ruling:
**Phase 3b** (Schedule provenance) → serialisation → Phase 4 → Phase 5.

---

## What landed (approved order)

1. **3.2 CalendarProtocol** — `is_business_day`/`adjust`/`add_business_days`/`identity`, shared arithmetic
   mixin, `JointCalendar` now a full calendar, **every consumer slot retyped** to the protocol.
2. **3.3 registries** — public `register_calendar`/`register_rate_index`; **all** `register_*` raise on
   conflict; `temporary_*` context managers.
3. **Q1 SIFMA rename** — `NEW_YORK_SIFMA → US_GOVERNMENT_SECURITIES`, SOFR bound explicitly, generic name
   gone (the "one USD calendar" trap disarmed). NYSE/Fed-bank siblings still deferred (arrive with
   consumers).
4. **3.5 long-stub RAISE** (kept in Phase 3 per amendment) — ACT/ACT ICMA refuses a long stub rather than
   returning a wrong single-period number.
5. **3.6 FX spot** — `fx_spot_date` (T+2 joint-calendar; T+1 USD/CAD; a cross cannot settle on a US
   holiday); `spot_lag` out of `CurrencyPair` identity into a pair-conventions registry.
6. **3.7** — `_denominator` raises on unsupported day counts; `FixingSource` protocol on `accrued_rate`.

---

## Five review inputs

1. **Oracle quality.** Each behaviour-testable item has a red→green test: JointCalendar adjust across a
   two-centre Christmas; conflict-raise on every registry; SOFR↔`US_GOVERNMENT_SECURITIES` binding; ICMA
   long-stub raise; FX spot T+2 / T+1 / cross-on-MLK (2024-01-15 → 01-16); `_denominator` raise;
   `FixingSource` duck-typed source. Structural retypes (CalendarProtocol) are proven by pyright + the
   cross-currency schedule building.
2. **Drawdown.** Unchanged (Topic 0 parked 13/793) — correctness/structure pass on live L0.
3. **Challenge me.** (a) **FX spot is a defensible simplification**, not the full ACI algorithm — it
   counts `lag` days good in both centres then enforces both-centres + (cross) USD on the value date;
   the "USD holiday on the counting path" nuance is handled as *skip-not-count*. Adequate for spot value
   dates; a Tier-4 note if exotic split-settlement conventions ever bite. (b) **`temporary_*` helpers use
   the mutable backing** behind the frozen `MappingProxyType` view — registration is the sanctioned
   mutation, the view still refuses direct writes. (c) **`_SPOT_LAGS` T+1 list** (USDCAD/TRY/PHP/RUB) is
   the common set; more can be added as a data edit.
4. **Smell + debt.** `verify.py debt` green — 0 suppressions. One `# noqa` slipped in and was removed
   (the debt gate caught it). The one PLR0913 (`_step_k`) was fixed by a signed `k`, not suppressed.
   `acyclic` green after `settlement → market_calendars` (no cycle). fields/layers/provenance green.
5. **Spine conformance.** L0 stays finance-free; `CalendarProtocol`/`FixingSource` are capability
   protocols (depend on the capability, not the class); FX spot is settlement-date machinery in
   `settlement.py`. DAG intact.

---

## Phase 3b plan (next)

**Schedule provenance** (the split you approved, landing INSIDE this closure before any `closed_`):
per-period records (start, end, `is_stub`, ICMA reference anchors, payment date); explicit stub-boundary
dates on `ScheduleTerms`; **ACT/ACT ICMA multi-quasi-period support** (so the long-stub raise from 3.5 can
compute the previously-refused case); decide where `payment_delay` applies (a different calendar than
fixing for XCCY). Then **serialisation convention** over the settled `Schedule`/`ScheduleTerms`/
`CurrencyPair` shapes, then **Phase 4** (A3 field cuts + doc-wording corrections A1/A2), then **Phase 5**
(ledger Tier 4 → `OPEN.md`; close the three reports to `closed_*.md` only once every finding is discharged
— 3.5 is NOT discharged until 3b lands, so the reports stay unprefixed until then).
