# Checkpoint — foundation-audit closure, end of Phase 3 (for ruling)

**Branch:** `fix/foundation-audit-closure`. **Version:** `0.78.0`. **154 L0 tests, `verify.py all` +
ruff + pyright green.** This is the report per redesign/11 — the thing to be ruled, not the green suite.

Phase 3 landed the Tier-3 structural items in the approved order (3.2 CalendarProtocol → 3.3 registries
→ Q1 SIFMA rename → 3.5 long-stub raise → 3.6 FX spot → 3.7 `_denominator`/`FixingSource`). The
per-item detail is in `CHANGELOG.md` v0.78.0; this checkpoint is the five review inputs.

---

## Review input 1 — Oracle-quality audit (the substantive one)

**"154 tests green" is worth less than it looks, and here is the measurement.** I classified every L0
test by *where its expected value comes from*: **ORACLE** = derivable without running the code (closed
form, published value/date, external cross-check, required contract, math invariant); **SELF-CONSISTENCY**
= asserts what the code currently produces.

**Honest tally (skeptical pass): ~140 / 152 ORACLE (~92%), ~12 SELF-CONSISTENCY (~8%).**

> A first automated pass returned 97.4% oracle. I do not trust it and did not use it: it classified by a
> test's *apparent subject* ("names ISDA §4.10 → oracle") rather than by whether the number/tuple was
> actually derived externally. That is precisely the failure mode below.

**The self-consistency set** (assert what the code does; no external anchor):
- registry counts — `test_all_37_markets_declared`, `test_all_37_currencies_declared` (assert a count the
  code produces).
- structural / field-existence — `test_decomposed_by_method_family`, `test_full_decomposition_vocabulary_optional`,
  `test_each_sibling_reports_its_asset_class`, `test_commodity_identity_carries_location_and_grade`
  (assert the shape the source declares).
- tautology / regression-against-self — `test_accrual_year_fraction_matches_primitive` (wrapper == the
  primitive it calls), `test_standard_frequencies_unchanged` (named "behaviour-preserving").
- schedule stub tuples — `test_short_vs_long_front_stub` / `_back_stub`: the interior roll dates are
  derivable, but *which date the merge drops* is implementation-defined, so the pasted tuple is partly
  self-consistency.

**The finding the ratio exists to surface (item 2 of the ruling).** Three tests were asserting WRONG
behaviour until this closure: SIFMA Sat→Fri (`test_us_vs_uk_saturday_divergence`), EOM-on-start
(`test_non_eom_start_preserves_day`), spot_lag-in-equality (`test_currency_pair...`). Each *looked* like
an oracle test — it named a convention — but its expected value had been read off the implementation, so
it **ratified the code instead of checking it**, and passed green against a bug for the life of the
module. Written alongside the code they test, they are self-consistency wearing an oracle's clothes. The
lesson is not the count; it is the **discipline**: the bar for a new test is that the expected value could
be written down, from an external source, *before* the code exists. The remaining ~8% are mostly benign
(counts, shape) — but the category is where bugs hide, so new L0 tests are held to the external-value bar.

The genuinely load-bearing oracles are strong: ISDA §4.16 / ICMA 251 worked examples (UST coupon =
2.0000), published holiday/observance dates, CME IMM tables, astronomical equinoxes, √2 / Φ closed forms,
round-trip identities, PCHIP no-overshoot, and required contracts (currency-mixing raises, registries
reject conflict). No fix in Phases 0–3 rests on self-consistency.

## Review input 2 — Quarry-drawdown reconciliation
**13 / 793 parked, unchanged.** The audit closure is a correctness + packaging + structure pass on the
already-parked L0; it retires no new quarry module and adds no parking. Drawdown is reported, not moved.

## Review input 3 — "Challenge me"
1. **FX spot is a defensible simplification**, not the full ACI algorithm (counts `lag` days good in both
   centres, then enforces both-centres + USD-for-cross on the value date; USD-holiday-on-path is
   skip-not-count). Adequate for spot value dates; flag if split-settlement conventions ever bite.
2. **Q1 rename deferred the NYSE/Fed-bank siblings** (still no consumer, §6b). The *trap* is disarmed (the
   generic name is gone); the siblings arrive with equity/EFFR.
3. **`temporary_*` helpers mutate the backing** behind the frozen view — sanctioned mutation; the view
   still rejects direct writes.

## Review input 4 — Smell + debt scan
- `verify.py debt` green — **0 suppressions**. One `# noqa` slipped in during 3.2 and was removed (the
  gate caught it); the one PLR0913 (`_step_k`) was fixed by a signed `k`, not suppressed.
- **NEW, logged now (item 1 of the ruling): NG-DEFER-1 — the ACT/ACT ICMA long-stub refusal.** It is a
  *refusal that Phase 3b removes* — a load-bearing temporary by construction. Per CLAUDE.md §5 it is in
  `OPEN.md` (deferred-scope ledger) **now**, with re-open trigger **Phase 3b**, not after 3b lands — so
  that if 3b slips, the refusal cannot silently read as a design decision. (It is not a suppression, so it
  is not in the `verify.py debt` balance, which counts only `# type: ignore`/`# noqa`/`# pragma`/`skip`;
  it is documented deferred scope.) **Finding 3.5 is NOT discharged** and `AUDIT.md` stays unprefixed
  until 3b deletes the raise.
- `acyclic`/`fields`/`layers`/`provenance` green after `settlement → market_calendars` (no cycle).

## Review input 5 — Spine conformance
L0 stays finance-free. `CalendarProtocol`/`FixingSource` are capability protocols (depend on the
capability, not the class). FX spot is settlement-date machinery in `settlement.py`. DAG intact; no upward
imports.

---

## Next (executing without waiting, per the ruling)
**Phase 3b** (Schedule provenance: per-period records · ICMA multi-quasi-period so NG-DEFER-1's refused
case computes · `ScheduleTerms` stub-boundary fields · `payment_delay` placement) → **serialisation**
(over the settled shapes) → **Phase 4** (A3 field cuts + A1/A2 doc corrections) → **Phase 5** (ledger
Tier 4 → `OPEN.md`; close reports to `closed_*.md` only when every finding is fixed-with-test or ledgered
— NG-DEFER-1 must be discharged first). Reports stay unprefixed until then.
