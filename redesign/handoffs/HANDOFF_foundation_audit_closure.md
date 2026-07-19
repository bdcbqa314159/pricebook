# HAND-OFF — close the independent foundation audit

Source: `redesign/independent_audits/` — `AUDIT.md` (4 parallel adversarial reviews),
`PONYTAIL_AUDIT.md` (over-engineering), `PONYTAIL-DEBT.md` (deferral ledger).

**Verdict from the audit, worth repeating:** *"The design is genuinely good — better than QuantLib
on the axes it targeted — but it is not ready to build layers on."* Nothing needs redesign. One
round of fixes.

---

## PART A — Cowork rulings on the conflicts (read first)

The audit contradicts three earlier design rulings. **The audit is right in all three.** These
supersede the design docs; the docs are corrected as part of this work.

### A1. `CONTINUE_SLOPE` extrapolation — my C4 ruling caused a real bug
Gate item S12/C4 said *"extrapolation policy stated per end (`FLAT | CONTINUE_SLOPE | RAISE`)"* but
never said **in which space**. Result (audit 1.2): `LOG_LINEAR` + `CONTINUE_SLOPE` extrapolates in
*value* space and returns **DF(30) = −0.275** — a negative discount factor.
**Ruling: `CONTINUE_SLOPE` operates in the interpolation's own space.** For `LOG_LINEAR` that is
log space: `exp(log(y_end) + slope_log·(x − x_end))`. Correct the wording in `19`/`AUDIT_topic0`.

### A2. S16 is WITHDRAWN — the business-day counting invariant
S16 recorded *"start-exclusive / end-inclusive; no present consumer needs the alternative."*
**That was wrong** (audit 1.3): CDI accrual uses `[start, end)` via `_overnight_days` while
`business_days_between` uses `(start, end]`, so BUS/252 discounting and CDI accrual disagree by one
business day. ANBIMA standard is `[start, end)`.
**Ruling: one primitive, `[start, end)`.** Route BUS/252 through the same helper `_overnight_days`
uses. Delete the S16 invariant.

### A3. Speculative FIELDS are cut — refining "design complete, populate incrementally"
Both independent audits agree I over-applied the principle (S6 · S10 · F4). The corrected boundary:

> **Design the SHAPE complete (what decomposes, what is open vs closed).**
> **Do NOT populate FIELDS speculatively.**

A decomposed `NumericalConfig` is *shape* and was right. Filling it with `cos_n`/`cos_l`/`tree_steps`
before any engine exists is *guessed fields* — and worse, **schema v1 freezes those names before an
engine has validated them**, which is a costlier retrofit than the one I was avoiding.

**Cut now (keep the shape, drop the guessed content):**
- `underlying.py` — **keep** `Underlying` protocol + `AssetClass` (RateIndex uses both);
  **delete** `ReferenceEntity`, `InflationIndex`, `FxFixing`, `EquityUnderlying`,
  `CommodityUnderlying`, `InflationInterp` (~66 lines, zero consumers, guessed fields such as
  `fixing_time: str`, required `grade: str`).
- `numerical_config.py` — **ship with the first engine that reads it.** Retain only knobs with a
  present consumer. Do **not** freeze a serialisation schema for unvalidated knob names.
- `results.py` — keep `pv` / `accrued` / `clean`; inline `basis: Currency | None`;
  **drop** `sensitivities` (a stringly-typed greek bag — real risk is curve × pillar × bump type),
  `cashflow_breakdown` (a parallel array with no dates or flow identity), `diagnostics`.

Each costs one file-touch to reintroduce correctly; each is a breaking change if built upon.

---

## PART B — execution order

### Phase 0 — make it verifiable (nothing else is checkable until this lands)
- **`src/pricebook_ng/foundation/__init__.py` is MISSING** (confirmed in-repo, not an artifact of
  the audited copy). Add it; define the public API surface with explicit `__all__` **before users
  freeze it for you**.
- Add **`py.typed`** (fully typed code currently reads as `Any` to downstream mypy) and a **root
  `pyproject.toml`** (only `python/pyproject.toml` exists, and that is the quarry's).
- **Rename `calendar.py` → `calendars.py`.** It shadows stdlib `calendar` the moment `foundation/`
  lands on `sys.path`. Cheap now, expensive after four layers import it.
- Confirm `verify.py all` + the full suite run green afterwards.

### Phase 1 — Tier 1 confirmed bugs (each counterexample is a ready-made regression test)
1. **1.1 ACT/ACT AFB** undercounts leap-to-leap: `2004-02-29 → 2008-02-29` gives **3.9972677…**,
   must be **4.0**. Count whole years by (y, m, d) with explicit 29-Feb ↔ 28-Feb equivalence.
2. **1.2 LOG_LINEAR + CONTINUE_SLOPE → negative DFs** (see A1). `DF(30) = −0.275`, correct `+0.209`.
3. **1.3 BUS/252 vs CDI interval mismatch** (see A2). One primitive, `[start, end)`.
4. **1.4 Backward schedule**: EOM keyed on `start` when generating backward from `end` (must key on
   the generation anchor, ISDA §4.10), **and** roll-day drift (May 31 quarterly backward →
   Nov **28**, must be Nov 30). Generate as `anchor + k·tenor` or carry the roll day explicitly.
5. **1.5 Furikae non-determinism**: iterating `set[date]` makes Tokyo business days process-dependent,
   and `lru_cache` freezes whichever answer came first. Iterate `sorted(holidays)`.

### Phase 2 — Tier 2 convention bugs (before any fixings/curves are stored against wrong windows)
6. **2.1 SOFR declaration is wrong.** ISDA-standard SOFR OIS is plain compounded-in-arrears, no
   observation shift: `RfrConvention(payment_delay=2)`. The *engine mechanics are correct*; this is a
   one-line declaration fix. **Keep** shift=2 on the LIBOR-fallback index (correct per ISDA fallback).
7. **2.2 USD calendar wrong for SOFR**: **Good Friday missing** (no SOFR fixing that day — breaks
   annually on real data); 2021-12-31 wrongly a holiday. Split into `US_GOVERNMENT_SECURITIES`
   (SIFMA, with Good Friday), `NYSE`, and a Fed-bank calendar; bind SOFR to SIFMA.
8. **2.3 LONDON** missing 2022–23 one-offs (Jubilee Jun 2–3, state funeral Sep 19 2022, coronation
   May 8 2023, 2020 VE-Day move). Add a `dates(*ds)` one-off combinator to the rule DSL (~5 lines).
9. **2.4 Tokyo equinoxes hardcoded** — wrong in 2024/2025/2026. Use the astronomical approximation;
   add the sandwiched-holiday rule and the Emperor's-Birthday move (`since=2020`).

### Phase 3 — Tier 3 structural (CalendarProtocol first — most expensive to retrofit)
10. **3.2 `JointCalendar` is unusable** — lacks `adjust`/`add_business_days`; every consumer slot is
    typed `Calendar`. Cross-currency schedules, FX dates and NY+LON payment calendars are a runtime
    `AttributeError` today. Define **`CalendarProtocol`**, implement the missing methods on
    `JointCalendar` (trivial — the algorithms only call `is_business_day`), retype every consumer.
11. **3.3 Registries**: add public `register_calendar` / `register_rate_index`; make **all**
    `register_*` **raise on conflicting re-registration** (`register_currency` silently overwrites,
    which also breaks its interning claim); add a test-isolation helper.
12. **3.4 Serialisation convention**: identities (Calendar, RateIndex) serialise **by name**; atoms
    (Money, Accrual, Cashflow) **by value**. Fix it in foundation — which owns the identities — or
    each layer invents its own.
13. **3.5 `Schedule` provenance**: emit per-period records (start, end, `is_stub`, ICMA reference
    anchors, payment date). Add explicit stub boundary dates to `ScheduleTerms`. **Raise loudly on
    long stubs** until ACT/ACT ICMA supports multiple quasi-coupon periods. Decide where
    `payment_delay` applies (often a different calendar than fixing, for XCCY) — it is dead code today.
14. **3.6 FX spot**: implement `fx_spot_date(pair, trade_date)` with per-currency intermediate-day
    checks, joint validity in both centres and USD-holiday involvement for crosses. Move `spot_lag`
    out of `CurrencyPair` equality (it fragments dict keys) into a pair-conventions registry.
15. **3.7 `_denominator` silent fallback** — any day count other than ACT/365F silently gets 360.
    **Raise on unsupported conventions.** This violates our own no-silent-fallback rule.
16. **3.7 `FixingSource` protocol** — have `accrued_rate` accept `FixingSource`
    (`rate(index_name, on) -> float`); keep `FixingHistory` as the trivial implementation. One line
    now, a two-truths migration later.

### Phase 4 — apply the Part A cuts
17. Delete the speculative types per A3. Update `19`/`20`/`AUDIT_topic0` wording for A1 and A2.

### Phase 5 — archive the findings so nothing is lost
18. **Tier 4 (19 items)** → record each in `OPEN.md` with its **trigger topic**, per our debt rule
    (a deferral with no re-open trigger does not exist). Notable ones with near-term triggers:
    zero/negative `Tenor` infinite loop · `least_squares` cannot bound (needed by every stoch-vol
    calibration) · missing EURIBOR_6M (the standard EUR vanilla floating leg) · `TimeMeasure` concept ·
    CDS2015 semiannual maturity roll (before the credit layer) · bivariate normal + non-central χ²
    (models layer, week one).
19. **`PONYTAIL-DEBT.md`** (interpolator rebuilt per call) → `OPEN.md` with its stated Topic-1 trigger.
20. Move `redesign/independent_audits/` into the tracked record and reference it from
    `redesign/README.md` under the reasoning trail.

### Closing convention — `closed_<name>.md`
21. **`git mv` each report to `closed_<name>.md`** once it is discharged —
    `closed_AUDIT.md` · `closed_PONYTAIL_AUDIT.md` · `closed_PONYTAIL-DEBT.md`.

    **A report earns the prefix only when every one of its findings is either FIXED (with a test) or
    LEDGERED in `OPEN.md` with a named re-open trigger.** Nothing is closed by being read, by being
    judged low priority, or by running out of session. If one finding is unresolved, the file keeps
    its bare name — the prefix is the audit trail, so it must be able to be *wrong*, which means it
    must be earned.

    Append a short **closure block** at the top of each renamed file: date · the commit range that
    discharged it · a per-finding disposition table (`fixed` / `ledgered→OPEN.md#id` / `rejected +
    reason`). A `rejected` verdict is legitimate but must carry the reasoning — it is a ruling, and
    rulings are recorded, never implied by silence.

    This convention now stands for **all** future audits in `redesign/independent_audits/`:
    unprefixed = live, `closed_` = discharged. Cowork spot-checks a sample of dispositions at the
    next checkpoint and may **un-close** a report (rename back) if a finding was waved through.

---

## Method (unchanged)
Red before green — every Tier 1/2 counterexample becomes a failing test first. Report findings and
proposed fixes before large structural changes (Phase 3). `verify.py all` green at each phase.
Checkpoint per `redesign/11` at the end of Phase 2 and Phase 3.

## Note on what the audit could NOT verify
The 37 markets' **holiday data content**, ACT/365L freq>1 against a primary source, end-to-end
long-stub schedules, scipy version-dependent behaviour, and **the project's own test suite — because
the package does not import.** Phase 0 unblocks the last one.
