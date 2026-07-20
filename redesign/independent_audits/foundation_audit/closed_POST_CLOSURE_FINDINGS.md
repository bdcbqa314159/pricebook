# Post-Closure Findings — actionable now

**Date:** 2026-07-19
**Scope:** `/Users/bernardocohen/work/analysis/foundation/` at `545a241` (post-audit-closure code)
**Method:** Full read of the 15 modules + re-execution of every audit counterexample against the
tree (all 17 pass: Tier 1 5/5, Tier 2 4/4, Tier 3 core, T4.2, ponytail dispositions). This report
lists only what surfaced *beyond* the closed audits — the seams the next layers hit first, plus
new edges found on this pass. Everything here is small; nothing reopens the closure.

---

## A — Retrofit-risk seams (do before the layer that needs them)

### A1. `TimeMeasure` is absent — promote from T4 to *before Topic 1*

- **Where:** absent (invariant recorded at `rate_basis.py`; audit T4 "No TimeMeasure concept")
- **Why now:** the curve layer needs `date → t` on day one, and everything above the curve
  layer touches it. Left absent, each curve invents its own `anchor + day_count` pairing and
  the drift compounds upward — the exact failure class this codebase is designed against.
- **Fix:** frozen `TimeMeasure(anchor: date, day_count: DayCountConvention)` with
  `t(d) -> float`. ~30 lines, but they must be L0's 30 lines, not Topic 1's.

### A2. `Frequency` ↔ ICMA `frequency: int` bridge is undecided

- **Where:** `schedule.py` (`Frequency`, a `Tenor` step) vs `day_count.py:48-57`
  (`CouponPeriod.frequency`, periods-per-year int)
- **Why now:** a fixed-leg builder maps one to the other in its first week. For 28D/TIIE the
  mapping is *undecidable* (365/28 is not an integer) — someone must rule. Undecided in
  foundation = decided ad-hoc in the trade layer.
- **Fix:** a `Frequency.per_year() -> int` that raises for tenors with no integer periods-per-year
  (28D, daily, bullet), plus a recorded ruling on what BUS-period products (TIIE) pass to ICMA
  contexts (they don't use ICMA — make that the recorded answer).

### A3. `JointCalendar` breaks the by-name serialization round trip

- **Where:** `calendars.py:455-477` (identity `"A+B"`), `calendars.py:410-414`
  (`Calendar.to_dict` → rehydrate via `get_calendar`), `market_calendars.py:855-860`
- **Why now:** the first serialized XCCY trade holds a `JointCalendar` on its `RollRule`;
  `get_calendar("US_GOVERNMENT_SECURITIES+LONDON")` raises. A day-one trade-layer need hiding
  behind audit 3.2/3.4 both marked FIXED.
- **Fix:** `get_calendar` splits on `"+"` and returns a `JointCalendar` of the parts; give
  `JointCalendar` the same `to_dict` shape. ~6 lines.

---

## B — Wrong-answer edges (small, fix with a test each)

### B1. `accrued_rate` lockout underflows on short windows

- **Where:** `rate_index.py:214-216` — `frozen = len(window) - 1 - rfr.lockout`
- **Edge:** `lockout >= len(window)` makes `frozen` negative; `rate_dates[min(i, frozen)]`
  then uses Python negative indexing — for `window < lockout < 2·window` every rate silently
  freezes to an *early* date (wrong answer); beyond that, a bare `IndexError`. A short stub
  period with a standard lockout reaches this.
- **Fix:** raise when `rfr.lockout >= len(window)` (a lockout longer than the window is
  undefined), with a message naming the window.

### B2. `PricingResult.clean` mixes currencies unguarded

- **Where:** `results.py:38-41` — `Money(self.pv.amount - self.accrued.amount, self.pv.currency)`
- **Edge:** the one place in `money.py`'s orbit that subtracts raw amounts without the currency
  guard; a cross-currency `accrued` silently produces a wrong `clean`.
- **Fix:** `return self.pv - self.accrued` — `Money.__sub__` already carries the guard. One line,
  and it *deletes* code.

### B3. `fx_spot_date` intermediate-day rule too strict for USD pairs

- **Where:** `settlement.py:118-122` — a counted day must be a business day in BOTH centres
- **Edge:** ACI practice: for most USD pairs a USD holiday on an intermediate date does NOT
  pause the count (USD constrains only the final value date); the pause applies to the
  LatAm-style pairs. As written, EURUSD traded before Columbus Day settles a day late.
- **Handle now (choose one):** implement the asymmetric rule (~10 lines: intermediate days
  checked on the non-USD calendar only for USD pairs, both for the exception list), OR pin the
  current joint-counting behaviour under **AC-3.6b** in the ledger explicitly so it cannot pass
  as complete. Do not leave it unstated.

---

## C — Ledger hygiene (minutes)

- **C1.** Add the B3 intermediate-day scope to **AC-3.6b**'s entry text (whichever branch of B3
  is taken) — today a reader of "3.6 FIXED" assumes the spot algorithm is complete.
- **C2.** This repo carries no note that its test anchor (`tests_ng/`, `OPEN.md`, `verify.py`,
  `__version__`) lives in the parent `pricebook_ng` tree. One paragraph in a README or in
  `__init__.py`'s docstring stating where the closure's falsifiers live. (Packaging itself is
  out of scope per instruction — this is only the pointer.)

---

## Explicitly NOT in this report

- The 19 ledgered T4 items and 3 deferred sub-parts (AC-2.2b, AC-2.4b, AC-3.6b): all verified
  still open and correctly triggered; they ride with their asset-class topics.
- solvers `trf`/bounds, bivariate normal CDF: additive when the models layer arrives; zero
  retrofit cost, correctly deferred.
- The `interpolation.py:77` per-call rebuild: ponytail-marked with a live Topic-1 trigger.

## Suggested order

1. B1, B2 — wrong-answer edges, each with its regression test (an hour, together).
2. A3 — six lines, unblocks XCCY serialization.
3. B3 + C1 — decide, implement or ledger, record.
4. A1, A2 — the two decisions to make before Topic 1 opens; A1 is the one that gets expensive
   if skipped.

---

## Closure disposition (2026-07-20, branch `fix/post-closure-seams`)

All done or ruled; full detail + the D2 falsifier-pointer in the companion `closed_POST_CLOSURE.md`
§"Closure disposition". Summary: **B1/B2/A3 FIXED** with named tests in
`tests_ng/L0/test_post_closure.py`; **A1/A2 RULED** in `redesign/20` Part A (code lands with its
first consumer; ledger `AC-T4.5`/`AC-T4.15`); **B3 keeps joint counting** (no citable green oracle)
with the scope pinned in `OPEN.md` **AC-3.6b** + the `fx_spot_date` docstring; **C residue** done
(`next_imm`/`next_cds_roll` left per rule of three); **C2 pointer** = D2. Red→green throughout;
151 tests + all gates + pyright green. Nothing reopened a closed disposition; the S5 half-day item
was *redirected* (classify-now → defer-to-first-fixing-cutoff-consumer), recorded in the CHANGELOG.
