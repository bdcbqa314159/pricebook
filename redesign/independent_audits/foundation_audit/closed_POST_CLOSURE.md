# Post-Closure Report — independent re-verification of the foundation audit closure

**Date:** 2026-07-19
**Scope:** `/Users/bernardocohen/work/analysis/foundation/` at `545a241` — 15 modules, ~3,300 lines
**Subject:** the closure of `AUDIT.md` (48 findings), `PONYTAIL_AUDIT.md` (8 findings),
`PONYTAIL-DEBT.md` (1 marker), as claimed by `CLOSURE_VERIFICATION.md`
**Method:** Full read of every module + **re-execution of every audit counterexample** against the
tree (via an import shim; this repo carries no tests). This is the numerical re-verification
CLOSURE_VERIFICATION.md scoped out of itself (its Part 5, step 3) — performed cold, without
trusting the closure report, the builder, or the verifier.

---

## Verdict

**The closure holds under execution, not just inspection.** All 17 counterexamples from the
original audit pass against this tree — Tier 1 five for five, Tier 2 four for four, the Tier 3
structural core, the T4.2 promotion, and every ponytail disposition. No fixed finding regressed;
no ledgered item was silently built or silently dropped; both rejections stand with their
recorded reasons in the code.

The new findings of this pass (Part 4) are small, and none contradicts a closure disposition.
They share one shape worth naming: they live **between** findings that are each correctly marked
FIXED — the census guarantees coverage of the findings, not of the seams between them.

---

# Part 1 — Executed counterexamples (the audit's own oracles)

| Finding | Oracle | Result |
|---|---|---|
| 1.1 AFB leap-to-leap | 2004-02-29 → 2008-02-29 | **exactly 4.0** ✓ (also 2016→2020) |
| 1.2 LOG_LINEAR + CONTINUE_SLOPE | DF nodes [1,2,5]/[0.97,0.94,0.80], t=30 | **+0.208661** (audit predicted +0.209; was −0.275), matches exact log-slope to 1e-12 ✓ |
| 1.3 BUS/252 vs CDI | São Paulo holiday-boundary span | `business_days_between` ≡ `_overnight_days`, both `[start, end)` ✓ |
| 1.4 backward schedule | May 31 quarterly backward; EOM vs 2030-06-30 maturity | Nov 30 (no roll-day drift); EOM keys on the maturity anchor ✓ |
| 1.5 furikae | overlapping Golden-Week set, both iteration orders | identical substitute set ✓ |
| 2.1 SOFR declaration | registry inspection | `shift=0, lookback=0, payment_delay=2`; fallback index keeps `shift=2` ✓ |
| 2.2 USD calendar | Good Friday 2024/2026; 2021-12-31 | GF closed, 2021-12-31 **open**, SOFR bound to `US_GOVERNMENT_SECURITIES` ✓ |
| 2.3 LONDON one-offs | Jubilee ×2, funeral, coronation, VE-Day + moved-off Mondays | all closed / correctly open ✓ |
| 2.4 Tokyo equinoxes | 2024–2026 published dates; era-gated birthdays | astronomical rule correct; Dec 23 open from 2019 ✓ |
| 3.2 JointCalendar | adjust / add_business_days / protocol membership | works, satisfies `CalendarProtocol` ✓ |
| 3.3 registries | conflicting re-registration (currency, calendar, index) | all raise ✓ |
| 3.4 serialization | identity-by-name round trips, atom-by-value round trips | `from_dict` returns the interned instance ✓ |
| 3.5 ICMA 251.2 | ISDA §4.16 long first coupon, 15 Aug 2002 → 15 Jul 2003 | **0.9157608695652174** to 1e-12 ✓; `SchedulePeriod` provenance + `RegularPeriod` stubs work ✓ |
| 3.6 FX spot | USD/CAD T+1; EURUSD over Easter 2026 | lag registry off the identity ✓; Apr 7 correct (TARGET closed GF **and** Easter Monday) ✓ |
| 3.7 fallbacks | `_denominator(30/360)`; `FixingSource` structural check | raises; protocol satisfied by any `rate()` bearer ✓ |
| T4.2 | `Tenor.parse("0D")`, `"-3M"` | both raise ✓ |
| Ponytail cuts | module/field/attr absence | `numerical_config` gone, siblings gone, results trimmed ✓ |

One verifier-side note: the EURUSD-over-Easter check initially "failed" against **my** expected
date — I had forgotten TARGET closes Easter Monday. The code was right. Recorded because that is
what an oracle dispute looks like, and the resolution direction matters.

---

# Part 2 — Disposition audit (all 48 + 8 + 1)

- **26 fixed:** all verified by execution above. The fixes are principled, not patched — AFB
  counts years from the end anchor; log-linear extrapolates in its own space; one business-day
  primitive serves BUS/252 and CDI; schedules step `anchor + k·tenor` from the seed.
- **19 ledgered + 3 deferred sub-parts:** spot-checked every one — all genuinely still open,
  exactly as recorded. `solvers.py` still `lm`-only, `distributions.py` still norm-only, no
  `TimeMeasure`, no CDS-2015 maturity roll, `is_holiday` no `year−1` spill, `observe()` Sat/Sun
  hardcode, NEAREST ties backward, `log(y≤0)` unguarded, CDI convexity trap, `Weekend`
  time-invariant, month-arithmetic triplicated, `Frequency` bridge undecided. Sub-parts
  consistent: no NYSE/Fed calendars (AC-2.2b), no Silver-Week/Olympic rules on TOKYO (AC-2.4b),
  no quote-order/triangulation (AC-3.6b).
- **2 rejected:** `exponential_growth` and `_rule_set` present, each carrying its ruling in a
  docstring/comment. Correctly not re-flagged by the fresh over-engineering sweep.
- **Ponytail 8:** 6 cuts landed byte-for-byte as prescribed, 2 rejections stand. The 1 debt
  marker (`interpolation.py:77` per-call spline rebuild) is in-code with its ceiling and
  Topic-1 trigger.

**Discipline signal:** the closure added ~370 net lines of audit-mandated machinery
(`CalendarProtocol`, `PaymentRule`, `RegularPeriod`, `SchedulePeriod`, `dates()`, `equinox()`,
`temporary_*`, FX spot) and a fresh whole-tree over-engineering scan afterwards found only
~41 lines of residue — most predating the closure. Growth under audit pressure without new
speculation is the durable improvement, beyond any individual fix.

---

# Part 3 — Position on CLOSURE_VERIFICATION.md

**Its verdict is confirmed on stronger evidence than it had.** It verified structure and
explicitly deferred numerical re-derivation; Part 1 above is that deferred step, and it passes.

- **V1 (OPEN.md self-description)** — not checkable from this repo (the file lives in the parent
  tree); its materiality reasoning is sound.
- **V2 (stale quarry header)** — not checkable here; agree with the fix.
- **V3 (ledger watched by nothing automated)** — correct, and Part 2 is the proof by
  demonstration: the 22 items are verifiably intact *because a human re-read them*, which is
  exactly V3's point. The checkpoint-sweep it prescribes is the right mechanism.
- **Extension, not disagreement:** its safeguard 6 warns about deferred sub-parts under FIXED
  parents. This pass found the subtler variant — gaps **between** two FIXED findings, holding no
  id at all (Part 4: A3 falls between 3.2 and 3.4; B3 between 3.6 and AC-3.6b; B1 between the
  verified lockout cases). The disposition census covers findings, not seams.

Standing caveat, once: this repo carries no tests, no `pyproject.toml`, no `__version__` — the
closure's named test anchor (`tests_ng/`, `verify.py`, `OPEN.md`) lives in the parent
`pricebook_ng` tree. From this tree alone the dispositions are verified by re-execution, not by
their named tests.

---

# Part 4 — New findings (actionable now; none reopens the closure)

## A — Retrofit-risk seams (do before the layer that needs them)

**A1 — `TimeMeasure` absent; promote from T4 to before Topic 1.** The curve layer needs
`date → t` on day one and everything above touches it. Left absent, each curve invents its own
`anchor + day_count` pairing and drift compounds upward. Frozen
`TimeMeasure(anchor, day_count)` with `t(d) -> float`, ~30 lines — but L0's 30 lines.

**A2 — `Frequency` ↔ ICMA `frequency: int` bridge undecided.**
(`schedule.py` Tenor-step vs `day_count.py:48-57` periods-per-year int.) A fixed-leg builder
maps one to the other in its first week; for 28D/TIIE the mapping is undecidable and needs a
ruling. Add `Frequency.per_year()` raising on non-integer cases; record that BUS-period
products don't meet ICMA.

**A3 — `JointCalendar` breaks the by-name round trip.** Identity `"A+B"`
(`calendars.py:466-468`) is unresolvable by `get_calendar` (`market_calendars.py:855-860`), so
the first serialized XCCY trade (a `JointCalendar` on its `RollRule`) cannot rehydrate. Falls
between 3.2 FIXED and 3.4 FIXED. ~6 lines: split on `"+"`, give `JointCalendar` the same
`to_dict` shape.

## B — Wrong-answer edges (fix with a test each)

**B1 — `accrued_rate` lockout underflow** (`rate_index.py:214-216`): `lockout ≥ len(window)`
makes `frozen` negative; Python negative indexing then silently freezes rates to *early* dates
for `window < lockout < 2·window`, bare `IndexError` beyond. A short stub with a standard
lockout reaches this. Raise when `lockout >= len(window)`.

**B2 — `PricingResult.clean` mixes currencies unguarded** (`results.py:38-41`): subtracts raw
amounts. Fix is `return self.pv - self.accrued` — `Money.__sub__` already guards. One line,
deletes code.

**B3 — `fx_spot_date` intermediate-day rule too strict for USD pairs**
(`settlement.py:118-122`): counted days require BOTH centres open; ACI practice is that a USD
intermediate holiday does not pause the count for most USD pairs (USD constrains the final date;
LatAm-style pairs are the exception). EURUSD before Columbus Day settles a day late as written.
Either implement the asymmetric rule (~10 lines) or pin the joint-counting behaviour under
AC-3.6b explicitly — do not leave it unstated.

## C — Residual over-engineering (fresh whole-tree scan; ~41 lines)

- `yagni:` half-day machinery — `day_type()`, `DayType`, `HolidaySet.half_days`,
  `_half_days_of`, 3 US half-day rules: zero readers, and `CalendarProtocol` deliberately
  excludes `day_type`. Ship with the first fixing-cutoff consumer (~28 lines).
- `shrink:` `_boundary_slope` 1e-6 numeric step is noise for LINEAR — end-segment slope is
  exact, mirroring `_boundary_slope_log` (`interpolation.py:82-92`).
- `shrink:` `register_unit(name, symbol)` — `name` unused in the body (`money.py:157-164`).
- `delete:` `_SPOT_LAGS["USDRUB"]` — RUB unregistered; pair unconstructible (`settlement.py:100`).
- `shrink:` (optional) `next_imm`/`next_cds_roll` same loop twice — rule-of-three says leave
  until a third roll family arrives (`schedule.py:303-325`).

## C2 — Ponytail debt ledger, re-scanned (vs `PONYTAIL-DEBT.md`)

`grep -rn 'ponytail:'` over the tree (note: the old report's `(#|//) ?ponytail:` form misses
mid-comment markers — this marker only matches the bare form):

| Marker | Ceiling | Upgrade trigger | Status |
|---|---|---|---|
| `interpolation.py:77` — scipy spline rebuilt per `interpolate()` call | O(N) rebuild per evaluation → O(N·M) for M queries | Topic 1: a hot curve caches the interpolator | live, trigger intact |

**1 marker, 0 with no trigger** — unchanged in count from the old ledger, with two deltas:

- The old cross-ref ("the rebuild also drives a 2× cost in `_boundary_slope` during
  extrapolation") has **narrowed**: the audit-1.2 fix gave LOG_LINEAR an exact
  `_boundary_slope_log`, so the doubled rebuild now hits only spline-method
  `CONTINUE_SLOPE` — the DF-curve path no longer pays it.
- The closure's ~370 new lines carry **zero new markers**, and the fresh over-engineering
  sweep found no unmarked shortcut in them either — deferrals in the new code went to the
  AC-ledger with ids, not to comments. One convention, consistently the stricter one.

## D — Ledger hygiene (minutes)

- **D1.** Write the B3 scope into AC-3.6b's entry (whichever branch is taken).
- **D2.** One paragraph in this repo stating where its falsifiers live (parent `pricebook_ng`
  tree: `tests_ng/`, `OPEN.md`, `verify.py`, version).

---

# Part 5 — Load-bearing assessment (can layers land on this?)

**Yes.** The properties that make it safe to build on, verified in code:

1. **Value semantics, no ambient state** — everything frozen/hashable/equal-by-value; the only
   mutable state is append-only raise-on-conflict registries behind frozen views; no global
   evaluation date. Upper layers can cache and key on any foundation value.
2. **Errors stop at layer 0** — no-silent-fallback is now actually enforced everywhere it was
   flagged (BUS/252, ICMA anchors, `_denominator`, degenerate windows, `add_business_days(d,0)`).
3. **Coherent identity flow** — calendar → `RollRule` → `AccrualConvention` → `RateIndex`; no
   currency inference; consumers depend on protocols (`CalendarProtocol`, `FixingSource`,
   `Underlying`), not concretions.
4. **Extension points proven by use** — the closure itself extended the DSL, registries and
   protocols without touching any consumer.

Per arriving layer: market-data ✓ (FixingSource is a drop-in seam); curves ✓ *except A1*;
trade ✓ *except A2/A3*; models — thin by design, additive, zero retrofit; new asset class —
one declaration per concept, shape proven by `RateIndex`.

---

## Recommended order

1. **B1, B2** — the only wrong-answer edges; each with its regression test (an hour together).
2. **A3** — six lines; unblocks XCCY serialization.
3. **B3 + D1** — decide, implement or ledger, record.
4. **A1, A2** — the two rulings to make before Topic 1 opens; A1 is the one that gets expensive
   if skipped.
5. **C** rides along with whatever touches those files next; **D2** any time.

---

## Bottom line

Forty-eight audit findings, eight ponytail findings, one debt marker: every disposition
re-verified, the fixed ones by executing the audit's own counterexamples. The closure is sound,
the discipline held, and the tree grew ~370 audit-mandated lines without new speculation. What
this pass adds is the seam residue the disposition census structurally cannot see — five small
fixes and two pre-Topic-1 rulings — and the foundation is fully load-bearing.

---

## Closure disposition (2026-07-20, branch `fix/post-closure-seams`)

Every Part-4 item is fixed-with-a-test, ruled-and-recorded, or ledgered. Verified in the real tree
(`src/pricebook_ng/foundation/`, not the audited copy). Red→green throughout; 151 tests, all
`verify.py` gates + pyright green.

| item | disposition |
|---|---|
| **A1** `TimeMeasure` | **RULED** — `redesign/20` Part A addendum: the only sanctioned `date→t` map, built as an L0 module in Topic 1's first slice. Code not built (30 unconsumed lines = what Phase 4 deleted). Ledger `AC-T4.5` promoted to "before Topic 1". |
| **A2** `Frequency` bridge | **RULED** — `redesign/20` Part A addendum: `Frequency.per_year()` raises for non-integer tenors; BUS-period products skip ICMA. Ledger `AC-T4.15`. |
| **A3** JointCalendar round-trip | **FIXED** — `JointCalendar.to_dict` + `get_calendar` split on `"+"`; test `test_a3_jointcalendar_round_trips_by_name`. |
| **B1** lockout underflow | **FIXED** — `accrued_rate` raises when `lockout >= len(window)`; test `test_b1_lockout_longer_than_window_raises`. |
| **B2** `clean` cross-ccy | **FIXED** — `return self.pv - self.accrued` (Money guard); test `test_b2_clean_rejects_cross_currency_accrued`. |
| **B3** fx_spot intermediate-day | **RULED (keep joint counting)** — no citable source with a verifiable worked example, and the green-oracle gate forbids coding an unverifiable convention. Scope written into `OPEN.md` **AC-3.6b** and the `fx_spot_date` docstring (D1). |
| **C** residue (~41 lines) | **DONE** — half-day machinery / `day_after_thanksgiving` / `_SPOT_LAGS["USDRUB"]` deleted; `register_unit(symbol)` unused param dropped; `_boundary_slope` exact for LINEAR. `next_imm`/`next_cds_roll` **left** (rule of three). |
| **C2 / debt marker** | **unchanged** — 1 live `ponytail:` marker (`interpolation.py` spline rebuild), ledgered `AC-PD.1`, Topic-1 trigger intact. |
| **V1 / V2** (from `CLOSURE_VERIFICATION.md`) | **already FIXED** on the closure branch (commit `ce72f5a9`, now on `main`) — the audited copy predated the fix, so this pass reported them open. |
| **V3** | **DONE** — `redesign/11` gains a 6th standing checkpoint review input: re-read the whole `AC-*` ledger and check each trigger, since no automated gate watches them. |

### D2 — where the falsifiers live

This report and its findings were audited against a *copy* (`work/analysis/foundation/` @ `545a241`)
that carries no tests, no `pyproject.toml`, no `__version__`. **The falsifiers live in the parent
`pricebook_ng` tree**, not beside these reports: the executable oracles are `tests_ng/L0/` (run
`verify.py tests --layer 0`), the deferred-item ledger is the repo-root `OPEN.md`, the structural
gates are `verify.py all` (`acyclic`/`fields`/`layers`/`debt`/`provenance`/`version`) plus pyright,
and the version is `src/pricebook_ng/__init__.py`'s `__version__`. To re-verify a fix: revert it and
confirm its named test in `tests_ng/L0/test_post_closure.py` (or `test_audit_closure.py`) goes red.
