# L0 Foundations — Layer Audit

**Started:** 2026-06-11 (Gate 2 prep)
**Scope:** `pricebook.core.*` — 36 modules organised into four passes by dependency depth.
**Method:** Fresh read per module; cross-reference existing `MODULE_HEALTH.md` findings; document status, real bugs, doc gaps, test gaps, slicing proposals.

## L0 audit — completion summary

| Pass | Modules | Confirmed bugs | HIGH fixed | MED fixed |
|---|---:|---:|---:|---:|
| **A** atomic primitives | 13 | 20 | 4 / 4 ✅ | 2 / 2 ✅ |
| **B** simple composites | 7 | 4 | 1 / 1 ✅ | 1 / 1 ✅ |
| **C** portfolio constructs | 8 | 3 | 0 / 0 | 0 / 1 *(C.7 B1 settlement lag queued)* |
| **D** engine plumbing | 7 | 3 | 0 / 0 | 0 / 2 *(D.1 B1, B2 queued)* |
| **Total** | **35** | **30** | **5/5** ✅ | **3/6** |

(36-module count in the scope minus `__init__.py` = 35 substantive modules audited.)

**Remaining open fixes from L0 audit:**
- C.7 B1 — settlement lag (calendar days, should be business days). MED.
- C.8 B1 — `dollar_gamma` docstring/formula mismatch. LOW.
- D.1 B1 — empty-dict → None on round-trip. MED.
- D.1 B2 — fields silently dropped on round-trip. MED.
- D.1 B3 — `replace()` aliases mutable dicts. LOW-MED.
- A.1 B1 final-slice — flip `strict_icma=True` default after auditing remaining callers. (Many slices.)
- A.11 B3-B7 — minor serialisation robustness. LOW.
- B.3 C1 — legacy `core.market_data` vs new `pricebook.market_data` (G1 P2). ARCH — Gate 2 decision.
- Generic `vars(self)` to_dict cleanup (~15 callsites). Single sweep slice.

All confirmed HIGH-severity bugs are fixed. All confirmed MEDIUM-severity bugs from Pass A and B are fixed. Pass C and D MEDs are still queued.

| Status legend | Meaning |
|---|---|
| ✅ Clean | No real bugs found; tests adequate; docs accurate. |
| ⚠️ Bugs found | At least one confirmed real bug. |
| 📝 Doc/test gaps | No bugs but gaps in docstrings or coverage worth filling. |
| ⏳ In progress | Reading underway. |
| ❓ Deferred | Audit pending. |

---

## Pass A — Atomic L0 (no internal pricebook deps)

| # | Module | Status | Confirmed bugs | Doc/test gaps |
|---|---|---|---|---|
| A.1 | `day_count.py` | ⚠️ | 2 (B1 silent-fallback, B2 ZeroDivisionError) | 5 |
| A.2 | `calendar.py` | ⚠️ | 5 (B1 Sat-substitute, B2 Tokyo no-substitute, B3 nth_weekday spill, B4 spill-window, B5 joint mutation) | huge |
| A.3 | `currency.py` | 📝 | 0 | 4 (settlement_lag, is_ndf, forward_rate, all_g10_pairs untested) |
| A.4 | `schedule.py` | ⚠️ | 1 (B1 EOM anchored to end not start in front-stub) | 3 (post-adjust dedupe untested, WEEKLY untested, stub-30-day heuristic) |
| A.5 | `solvers.py` | ⚠️ | 2 minor (B1 to_dict mutation, B2 itp maxiter contract) | NaN/Inf paths untested |
| A.6 | `interpolation.py` | 📝 | 0 | Akima untested; right-extrap not slope-continued for cubic methods |
| A.7 | `approximation.py` | 📝 | 3 trivial (same vars(self) to_dict pattern) | small |
| A.8 | `caching.py` | ✅ | 0 | type hint mismatch (cosmetic) |
| A.9 | `protocols.py` | ✅ | 0 | none |
| A.10 | `fixings.py` | ✅ | 0 | get_with_lag fallback semantics + persistence error handling |
| A.11 | `serialisable.py` | ⚠️ | 7 documented-design-limitations (list[T] dispatch, Union[A,B,None], registry no-op, .item() ordering, CurrencyPair parse, bare params lookup, Enum int-vs-str) | 4 |
| A.12 | `serialization.py` | ⚠️ | 2 (B1 curated import list footgun, B2 CurrencyPair parse dup) | facade — coverage via downstream |
| A.13 | `numerical_config.py` | ✅ | 0 | 0 (just landed; 14 tests; clean) |

---

## A.1 — `core/day_count.py`

**Purpose:** Year-fraction calculations across 7 day-count conventions (ACT/360, ACT/365F, 30/360 US, 30E/360, ACT/ACT ISDA, ACT/ACT ICMA, BUS/252). Plus `business_days_between` and `date_from_year_fraction` helpers.

**Internal deps:** Only `pricebook.core.calendar` (TYPE_CHECKING import + one lazy runtime import of `SaoPauloCalendar`). True L0.

**Caller fan-in:** 72 references to `ACT_ACT_ICMA` alone across the library — fixed-income (sovereigns, EM, linkers), inflation, govt-bond pricing. Anything that touches a bond touches this module.

**Test file:** `python/tests/test_day_count.py` (144 lines, 4 test classes).

### Status: ⚠️ Real bugs found

### Confirmed bugs

#### B1 — `_act_act_icma` silent fallback to ACT/365F masks bond mispricing  *[HIGH]*

**Location:** `day_count.py:167-168, 172`.

```python
if ref_start is None or ref_end is None or frequency is None:
    return (end - start).days / 365.0  # silent fallback to ACT/365F
period_days = (ref_end - ref_start).days
if period_days <= 0:
    return (end - start).days / 365.0  # second silent fallback
```

When a caller selects `DayCountConvention.ACT_ACT_ICMA` but forgets to supply the coupon-period anchors, the function quietly degrades to ACT/365F and returns a *similar-but-wrong* year fraction. The error is small in magnitude (~0.1–0.4% per period) but compounds across coupons.

**Downstream impact (confirmed via MODULE_HEALTH §`fixed_income/bond.py` audit):** `FixedRateBond.treasury_note` calls `year_fraction(..., ACT_ACT_ICMA, ...)` *without* passing `ref_start`, `ref_end`, `frequency`. So every US Treasury note priced via this path is silently using ACT/365F:

- Semi-annual UST coupons come out as 1.9836 or 2.0164 per 100 instead of the canonical exact 2.0000.
- Accrued interest is wrong by ~0.8% mid-period.
- Par-yield round-trip lands at 99.9998 (5y) or 99.9995 (30y) instead of exactly 100.

UST is quoted in 32nds (~3.1 bp). The error is **observable in market quotes**.

**Fix shape:** raise `ValueError` instead of falling back. Multi-slice because all 72 ICMA call-sites that don't pass refs need to be located + fixed (likely many).

**Slicing proposal:**
- *Slice 1*: add a `strict_icma` flag (default `False` — keep current behaviour). Tests: ref-dates-present passes; ref-dates-absent raises under `strict_icma=True`.
- *Slice 2..N*: one slice per affected caller — switch caller to pass ref dates; verify with a hand-calc test (e.g. "par UST gives 2.0000 per coupon").
- *Final slice*: flip default to `True`, remove flag.

This pattern surfaces every miscalibrated bond in the library one at a time, with a failing test per call-site.

#### B2 — `_act_act_icma(frequency=0)` raises `ZeroDivisionError`  *[LOW]*

**Location:** `day_count.py:174`.

```python
return (end - start).days / (period_days * frequency)
```

When `frequency=0` is passed (caller error or deserialised-bad-config), the second guard (`period_days <= 0`) doesn't catch it because `period_days > 0`. We hit `divide-by-zero`.

**Repro (confirmed live):**
```
_act_act_icma(date(2024,1,1), date(2024,7,1),
              date(2024,1,1), date(2024,7,1), 0)
→ ZeroDivisionError
```

**Fix:** validate `frequency > 0` (and `period_days > 0`) and raise `ValueError` with a clear message. Single slice.

### Documentation / naming issues

- **D1 — `_thirty_360` is misnamed "ISDA 2006".** The docstring labels the rule set "ISDA 2006" but actually implements 30/360 US (SIA Bond Basis). Pure ISDA 2006 §4.16(f) "Bond Basis" does *not* include the end-of-Feb adjustment. Code is correct for US bonds; the *citation* is wrong. Trivial doc fix.
- **D2 — `date_from_year_fraction` uses 365.25 regardless of convention.** That's "calendar years" (Julian), not "ACT/365 years" or any other. Doesn't round-trip with any specific day-count. The contract is fine for charts and approximate alignment, but the docstring should explicitly say "calendar year (365.25 days)" so callers don't expect convention-aware behaviour.
- **D3 — `business_days_between` uses `(start, end]`** — settlement counts, trade does not. Reasonable but disagrees with the literal Anbima/B3 `[t, T)` for BRL DU. Should be documented explicitly.

### Test gaps

The existing test file covers ACT/360, ACT/365F, 30/360 (regular + d=31), and BUS/252 well. Missing:

- **30E/360** — zero tests. Convention used for Eurobonds, Bunds, EUR corporates.
- **ACT/ACT ISDA cross-year** — zero tests. Year-boundary split is non-trivial.
- **ACT/ACT ICMA** — zero tests. The most important convention for govt bond markets. A test like "par UST 5y semi-annual → exact 0.5 per coupon" would catch B1 immediately.
- **30/360 end-of-Feb edge cases** — zero tests. e.g. Jan 31 → Feb 28 (non-leap) or Feb 29 (leap).
- **`date_from_year_fraction`** — zero tests. Function has no coverage at all.

---

## A.2 — `core/calendar.py`

**Purpose:** Business-day calendars for 39 currencies + `JointCalendar` for multi-cal intersections + `BusinessDayConvention` enum (FOLLOWING / MODIFIED_FOLLOWING / PRECEDING / MODIFIED_PRECEDING / UNADJUSTED) + `get_calendar(ccy)` factory.

**Internal deps:** None. Pure datetime arithmetic + the Anonymous Gregorian Easter algorithm + Meeus Julian (orthodox Easter). Truly L0.

**Size:** 1316 lines, 38 calendar classes, ~30 holiday-rule helpers.

**Test file:** `python/tests/test_calendar.py` (133 lines, 1 calendar tested out of 39).

### Status: ⚠️ Real bugs found

### Confirmed bugs

#### B1 — Wrong weekend-substitute rule for UK / AU / NZ / CA  *[HIGH]*

**Location:** `calendar.py:93-100` (base `_observe`) + `LondonCalendar`, `AUDCalendar`, `NZDCalendar`, `CADCalendar`.

The base `Calendar._observe(d)` implements the **US** rule:
- Saturday → previous Friday
- Sunday → next Monday

That is correct for US federal holidays (5 U.S.C. § 6103) but **wrong** for:
- **UK** — Banking and Financial Dealings Act 1971: Sat/Sun → next available working day (i.e. Saturday → Monday, NOT Friday).
- **AU** — Public Holidays Act: same rule, Sat → Mon.
- **NZ** — Holidays Act 2003: same rule.
- **CA** — federal/provincial bank holidays: same rule.

**Live repro:** Christmas 2021 (Dec 25 Saturday) and Christmas 2027 (Dec 25 Saturday). The wrong dates are produced by all four calendars:

```
London 2021-12-24 Fri: is_holiday=True  ❌  (should be business day)
London 2021-12-27 Mon: is_holiday=True  ✓   (Christmas-observed-Mon by happy accident
                                              from Boxing-Day-Sunday → Monday)
London 2021-12-28 Tue: is_holiday=False ❌  (should be Boxing-Day-observed-Tue)
```

Same pattern on AUD, NZD, CAD. Recurs whenever Dec 25 falls on Saturday — 2021, 2027, 2032, etc. — and for other holidays whose date falls on Saturday in the locale (Australia Day Sat 26 Jan → wrong Fri Jan 25 in AUD).

**Fix shape:**
1. Add a class-level `_observe_rule` constant or override per locale (US-style vs UK-style vs "no substitution" for TARGET/Tokyo).
2. Bump tests dramatically — for each substitute-rule calendar, test a Sat-Christmas year (2021 or 2027) explicitly.

Multi-slice — one slice per affected calendar to keep blast radius small.

#### B2 — Tokyo has no substitute-day rule at all  *[MEDIUM]*

**Location:** `calendar.py:225-246` (`TokyoCalendar._compute_holidays`).

Japanese Public Holiday Act § 3.2: "if a national holiday falls on Sunday, the next non-holiday day is a holiday in place of it" (振替休日, *furikae kyūjitsu*). None of TokyoCalendar's holidays use `_observe` — Sunday holidays are simply lost.

**Impact:** Lower than B1 because (a) the multi-day 1-3 Jan New Year block absorbs most year-start cases and (b) JPY FX/IR has fewer Saturday-on-holiday cases than USD/GBP. But every year where (say) Showa Day Apr 29 falls on Sunday, JPY funding curves silently disagree with market conventions.

**Fix:** Apply `furikae kyūjitsu` rule to single-day Tokyo holidays.

#### B3 — `_nth_weekday` silently spills into adjacent month  *[LOW]*

**Location:** `calendar.py:103-108`.

```python
@staticmethod
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    days_to_add = (weekday - first.weekday()) % 7
    first_occurrence = first + timedelta(days=days_to_add)
    return first_occurrence + timedelta(weeks=n - 1)
```

No validation of `n`. Live repro:

```
5th Mon of Feb 2024 → 2024-03-04 Mon   (Feb 2024 has only 4 Mondays)
0th Mon of Feb 2024 → 2024-01-29 Mon   (n=0 rolls back)
-1th Mon of Feb 2024 → 2024-01-22 Mon  (negative rolls back further)
```

The "5th" case is a real concern if any new calendar tries to encode something like "5th Friday of the month" (DST transitions in some jurisdictions). The `n <= 0` cases are pure defensive — no caller exercises them today.

**Fix:** Validate `1 <= n <= 5`; raise `ValueError`; return the result only if it stays in `month`.

#### B4 — `is_holiday` Y-1 spill window is defensive but fragile  *[LOW]*

**Location:** `calendar.py:31-34`.

The current window is `(d.year, d.year + 1)`. Catches the typical case (Jan 1 Sat → Dec 31 prev year observed) because that observed date is generated by *next* year's `_compute_holidays`. But it does **not** catch the symmetric case where year Y's compute generates a date in year Y+1. No current calendar emits such a Y→Y+1 spill, so this is dormant.

**Fix:** Widen window to `(d.year - 1, d.year, d.year + 1)` for safety. Tiny cost (one extra year cached), zero behaviour change today.

#### B5 — `JointCalendar` mutates components' `_holiday_cache`  *[LOW]*

**Location:** `calendar.py:1233-1239`.

```python
def _compute_holidays(self, year: int) -> set[date]:
    holidays = set()
    for cal in self._calendars:
        if year not in cal._holiday_cache:
            cal._holiday_cache[year] = cal._compute_holidays(year)
        holidays |= cal._holiday_cache[year]
    return holidays
```

Pollutes each component's private cache. If the same component calendar is shared across threads OR is used standalone elsewhere, there's a (cooperative-multitasking) race on `_holiday_cache`. Not a Python-thread-safety bug (GIL covers the dict ops), but the implicit aliasing makes ownership murky.

**Fix:** Use a local computation only; don't reach into `cal._holiday_cache`. Cost: re-computing the component's holidays when used inside a Joint — but joint cals are rare and small.

### Test gaps — major

`test_calendar.py` exercises **only** `USSettlementCalendar` (NYSE/SIFMA holidays + adjust conventions). The other 38 calendars — TARGET, London, Tokyo, all EM cals — have **zero direct test coverage**.

Highest-impact additions:
- TARGET (Easter shifting, no-substitute rule, 26 Dec)
- London + AUD + NZD + CAD on Sat-Christmas years (2021, 2027) — would catch B1 immediately.
- SaoPaulo Carnival + Independence Day (already lightly covered in `test_day_count.py`).
- Mumbai Diwali, Beijing Spring Festival — these are openly *not* implemented (lunar calendars are accepted approximations); document the gap in a test that asserts what IS covered.

### Accepted approximations (already documented in code)

- Tokyo vernal equinox = Mar 21 fixed; autumnal = Sep 23 fixed (actual dates vary ±1 day).
- Tel Aviv Hebrew calendar approximated by fixed Gregorian dates.
- Islamic-calendar holidays in MENA calendars (RiyadhCalendar, etc.) — omitted.
- Chinese lunar New Year, Mid-Autumn — omitted in BeijingCalendar, HongKongCalendar, etc.

These are openly disclaimed and acceptable for the library's current scope (G10 + EM curves, not regional equity).

---

## A.3 — `core/currency.py`

**Purpose:** G10 currency enum + `CurrencyPair` with ACI market-convention base/quote priority + settlement lag + CIP forward formulas + `all_g10_pairs()`.

**Internal deps:** None (`DiscountCurve` is TYPE_CHECKING only). True L0.

**Size:** 138 lines.

**Test file:** `python/tests/test_currency.py` (80 lines, 4 test classes).

### Status: 📝 No bugs found; test gaps

### Correctness review

- `Currency` enum: 10 G10 ISO codes (EUR, GBP, AUD, NZD, USD, CAD, CHF, NOK, SEK, JPY). ✓
- `_BASE_PRIORITY`: matches ACI standard (EUR > GBP > AUD > NZD > USD > CAD > CHF > NOK > SEK > JPY). ✓
- `from_currencies` correctly picks the higher-priority base. ✓
- `settlement_lag` correctly handles inverse-key lookup so `CAD/USD` and `USD/CAD` both return T+1. ✓
- `forward_rate(spot, r_base, r_quote, T) = spot × exp((r_quote − r_base) × T)`: correct continuous-rate CIP for a quote-per-base convention. ✓
- `forward_rate_from_curves`: `S × df_base / df_quote`. ✓
- `CurrencyPair` hash/equality look standard. ✓

### Minor concerns (not bugs)

- **`is_ndf` is dead code** given the strict `Currency` enum. The property asks "is base or quote *not* in G10?" but the type system prevents non-G10 construction. Two ways to fix: (a) add an `EMCurrency` enum & accept it in `CurrencyPair` so the check has teeth, or (b) remove `is_ndf` until EM-FX lands. Pick when EM-FX work starts.
- Docstring of `forward_rate` doesn't state that `r_base`/`r_quote` are *continuously* compounded — caller could plausibly plug in simple rates and get a small but systematic error.

### Test gaps

- **`settlement_lag`**: not tested. Especially the USD/CAD T+1 exception.
- **`forward_rate` / `forward_rate_from_curves` / `forward_points`**: untested. These are the core FX formulas.
- **`is_ndf`**: untested (will always be `False` given the current enum, but the test would document the contract).
- **`all_g10_pairs`**: untested. Should assert count = 45 and all pairs in market convention.

Single test-coverage slice to land all four. No source changes needed.

---

## A.4 — `core/schedule.py`

**Purpose:** Generate cashflow schedules between `start` and `end` at a given `Frequency`, with stub type (SHORT/LONG × FRONT/BACK), EOM rule, and optional calendar adjustment.

**Internal deps:** `pricebook.core.calendar` (for adjustment). External: `dateutil.relativedelta`.

**Size:** 127 lines.

**Test file:** `python/tests/test_schedule.py` (164 lines).

### Status: ⚠️ One real bug found

### Confirmed bugs

#### B1 — EOM anchored to `end` (not `start`) in front-stub backward generation  *[HIGH for amortising / off-EOM-end trades]*

**Location:** `schedule.py:85-92` + `_add_months:25-30`.

The backward generator initialises `current = end` and steps backward via `_add_months(current, -months, eom)`. Inside `_add_months`, the EOM check is:
```python
if eom and d == _end_of_month(d):
    result = _end_of_month(result)
```
which is anchored to `d` — i.e. the *current rolling date*. For the front-stub path that's effectively `end`. So when `start` is EOM but `end` is **not** EOM, the EOM rule never fires and interior rolls land mid-month.

**ISDA 2006 §4.10 (End of Month Convention):** *"if the [period start date] is the last day of February, the [period end date] is the last day of February. If the [period end date] is the last day of February, the [next period end date] is the last day of February."* — i.e. EOM anchors to *start*, propagates forward.

**Live repro:**
```
start=2024-01-31 (EOM), end=2024-08-15 (not EOM), semi-annual, SF, eom=True:
  current behaviour: [2024-01-31, 2024-02-15, 2024-08-15]
                                  ^^^^^^^^^^ should be 2024-02-29 (EOM)

start=2024-01-31 (EOM), end=2025-04-15 (not EOM), semi-annual, SF, eom=True:
  current behaviour: [2024-01-31, 2024-04-15, 2024-10-15, 2025-04-15]
                                  ^^^^^^^^^^ should be 2024-04-30 (EOM)
                                              ^^^^^^^^^^ should be 2024-10-31 (EOM)
```

Affected trades: any with EOM start and non-EOM end. Common cases:
- Amortising swaps where end is a calendar quarter-end day-15 instead of EOM.
- Bond schedules where start is the issue date (often EOM) and end is the maturity (often *not* EOM, set by deal economics).
- Cross-currency basis swaps where EOM convention is asymmetric between legs.

**Fix shape:** Pass an explicit `eom_anchor` date into `_add_months` (i.e. always `start`), independent of the rolling `current`. Or, recompute backward rolls forward after the loop. Either way, a focused 1-slice fix with a regression test covering the exact ISDA §4.10 cases above. The forward path is already correct (it anchors on `start` by accident because `current` is initialised to `start`).

### Robustness concerns (not bugs, but worth noting)

- **`months * 30` stub heuristic** at lines 98 and 116: classifies a stub as "tiny" if its day-count < 50% of `months * 30`. For quarterly (months=3) the threshold is 45 days, so a 44-day quarterly stub is silently merged on LONG_FRONT. The 30-day approximation also doesn't account for actual month lengths; could swing  ±10% per case.
- **Post-adjustment dedup**: after `calendar.adjust(d, convention)` two consecutive raw rolls could both shift to the same business day (dense stubs, holiday clusters) or even invert (if `MODIFIED_FOLLOWING` pushes one forward and the next backward). The code returns the adjusted list as-is. Audit critic flagged this.
- **`WEEKLY` ignores `eom`** — silently. Probably fine (EOM ⊥ weekly) but documenting would help.
- **`stub` default is `SHORT_FRONT`** — sensible default for fixed-income; sometimes surprising for FX/EQ. Mention in docstring.

### Test gaps

- **EOM anchoring bug coverage** — currently `test_eom_preserved` only checks start=EOM **and** end=EOM, so the bug is invisible. A test with `start=2024-01-31`, `end=2025-04-15`, semi-annual would catch it immediately.
- **Post-adjustment ordering/dedup**: no test. Build a scenario where two raw rolls collide post-adjust.
- **WEEKLY**: no test (frequency=0 branch).
- **LONG_BACK**: not tested (only LONG_FRONT).
- **stub heuristic boundary**: no test for the "44-day quarterly stub merges, 46-day doesn't" case.

---

## A.5 — `core/solvers.py`

**Purpose:** 1D root finders — `newton` (Newton-Raphson), `secant`, `halley` (cubic convergence using f, f', f''), `itp` (Interpolate-Truncate-Project bracketing), `brentq` (Brent's method). All return `SolverResult` except `brentq` (returns `float` for backward compat).

**Internal deps:** None.

**Size:** 247 lines. Test file: 152 lines.

### Status: ⚠️ 2 minor bugs

### Confirmed bugs

#### B1 — `SolverResult.to_dict()` returns mutable shared `__dict__`  *[LOW]*

**Location:** `solvers.py:27-28`.

```python
def to_dict(self) -> dict:
    return vars(self)
```

`vars(self)` returns `self.__dict__` directly. Mutating the returned dict mutates the dataclass.

**Live repro:**
```
r = SolverResult(root=1.0, ...)
d = r.to_dict()
d['root'] = 999.0
# r.root is now 999.0
```

**Fix:** `return dict(vars(self))` (one-character fix).

#### B2 — `itp` silently overrides user `maxiter` cap and misreports iteration count  *[LOW]*

**Location:** `solvers.py:137, 174`.

```python
for i in range(max(maxiter, n_max)):     # line 137
    ...
return SolverResult(..., iterations=maxiter, ...)  # line 174
```

Two issues:
1. The loop uses `max(maxiter, n_max)` where `n_max = ceil(log2((b-a)/(2*tol))) + 1`. For tight `tol` and wide bracket, `n_max` can exceed `maxiter`. User's request for "no more than 5 iterations" is silently increased.
2. When the loop budget is exhausted, the result reports `iterations=maxiter` instead of the actual loop counter `i`. Reported count disagrees with the actual evaluations.

**Live repro:**
```
itp(f, 0.0, 1.5, tol=1e-14, maxiter=5)
# Actually runs 7 iterations (n_max=7 > maxiter=5)
# Reports iterations=5
```

**Fix:** Either (a) honour `maxiter` strictly and emit a "did-not-converge" warning, or (b) document that ITP's worst-case bound supersedes `maxiter` and at least set `iterations=i+1` in the timeout-return path.

### Robustness concerns (not bugs)

- **`newton`, `secant`, `halley`**: silently `break` on near-zero denominator (|f'| < 1e-15) without warning. The `converged` field captures it, but a caller iterating multiple seeds gets no signal which one failed and why. Add `RuntimeWarning`.
- **`brentq` warning threshold** at line 239 is hard-coded `tol * 1000`. Inconsistent: ITP/Newton don't have a similar "loose-OK" tolerance. Pick one policy.
- **`brentq` returns `float`** — breaks the `SolverResult` contract of every other solver in the module. Docstring acknowledges this is for back-compat. Worth tracking for a future migration.
- **NaN/Inf handling**: none of the solvers explicitly check for NaN/Inf. If `f(x)` returns NaN, the abs-comparison silently returns False, and the iteration continues until maxiter exhaustion. Result: `root=NaN, converged=False`. Mostly survives but provides no diagnostic.

### Test gaps

- No tests for `to_dict` (would catch B1).
- No tests for `itp` with tight tolerance / small maxiter (would catch B2).
- No tests for the maxiter-exhausted return path on any solver.
- No tests for NaN/Inf propagation.
- No tests for near-zero-derivative cases in `newton`/`halley`/`secant`.

---

## A.6 — `core/interpolation.py`

**Purpose:** 5 1-D interpolators: `Linear`, `LogLinear` (linear-in-log-y, the discount-curve standard), `CubicSpline` (scipy natural), `MonotoneCubic` (Fritsch-Carlson + Hyman filter), `Akima`. Each subclasses an `Interpolator` ABC and the module provides a `create_interpolator(method, x, y)` factory.

**Internal deps:** None (NumPy + scipy.interpolate). True L0.

**Size:** 271 lines.

### Status: 📝 No bugs found

### Correctness review

- **Base `Interpolator`**: validates length, ≥2 points, strictly-increasing x. Left extrapolation is hardcoded flat. Right extrapolation is overridable; only `LogLinearInterpolator` actually overrides (with the standard "extend last segment's slope in log space" → piecewise-constant forward in the last segment). ✓
- **`LinearInterpolator`**: standard t-blend on `[x_i, x_{i+1}]`. ✓
- **`LogLinearInterpolator`**: linear in `log(y)`, equivalent to piecewise-constant forward rates between knots. Slope-continued right extrapolation is correct. ✓
- **`CubicSplineInterpolator`**: defers to `scipy.interpolate.CubicSpline(..., bc_type="natural")`. ✓
- **`MonotoneCubicInterpolator`**:
  - Fritsch-Carlson interior slopes via harmonic mean (set to 0 when adjacent secants disagree in sign or one is flat). ✓
  - One-sided endpoint slopes. ✓
  - Hyman filter `α² + β² ≤ 9` (de Boor / Hyman scaling). ✓
  - **Subtle**: filter is applied per segment in a single forward pass, mutating `slopes[i+1]` in place; the next iteration reads the already-modified slope. This is the standard "naive Hyman" approach; can produce slightly different slopes from a two-pass "compute-then-clip-all" implementation in pathological cases. Not a bug — matches the textbook formula — but worth knowing if comparing against another library.
- **`AkimaInterpolator`**: Akima slopes with ghost-secant boundary treatment, fallback to mid-secant when both weights are zero. ✓
- **`_find_segment`**: `np.searchsorted` with clamping to `[0, n-2]`. ✓
- **`create_interpolator` factory**: covers all 5 methods. ✓

### Design observations (not bugs)

- **Right extrapolation default is flat for cubic methods** — `CubicSpline`, `MonotoneCubic`, `Akima` all inherit the base `_extrapolate_right` that returns `self._y[-1]`. This introduces a derivative kink at the right boundary, which can be surprising for callers who interpolated with a cubic method expecting smooth extrapolation. The behaviour is acceptable for curves (rates are typically extrapolated flat) but should be explicit in the docstrings.
- **Left extrapolation is not overridable** — there's no `_extrapolate_left` hook symmetric to `_extrapolate_right`. Niche but reduces customisability.
- **No `to_dict`** on interpolators — fine, curves serialise via the method enum and rebuild.

### Test gaps

- **`AkimaInterpolator` has zero tests** (entire `TestAkima` class is missing from `test_interpolation.py`). The `TestFactory::test_all_methods` exercises it weakly via knot-recovery only.
- **`LogLinearInterpolator` right-extrap slope continuation** — the override is non-trivial and untested. Easy to add.
- **Monotone Hyman clipping path** — no test specifically covers a case where `α² + β² > 9` triggers `tau` scaling. Build a configuration with a sharp slope change.
- **2-point edge cases** — Akima and MonotoneCubic have n=2 guards (line 224/226 for Akima; loop range for MonotoneCubic) but no direct tests.
- **`Interpolator._find_segment` boundary behaviour** at exact knot values is untested at the deepest internal level (though indirectly covered by `test_at_knots`).

---

## A.7 — `core/approximation.py`

**Purpose:** Approximation-theory primitives:
- `chebyshev_interpolate` / `_clenshaw` / `ChebyshevInterpolant` — Chebyshev-Lobatto polynomial interpolation + Clenshaw eval.
- `pade_approximant` / `PadeApproximant` — Padé [L/M] rational approximation from Taylor coefficients.
- `richardson_table` / `RichardsonTable` — full Romberg/Richardson extrapolation table.
- `bspline_basis` — Cox-de Boor recursion.

**Internal deps:** None. True L0.

**Size:** 274 lines.

### Status: 📝 No real bugs; same `to_dict` mutation pattern as A.5

### Correctness review

- **Chebyshev** path: standard Chebyshev-Lobatto nodes `x_k = cos(kπ/n)` and DCT-I formula for coefficients. Endpoint weights `1.0` (vs `2.0` for interior) ✓. Divide-by-2 on first/last coefficients ✓ — standard Chebyshev convention. The DCT is implemented as a double Python loop (O(n²)) — correct but slow; could use `scipy.fft.dct` for a 10-100× speedup at n=50+. Not a bug.
- **Clenshaw** recurrence is the textbook formulation. ✓
- **Padé** linear system — checked the indexing carefully against the matching equations `Σ_{j=1}^{M} c_{L+i-j} q_j = -c_{L+i+1}` for i=1..M, then for the numerator `p_k = Σ_{j=0}^{min(k,M)} c_{k-j} q_j`. Both correct (`A[i,j] = c[L+i-j]`; `b[i] = -c[L+i+1]`; `numer[k] = Σ denom[j] c[k-j]`). ✓
- **Richardson table**: `T[i,j] = (r·T[i,j-1] − T[i-1,j-1]) / (r−1)` with `r = 2^{p·j}`. Standard Romberg formula. ✓
- **Cox-de Boor**: standard recursion with the half-open-on-the-right convention. The right-boundary `x == knots[-1]` evaluating to 0 for the last basis support is the textbook quirk — accepted approximation.

### Confirmed (trivial) bugs

#### B1 — Three `to_dict` methods return mutable shared `__dict__`  *[LOW]*

**Location:** `approximation.py:48-49, 133-134, 202-203`.

Same shape as A.5 B1. `ChebyshevInterpolant.to_dict`, `PadeApproximant.to_dict`, `RichardsonTable.to_dict` all do `return vars(self)`. Caller mutation propagates to the dataclass instance.

**Fix:** One audit slice will fix all four occurrences (3 here + 1 in `solvers.py`). Trivial.

### Test coverage

`test_approximation.py` covers happy paths for all four primitives — polynomial exactness for Chebyshev, exp accuracy, Padé [2/2] for exp, Richardson exact extrapolation, B-spline partition of unity. Adequate; no major gaps.

### Style / non-blocking

- Chebyshev DCT in pure-Python loops — replaceable by `scipy.fft.dct(fx, type=1)` for clean code + significant speedup at high `n`. Defer.
- `bspline_basis` is recursive in pure Python — fine for moderate degrees; could memoise if used hotly (no caller does today).

---

## A.8 — `core/caching.py`

**Purpose:** `CurveCache` (LRU with per-curve-name invalidation), `CalibrationCache` (model-params cache keyed by inputs-hash), `LazyValue` (deferred compute, computed-at-most-once).

**Internal deps:** None. True L0.

**Size:** 169 lines.

### Status: ✅ Clean

- `CurveCache` uses `OrderedDict` with `move_to_end` on hit and `popitem(last=False)` on overflow — textbook LRU. Stats track hits/misses; `hit_rate` short-circuits on zero queries. ✓
- `CalibrationCache` uses tuple `(model_name, inputs_hash)` as the dict key, but the **type annotation** says `dict[str, dict]`. Cosmetic mismatch — the runtime works fine. Worth a one-line type-hint fix.
- `LazyValue` is idempotent + `reset()`. ✓
- 16 tests cover hit/miss/invalidate/LRU/clear/stats + persistence + lazy semantics.

### Slicing items: none required. (Optional type-hint cleanup folded into a "small cleanups" slice later.)

---

## A.9 — `core/protocols.py`

**Purpose:** Structural Protocols for the numerical layer — `RootFinder`, `Integrator`, `OptionPricer`, `MCEngine`, `VolModel`, `VolSurface`, `CharFunc`. All `@runtime_checkable`.

**Internal deps:** Eagerly imports `SolverResult` from `core.solvers` for re-export; uses `TYPE_CHECKING` for `QuadratureResult` (curves.quadrature) and `MCResult` (models.mc_pricer).

**Size:** 129 lines.

### Status: ✅ Clean

- All Protocol definitions follow the same pattern: minimal interface, `...` body, `@runtime_checkable`. Zero implementation logic to audit. ✓
- **Asymmetry note (not a bug):** `SolverResult` is imported eagerly at runtime while the other two result types are TYPE_CHECKING-only. That's because `core.solvers` is at the same dependency layer (no cycle), whereas the others would introduce upward dependencies. Reasonable choice; could be made symmetric (all three TYPE_CHECKING) for consistency, but no behaviour change.

### Slicing items: none.

---

## A.10 — `core/fixings.py`

**Purpose:** `FixingsStore` for daily rate fixings (SOFR, EURIBOR, CPI, ...) with file-based JSON persistence + CSV import + sample-fixings generator.

**Internal deps:** `core.calendar` (TYPE_CHECKING + lazy at call-time for `add_business_days`). True L0.

**Size:** 195 lines.

### Status: ✅ Clean — two robustness concerns, no bugs

- `set/get/has/get_or_raise` straightforward. ✓
- `get_with_lag(rate_name, d, lag, calendar=None)`: with calendar → `add_business_days(d, -lag)` ✓. Without calendar → uses `timedelta(days=lag)` (calendar days, not business days). The docstring says "business-day lag" but silently falls back to calendar lag when calendar is None. Misleading — should either raise (forcing the caller to provide a calendar) or rename the parameter / docstring to "lag in days (business if calendar given, else calendar)". Low impact but a foot-gun.
- `_load_all` doesn't try/except per file — one corrupt JSON crashes the entire store load. Niche; acceptable for a tooling layer but worth noting.
- `load_csv` has per-row error handling with a `skip_invalid` toggle. ✓
- 18 tests cover most APIs. Gaps: `get_with_lag`, `load_csv`, edge cases on persistence (empty store, missing dir).

### Slicing items (defer; non-blocking)

- Clarify `get_with_lag` semantics: either require `calendar` (raise on None) or rename to `get_with_calendar_day_lag` fallback. Single test + docstring + signature tweak.

---

## A.11 — `core/serialisable.py`

**Purpose:** Mixin / decorators (`Serialisable`, `@serialisable`, `@serialisable_convention`) that auto-derive `to_dict` / `from_dict` from constructor type hints. Registry-based dispatch via `from_dict(d)`. Schema versioning via `_SERIAL_SCHEMA_VERSION` (added in G1 P3 Slice 2).

**Internal deps:** None (no upward imports). True L0.

**Size:** 428 lines.

### Status: ⚠️ Several documented-design-limitation bugs

The module is solid for the common case (instrument with primitive / single-Serialisable fields) but has known limits at the edges. None are catastrophic; all matter for specific shapes of serialised data.

### Confirmed bugs / limitations

#### B1 — `_deserialise_atom` only reconstructs `list[date]`; `list[SomeSerialisable]` silently returns raw  *[MEDIUM]*

**Location:** `serialisable.py:210-214`.

```python
if get_origin(hint) is list:
    args = get_args(hint)
    if args and args[0] is date:
        return [date.fromisoformat(x) if isinstance(x, str) else x for x in v]
```

That's the whole list handler. Any other parameterised list — `list[Quote]`, `list[CashFlow]`, `list[Schedule]` — falls through to `return v` (raw list of dicts) at line 217. Classes whose constructor takes a polymorphic list field get a wrong-typed value back unless they override `from_dict`.

**Fix shape:** detect `_SERIAL_TYPE` on `args[0]` and recursively `from_dict` each element. Single small slice + round-trip test using a class with a `list[SerialisableThing]` field.

#### B2 — `Union[A, B, None]` (3+ types) returns raw value, skipping reconstruction  *[MEDIUM]*

**Location:** `serialisable.py:172-178`.

```python
if origin is Union or isinstance(hint, _types.UnionType):
    args = [a for a in get_args(hint) if a is not type(None)]
    if len(args) == 1:
        hint = args[0]
    else:
        return v
```

Optional[T] (i.e. `T | None`, len=1 after stripping `NoneType`) unwraps correctly. But `A | B | None` returns `v` raw — silently bypasses reconstruction. Classes with polymorphic discriminated-union fields (e.g. `underlying: Stock | Index | None`) need a manual `from_dict` override.

**Fix shape:** if the value is a dict with `"type"`, dispatch via the global registry. Otherwise return v as today. Slice covers this + a round-trip test.

#### B3 — Registry re-registration is silently a no-op  *[LOW]*

**Location:** `serialisable.py:102`.

```python
if key and key not in _REGISTRY:
    ...
    _REGISTRY[key] = cls
```

A second `_register(cls2)` with the same `_SERIAL_TYPE` does nothing. Module reloads (e.g. during `importlib.reload` in tests / notebooks) leave the stale class in the registry — `from_dict` rebuilds via the OLD class. This is "fine until it bites you in dev workflow."

**Fix shape:** overwrite + emit a `DeprecationWarning` so the reload path is at least visible.

#### B4 — `_serialise_atom` checks `.item()` before `to_dict` — collapses any object with a callable `.item()`  *[LOW]*

**Location:** `serialisable.py:150-153`.

```python
if hasattr(v, "item") and callable(v.item):
    return v.item()
if hasattr(v, "to_dict"):
    return v.to_dict()
```

NumPy scalars and a user class that happens to define `.item()` (e.g. an `EnumMember.item()` helper) get duck-typed as numpy first. The check should be `isinstance(v, (np.generic, ...))` or moved AFTER the `to_dict` check.

**Fix shape:** narrow `.item()` to `isinstance(v, np.generic)`. Trivial.

#### B5 — `CurrencyPair` round-trip assumes exactly one `/`  *[LOW]*

**Location:** `serialisable.py:206`.

```python
base_str, quote_str = v.split("/")  # raises ValueError on != 1 slash
```

Inputs like `"EUR/USD/JPY"` or `"EURUSD"` produce confusing unpack errors. Use `v.split("/", 1)` + explicit length check + a clear error message.

#### B6 — `Serialisable.from_dict` does a bare `d["params"]` lookup  *[LOW]*

**Location:** `serialisable.py:261`.

```python
p = d["params"]   # KeyError if absent
```

A malformed payload (truncation, half-written file, schema-version mismatch with a v2 that dropped `params`) gets a generic `KeyError: 'params'` instead of a structured `ValueError("Bad payload for ClassName: missing 'params'")`.

#### B7 — Enum deserialisation may fail for int-valued enums delivered as string  *[LOW]*

**Location:** `serialisable.py:187-190`.

```python
if isinstance(hint, type) and issubclass(hint, Enum):
    if isinstance(v, hint):
        return v
    return hint(v)
```

If `EnumValue.value` is `int` (e.g. `Frequency.SEMI_ANNUAL = 6`) but a non-JSON layer delivers it as `"6"`, `hint("6")` raises. JSON itself preserves int-as-int so this is dormant inside the library's JSON path. Worth a `try int(v)` fallback for robustness against YAML/CSV / arbitrary string payloads.

### New concern from G1 P3 Slice 2 (just landed)

`_check_schema_version` raises on `v > expected` but silently accepts `v < expected`. For now there's only v1 in the wild, so this is correct. **When v2 lands**, we need a migration hook (or an explicit "no migration available → raise" policy) so old payloads aren't silently misinterpreted by newer code. Doc the contract.

### Test coverage

- Existing tests (in `test_serialisable.py`) cover the happy path and the new schema-version semantics. ✓
- Gaps: no round-trip for `list[SomeSerialisable]` (would catch B1), no `Union[A, B, None]` test (would catch B2), no malformed-payload test (would catch B6), no Enum-from-string test (B7).

---

## A.12 — `core/serialization.py`

**Purpose:** Public-facing facade over `core.serialisable`. Eagerly imports all `@serialisable` modules to populate the registry. Provides `to_dict`/`from_dict`/`to_json`/`from_json`, registry helpers (`registered_types`, `get_instrument_class`), plus legacy aliases (`instrument_to_dict`, `trade_to_dict`, ...) and legacy loaders (`load_trade`, `load_portfolio`) that handle pre-G1 dict formats.

**Internal deps:** `core.serialisable` + a curated import list of 24 modules.

**Size:** 181 lines.

### Status: ⚠️ Curated-import-list footgun + duplicated CurrencyPair parse

### Confirmed bugs

#### B1 — `_ensure_loaded` is a hand-maintained whitelist of 24 modules  *[MEDIUM]*

**Location:** `serialization.py:34-62`.

```python
def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    _loaded = True
    import pricebook.fixed_income.swap
    import pricebook.fixed_income.bond
    ...                                  # 24 imports
```

Every new module that uses `@serialisable` MUST be added here, or its types won't be in `_REGISTRY` and `from_dict` raises `Unknown type ...`. This is a *silent maintenance trap* — the failure mode is "round-trips work for some types but not others depending on which modules happened to be imported".

I count many `@serialisable` / `@serialisable_convention` usages across the codebase (sovereign_cds, cds_conventions, market_conventions, rate_index, composite_convention, repo_specialness, esg_bonds, sukuk, sovereign_bonds, inflation_indices, supranational, currency_conventions, em_curve_builder, ...). Of those, ONLY the ones imported by some path in `_ensure_loaded` get registered automatically. Quietly broken for the rest.

**Fix shape:** use `pkgutil.walk_packages` + `importlib.import_module` to walk `pricebook.*` and import every submodule once. Slight startup cost (~tens of ms) for total coverage. Or: keep the curated list but add a CI check that every `@serialisable*` callsite has a corresponding `import` somewhere in `_ensure_loaded`'s transitive closure.

#### B2 — Duplicated `CurrencyPair.split("/")` foot-gun  *[LOW]*

**Location:** `serialization.py:113-116`.

```python
def deserialise_currency_pair(s: str):
    base_s, quote_s = s.split("/")  # explodes on != 1 slash
    return CurrencyPair(Currency(base_s), Currency(quote_s))
```

Same bug as A.11 B5, duplicated in this facade. Fix together with B5.

### Style / non-blocking

- Many legacy aliases (`instrument_to_dict`, `trade_to_dict`, …) all just alias `to_dict`. Considered dead deprecation surface but cheap to keep. Removal would be a Gate 3 cleanup.
- `load_trade` / `load_portfolio` accept multiple legacy formats — well-commented; behaviour is reasonable but the multiple-format logic should be considered as part of the schema-versioning story.

---

## Legacy-debt ledger

Every fix that keeps a backwards-compat fallback writes a row here so we don't lose track of the eventual tightening. Each entry: what the fallback does, why it exists, what will tighten it, what condition triggers the tightening.

### LD.1 — `year_fraction(strict_icma=False)` default

**Location:** `core/day_count.py:year_fraction`, `_act_act_icma`.
**Added:** v0.894 (A.1 B1 Slice 1).
**Behaviour:** Default `strict_icma=False`. ACT/ACT ICMA silently falls back to ACT/365F when `ref_start`, `ref_end`, or `frequency` is missing or invalid (`frequency<=0`, `period_days<=0`).
**Why kept:** flipping default to True would break every legacy caller that hadn't been migrated to pass refs. Multi-slice migration plan in the audit doc.
**Tightens when:** every ICMA call-site in `pricebook/*` passes refs (and tests assert exact ICMA values per ICMA 251.1). Then flip default to `True` and ideally remove the flag entirely.
**Trigger condition:** zero strict-mode failures in CI when default is set to True for one suite run. The final-slice commit will rip the flag out.

### LD.2 — `FixedLeg.coupons_per_year is None` for `Frequency.WEEKLY`

**Location:** `fixed_income/fixed_leg.py:69-93`.
**Added:** v0.896 (A.1 B1 Slice 3).
**Behaviour:** `Frequency.WEEKLY` has `.value == 0`, so we set `coupons_per_year = None` and pass that to `year_fraction(..., frequency=None)`. For ACT/ACT ICMA this triggers the legacy ACT/365F fallback (LD.1). For other day-counts no effect.
**Why kept:** WEEKLY + ICMA is a nonsensical combination — no real bond uses it. Raising in this case would be a defensive measure; the actual customer impact is zero.
**Tightens when:** LD.1 flips strict by default. At that point WEEKLY + ICMA will raise `ValueError` automatically (from `_act_act_icma`). Add a unit test asserting that error message.

### LD.3 — `FixedRateBond._ytm_time_to` fallback paths

**Location:** `fixed_income/bond.py:_ytm_time_to`.
**Added:** v0.897 (A.1 B1 Slice 4).
**Behaviour:** For ACT/ACT ICMA, computes period-counts directly. Falls back to `year_fraction(settle, target, ACT_ACT_ICMA)` (which hits LD.1) when any of:
- `coupons_per_year is None` (WEEKLY)
- `target` not in `coupon_dates` (custom payment dates not on the coupon schedule)
- `settle` is before the first coupon date or after the last
**Why kept:** these branches are defensive — none of the standard pricer call paths hit them today.
**Tightens when:** every fallback case audited. Either reachable cases get explicit handling, or unreachable cases get a `raise ValueError` to flag a future caller that strays.

### LD.4 — `_ensure_loaded` silently records import failures

**Location:** `core/serialization.py:_ensure_loaded`, `_failed_imports`.
**Added:** v0.891 (A.12 B1 fix).
**Behaviour:** `pkgutil.walk_packages` + `importlib.import_module` catches all exceptions, records `(module_name, exception_class_name)` in `_failed_imports`. Currently empty. Subsequent calls to `from_dict` for those types raise "Unknown type ..." — same failure mode as before the fix but with better diagnostic information available.
**Why kept:** an import failure in some peripheral module shouldn't crash the entire serialisation layer for unrelated types. Recording the failure is enough; the CI test (`test_auto_discovery_succeeds_with_no_import_failures`) catches any non-empty list.
**Tightens when:** the per-test assertion is enough — production-runtime callers shouldn't need different behaviour. Could elevate to a `warnings.warn` if a real failure ever surfaces. Reconsider when a peripheral-module import failure actually happens.

### LD.5 — `_act_act_icma` multi-period spans (NOT yet handled)

**Location:** `core/day_count.py:_act_act_icma`.
**Status:** unchanged; `_act_act_icma` correctly handles only single-period spans (one full coupon period or a stub within one). Multi-period spans silently fall back to ACT/365F even with refs supplied.
**Why kept:** the single-period path is what `FixedLeg` needs (each cashflow is one accrual period). Multi-period spans appear in `FixedRateBond._ytm_time_to` — which I solved by bypassing `year_fraction` entirely and computing periods directly.
**Tightens when:** if a future caller actually needs `year_fraction(settle, payment, ACT_ACT_ICMA)` over a multi-period span with refs, extend `_act_act_icma` to implement ICMA 251.2's stub-plus-full-period decomposition. No caller needs this today.

### LD.6 — `DiscountCurve.from_dict` defaults to `LOG_LINEAR` when `interpolation` key absent

**Location:** `core/discount_curve.py:_dc_from_dict`.
**Added:** v0.899 (B.1 B2 fix).
**Behaviour:** pre-fix payloads on disk lack the `interpolation` key (it was silently dropped on serialise). When the key is absent, `from_dict` falls back to `LOG_LINEAR` — the constructor default.
**Why kept:** absent the back-compat default, every persisted DiscountCurve from before v0.899 would fail to load. The fix shipped together with the back-compat shim so existing data keeps working.
**Tightens when:** schema_version bumps to v2 with an explicit "interpolation is mandatory" semantic. Then absent-key payloads on disk get migrated explicitly (re-load → re-serialise) before v2 is deployed.

### LD.7 — `PricingContext.from_dict` back-compat defaults

**Location:** `core/pricing_context.py:_ctx_from_dict`.
**Added:** v0.902 (D.1 B2 fix).
**Behaviour:** every previously-dropped field (`discount_curves`, `inflation_curves`, `repo_curves`, `reporting_currency`, `stochastic_credit_models`, `credit_vol_surfaces`, `credit_correlations`, `numerical_config`) is read via `p.get(name, default)`. Older payloads have NONE of these keys and load as empty dicts / `"USD"` / `None`.
**Why kept:** any persisted pre-v0.902 context loads cleanly; downstream code sees the documented "default to empty" semantic.
**Tightens when:** all on-disk pre-v0.902 payloads are re-serialised under v2 schema. After that, missing-key in v2 payload becomes a hard error.

### LD.8 — `_check_schema_version`: absent → v1

**Location:** `core/serialisable.py:_check_schema_version`.
**Added:** v0.889 (G1 P3 Slice 2).
**Behaviour:** a payload with no `schema_version` key is treated as v1 silently. Future-version payloads raise; older-version payloads are accepted (caller's responsibility to migrate).
**Why kept:** the universe of pre-G1-P3 payloads has no version key. Treating absent as v1 is the only way to read them.
**Tightens when:** a class bumps to v2 AND wants to refuse v1 payloads (no migration). Then `_check_schema_version` could be extended with a minimum-version check. Not needed for any class today.

### LD.9 — `@serialisable_convention.from_dict` accepts both flat and envelope formats

**Location:** `core/serialisable.py:serialisable_convention` (`cls_from_dict`).
**Added:** pre-existing; not added by a fix but flagged here for completeness.
**Behaviour:** convention objects serialise to a flat dict (no `{"type": ..., "params": ...}` envelope). The `from_dict` accepts EITHER format — if the input has `"type"` and `"params"` keys, it unwraps; otherwise it treats the dict as flat.
**Why kept:** allows nested convention objects inside envelope-format parent dicts. Useful in practice but means the same logical payload has two equally-valid wire formats.
**Tightens when:** Gate 2 design decides on a single canonical wire format. Right now both formats coexist and `to_dict` always emits the flat form; the envelope-accept path is purely back-compat.

### LD.10 — ~42 custom `to_dict` overrides emit no `schema_version`

**Location:** option/credit modules — `options/asian_option.py`, `options/barrier_option.py`, `options/autocallable.py`, `options/swaption.py`, `options/capfloor.py`, `options/cliquet.py`, `options/tarf.py`, `options/basket_option.py`, `options/american_option.py`, `options/vol_surface.py`, `credit/cds.py`, `credit/cds_index_product.py`, `credit/cds_strategies.py`, `credit/loan_cashflow.py`, `credit/loan_participation.py`, `credit/loan_portfolio.py`, `credit/tranche_pricing.py`, `desks/repo_desk.py`, ...
**Status:** ~42 callsites still hand-roll `{"type": ..., "params": ...}` instead of going through the `make_payload`/`read_payload` helpers from B.1 B2 fix.
**Why kept:** B.1 B2 only migrated `core.discount_curve`, `core.survival_curve`, `core.trade`, `core.pricing_context`. The option/credit modules were left alone to keep that slice scoped to L0; they'll migrate as each upstream module gets audited.
**Tightens when:** each module's audit pass (L1/L2). Each class gets one slice: swap `to_dict` over to `make_payload`, swap `from_dict` over to `read_payload`. Behaviour-preserving for any payload that doesn't try to write a new version.

### LD.11 — Two coexisting `MarketSnapshot` types (ARCHITECTURAL)

**Location:** `core/market_data.py` (legacy, dataclass `MarketDataSnapshot(snapshot_date, quotes: list)`) vs `pricebook.market_data._types.MarketSnapshot` (G1 P2, frozen `(id: UUID, as_of, quotes: tuple)`).
**Status:** both types exist, both are used. G1 P2 calibrators stamp `MarketSnapshot.id` onto `CalibrationResult.market_snapshot_id`. The demo `build_context(MarketDataSnapshot, PipelineConfig)` pipeline still uses the legacy types.
**Why kept:** B.3 audit deferred the decision to Gate 2. Removing legacy MarketDataSnapshot would break `build_context` and any notebooks that use it; keeping both is the path-of-least-resistance.
**Tightens when:** Gate 2 decides — either (a) migrate `build_context` to G1 P2 types and delete the legacy types, or (b) explicitly cast at the boundary with a `MarketSnapshot.from_legacy(MarketDataSnapshot)` helper. The current ambiguity is the worst option.

---

## When is the breaking full migration worthwhile?

A "breaking migration" would coordinate the following changes into a single bump (call it pricebook v1.0 or Gate 1.5):

1. Flip `strict_icma` default to `True`; remove the flag entirely (LD.1).
2. Schema-version bump to v2 across all serialisable types; remove `LD.6/LD.7/LD.8` back-compat reads.
3. Migrate the remaining ~42 custom `to_dict` overrides to `make_payload`/`read_payload` (LD.10).
4. Delete legacy `core.market_data`; route everything through `pricebook.market_data` (LD.11).
5. Pick one wire format for conventions (LD.9).
6. Remove the `_failed_imports` silent-record (LD.4) — promote to a warning.

### Trigger conditions

The migration becomes worthwhile when **any one** of these holds:

- **A. The back-compat shims are concentrated enough that the cost-to-maintain exceeds the cost-to-migrate.** Today's count: 11 entries; manageable. Re-evaluate when the count crosses ~25 — past that, the shims start interacting (a fix in one path needs to consider the back-compat behaviour of three others).
- **B. The audit chain catches a divergence that the shims hide.** E.g. a calibration result links a `MarketSnapshot.id` from G1 P2 types, but the payload's curves were built via the legacy `build_context` path — same conceptual snapshot, two unrelated objects, audit trail broken. If this *actually happens to a user*, ship the migration.
- **C. A new feature requires the post-migration shape.** E.g. a real schema-v2 design (new fields, renames, required-not-optional defaults) needs every payload to be v2. The migration becomes the prerequisite.
- **D. The library is preparing a major release/rebase** (v1.0 mark, C++ port, external publishing) where the "we still carry back-compat to v0.x payloads" cost is no longer justifiable.

### NOT triggers

- The list of shims feels long. (It's not — every entry has a clear purpose.)
- A new audit finding lands. (Audit findings get fixed in-place; they don't usually require breaking the wire format.)
- One of these shims comes up in a code review and someone wants to clean it up. (Tempting, but coordinated migration is much cheaper than per-shim removal — wait for the bundle.)

### Recommended path

1. **Now → end-of-L1-audit:** keep adding to this ledger as further shims appear during L1+ audits. Don't migrate yet.
2. **At end of full audit (L0+L1+L2):** stocktake. Decide whether to ship a "Gate 1.5 migration" before Gate 2, or fold the migration into Gate 3 (capability-complete) when the library is otherwise stable.
3. **Migration shape (when scheduled):**
   - One commit per legacy debt to characterise the contract loss (xfail-style, like A.1 B1 Slice 2 did for UST).
   - One commit per migration step (each LD removed).
   - Final commit bumps `_SERIAL_SCHEMA_VERSION = 2` on every affected class.
   - A migration helper: `migrate_payload_v1_to_v2(d: dict) -> dict` reads any pre-migration payload and rewrites it to the new shape. Keep this helper indefinitely so the old payloads don't actually become unreadable — they become *one explicit migration call* away from readable.

The key insight: a "breaking migration" doesn't have to break readability of existing data. It breaks the *implicit* back-compat (silently accept old format) and replaces it with an *explicit* migration step (`migrate_payload_v1_to_v2`). Old data still loads — but you have to ask.

---

---

### Side discovery from auto-discovery fix (2026-06-11)

When A.12 B1 was fixed, auto-discovery surfaced a NEW finding outside L0:

#### EXTRA — `fixed_income.amortising_bond` has bogus `_SERIAL_FIELDS`  *[MEDIUM]*

**Location:** `fixed_income/amortising_bond.py:337`.

```python
_serialisable("amortising_bond",
              ['face_value', 'coupon_rate', 'n_periods', 'frequency'])(AmortisingBond)
```

But `AmortisingBond.__init__` actually takes `['amortisation_type', 'coupon_rate', 'maturity_years', 'n_payments', 'notional']`. The `_register` validator emits a `UserWarning` per mismatched field at import time. Round-trip via `from_dict` would fail because `face_value`, `n_periods`, `frequency` don't exist as constructor params.

Was hidden by the old curated whitelist (`amortising_bond` wasn't in it). Fixed in a future slice when the audit reaches fixed_income.

---

## Pass B — simple composites

| # | Module | Status | Confirmed bugs | Doc/test gaps |
|---|---|---|---|---|
| B.1 | `discount_curve.py` | ⚠️ | 3 (B1 roll_down anchoring, B2 serialisation loses interpolation, B3 bumped_at no bounds check, schema_version absent) | small |
| B.2 | `survival_curve.py` | 📝 | 0 real bugs; asymmetry vs DiscountCurve (no `calibration_result`), missing `schema_version`, no roll_down (good thing!) | half the methods untested |
| B.3 | `market_data.py` (old) | ⚠️ | 0 hidden bugs; 4 documented approximations + architectural duplication with the new `pricebook.market_data` package from G1 P2 | tenor parsing untested for unusual strings |
| B.4 | `market_conventions.py` | ✅ | 0 | no dedicated test file |
| B.5 | `rate_index.py` | ✅ | 0 | BADLAR `is_overnight=False, tenor_months=None` is a minor data inconsistency |
| B.6 | `notional.py` | ✅ | 0 | 0 |
| B.7 | `forward_interpolation.py` | 📝 | 1 docstring misnomer (`_monotone_convex_forwards` is piecewise linear, not Hagan-West quadratic) | 0 |

---

## B.1 — `core/discount_curve.py`

**Purpose:** Maps dates → discount factors. Interpolates in year-fraction space (LOG_LINEAR default → piecewise-constant forwards). `df`, `zero_rate`, `forward_rate`, `instantaneous_forward`, `flat`, `bumped`, `bumped_at`, `roll_down`, plus custom `to_dict`/`from_dict`.

**Internal deps:** `core.day_count`, `core.interpolation`, `core.serialisable` (registration only). True L1 (one layer above Pass A primitives).

**Size:** 252 lines.

### Status: ⚠️ Multiple real bugs

### Confirmed bugs

#### B1 — `roll_down` produces wrong rolled curve  *[HIGH]*

**Location:** `discount_curve.py:126-146`.

When `roll_down(days)` shifts the reference date forward by `days`, the new curve should map pillar dates → `P(new_ref, pillar_date)`. By no-arbitrage: `P(new_ref, d) = P(0, d) / P(0, new_ref)`. The code does NOT divide by `P(0, new_ref)`:

```python
future_dfs = [float(self.df(d)) for d in future_dates]   # ← P(0, d), wrong
return DiscountCurve(new_ref, future_dates, future_dfs, ...)
```

**Live repro** — flat 5% curve, ref `2024-01-01`, 1-day rolldown:
```
Expected: zero_rate to 2025-01-01 = 5.0000%
Actual:                              5.0137%   ← +1.4 bp/day error
Expected: df(2025-01-01) = exp(-0.05 × 364/365) = 0.951360
Actual:                                          0.951099
```

The error scales linearly with `days`. Misstates rolldown P&L at exactly +1.4 bp per day for a typical 5% curve.

**Fix shape:** divide by `self.df(new_ref)`:
```python
disc_ref = self.df(new_ref)
future_dfs = [float(self.df(d) / disc_ref) for d in future_dates]
```

Plus the "all pillars in the past" branch (line 136-139) drops the original `day_count` and `interpolation` via `DiscountCurve.flat(...)`. That's a separate footgun on top of the main bug.

#### B2 — Custom `to_dict` drops `interpolation` and `schema_version`  *[MEDIUM]*

**Location:** `discount_curve.py:234-248`.

The custom `_dc_to_dict` emits only `{reference_date, dates, dfs, day_count}` — missing `interpolation` AND `schema_version`. **Live repro:** a curve built with `MONOTONE_CUBIC` interpolation, serialised and rebuilt, silently becomes `LOG_LINEAR` (the constructor default).

`schema_version` absence is a wider concern: G1 P3 Slice 2 added the schema-version slot to `Serialisable.to_dict` and the two decorators, but **custom `to_dict` overrides like this one bypass it**. Every model in the library that hand-writes its `to_dict` (and there are several) has the same gap.

**Fix shape:**
1. Include `interpolation` in the params dict; default to `LOG_LINEAR` on `from_dict` if absent (back-compat for pre-fix payloads).
2. Include `schema_version` at the envelope level.
3. Add a generic helper `_make_payload(type, params, version)` so all custom `to_dict`s stop forgetting it.

#### B3 — `bumped_at(pillar_idx)` has no bounds check + accepts negative indices  *[LOW]*

**Location:** `discount_curve.py:148-156`.

```python
def bumped_at(self, pillar_idx: int, shift: float) -> "DiscountCurve":
    pillar_t = [t for t in self._times if t > 0]
    pillar_df = [float(df) for t, df in zip(self._times, self._dfs) if t > 0]
    pillar_df[pillar_idx] = pillar_df[pillar_idx] * math.exp(-shift * pillar_t[pillar_idx])
```

`pillar_idx = -1` silently bumps the last pillar (Pythonic but probably surprising for a "bump pillar k" API). `pillar_idx = 1000` raises `IndexError` with no context. Add explicit validation: `if not 0 <= pillar_idx < len(pillar_t): raise IndexError(...)` with a clear message.

### Other concerns (not bugs)

- **`zero_rate(d <= ref)`** returns the t→0 instantaneous forward (lines 178-184) using only the first non-zero pillar. For a non-LOG_LINEAR curve this approximation drifts slightly from the true short-rate. Documented inline; acceptable approximation.
- **`forward_rate` docstring** at line 217 says "numerically stable form" but the formula `(df1-df2)/(tau*df2)` is the standard mathematical definition, not a special stabilised form. Cosmetic.
- **`instantaneous_forward(t)`** with very large `t` constructs a date that can overflow near `date.max` (`OverflowError`). Niche — no caller exercises this region today.
- **Silent zero-fallback** in `df`, `zero_rate`, `forward_rate` (when `df_val <= 0` or `tau <= 0`) masks pathological interpolator output. Should at minimum log a warning.

### Test coverage

`test_discount_curve.py` covers `df`, `zero_rate`, `forward_rate`, `flat`, `bumped`. Missing:
- `roll_down` — no test (would catch B1 immediately).
- `bumped_at` — no test (B3 untested).
- Round-trip through `to_dict`/`from_dict` for non-default `interpolation` (would catch B2).
- `instantaneous_forward` edge cases.

---

## B.2 — `core/survival_curve.py`

**Purpose:** Maps dates → survival probabilities `Q(t)`. Provides `survival`, `hazard_rate`, `default_prob`, `forward_hazard`, `forward_survival`, `marginal_default_density`, `pillar_hazards`, `term_structure`, `bumped` / `bumped_at`, `flat`. Custom `to_dict`/`from_dict`.

**Internal deps:** `core.day_count`, `core.interpolation`, `core.serialisable`. **Concerning:** `bumped` / `bumped_at` lazily import `credit.credit_risk._bump_survival_curve*` — `core → credit` is a layer inversion (deferred so no import-time cycle, but it shouldn't exist).

**Size:** 237 lines.

### Status: 📝 No real bugs; asymmetry + the same schema-version gap as B.1

### Correctness review

- `__init__` validates `0 < sp <= 1`. ✓
- `survival(d) = 1.0` for `d <= ref`. ✓
- `hazard_rate(d)` = `-ln(q2/q1)/(t2-t1)` for the bracket segment. ✓
- `default_prob(d1, d2) = Q(d1) - Q(d2)`. ✓
- `forward_hazard(d1, d2) = -ln(Q(d2)/Q(d1)) / τ`. ✓
- `forward_survival(d1, d2) = Q(d2)/Q(d1)`. ✓
- `marginal_default_density(d) = h(d) × Q(d)`. ✓
- `pillar_hazards()` returns piecewise-constant per-segment hazards. ✓
- **No `roll_down`** — and that's good. If one were added it would inherit the discount_curve B1 anchoring bug; this module avoiding the API entirely is the right call.
- Custom `to_dict` correctly includes `interpolation` (unlike `DiscountCurve` B2). Still missing `schema_version` (the wider G1 P3 Slice 2 gap).

### Concerns (not bugs)

- **Asymmetric with `DiscountCurve`:** no `self.calibration_result` field. G1 P2 wired the `MarketSnapshot` id into `DiscountCurve.calibration_result` but `SurvivalCurve` has no analogous storage. The hazard-bootstrap `HazardBootstrapResult.calibration_result` holds it, but a downstream consumer working from the curve alone has no provenance handle. Minor — the audit chain still reaches the snapshot via the `HazardBootstrapResult`.
- **Silent zero-fallbacks** in `hazard_rate` / `forward_hazard` when `q1<=0` or `q2<=0`. Mirrors discount_curve's pattern; same recommendation (at minimum log a `RuntimeWarning`).
- **Layer inversion:** `bumped` / `bumped_at` lazily import from `pricebook.credit.credit_risk`. The underlying functions should live in `core.survival_curve` (or a `core.survival_curve_bump` helper) since they don't depend on credit-specific logic — they just rebuild survival probs.
- **`term_structure().default_prob_1y` clamp** at line 177 caps to 0 if `d+1y > last_pillar+1y`. Ad-hoc; should either drop the clamp or document its intent.

### Test coverage

`test_survival_curve.py` has 16 tests covering `survival`, `hazard_rate`, `default_prob`, validation. Untested:
- `forward_hazard`, `forward_survival`, `marginal_default_density`, `pillar_hazards`, `bumped`, `bumped_at`, `flat`, `term_structure`.
- to_dict/from_dict round-trip (this one happens to be correct for interpolation, but no test asserts that).

Half the API surface is untested. Coverage slice would be a single targeted test file addition.

---

## B.3 — `core/market_data.py` (legacy)

**Purpose:** Older market-data layer with `Quote(quote_type, tenor, value, ...)`, `MarketDataSnapshot(snapshot_date, quotes)`, tenor parsing, and a `build_context` pipeline that assembles a `PricingContext` from a snapshot. Predates the G1 P2 work that introduced the **new** `pricebook.market_data` package.

**Internal deps:** `core.discount_curve`, `core.survival_curve`, `core.pricing_context`. **Outward:** lazily references `options.vol_surface.FlatVol`.

**Size:** 302 lines.

### Status: ⚠️ Architectural debt, no hidden bugs

The math here is intentionally approximate — it's a quick demo / sandbox pipeline, not the production calibration path. The real issues are structural.

### Confirmed concerns

#### C1 — Architectural duplication with the new `pricebook.market_data` (G1 P2 Slice 1)  *[ARCH]*

Two parallel snapshot types now exist:

| Aspect | `core.market_data.MarketDataSnapshot` (this file) | `pricebook.market_data.MarketSnapshot` (G1 P2) |
|---|---|---|
| Identity | None — just `snapshot_date` | UUID `id` + `as_of: datetime` |
| Mutability | mutable; `add()` method | frozen `dataclass(frozen=True)` |
| Quote shape | `(quote_type: QuoteType, tenor, value, currency, name)` | `(id: QuoteId, value, bid_ask_bp)` with a separate `QuoteId(kind, tenor, currency, label)` |
| Wire format | `to_dict` / `from_dict` (plain dict, no envelope, no version) | not yet serialised (the type is the audit primitive; downstream calibrators carry the id) |
| Audit-chain integration | none | every G1 P2 calibrator accepts `market_snapshot=` and stamps `MarketSnapshot.id` onto `CalibrationResult.market_snapshot_id` |

The legacy types are still imported by `build_context` and a handful of older notebooks / tests. Net effect: two coexisting "market data" abstractions, with calibrators routed through the new one and the demo `build_context` pipeline routed through the old one. Bridging or sunsetting is an open architectural decision (Gate 2 territory).

**Decision needed:** retire `core.market_data` (and migrate `build_context` to the new types), OR keep both and explicitly cast one to the other at the boundary. Either is fine but the current state — silent coexistence — is the worst option.

#### C2 — `_build_discount_curve` treats every rate as continuously-compounded zero  *[APPROXIMATION, documented]*

**Location:** `market_data.py:187`.

```python
dfs = [math.exp(-q.value * tenor_to_years(q.tenor)) for q in sorted_quotes]
```

Code comment acknowledges: *"Simple approach: treat all as continuously compounded zero rates"*. DEPOSIT_RATE quotes are *simply compounded* in market convention; SWAP_RATE quotes need bootstrap. Anyone calling `build_context` gets curves that disagree with the proper `curves.bootstrap` path by ~ rate²·T/2 ≈ 1.3 bp at 5y, 5% rate. Acceptable for a sandbox pipeline; not for production.

#### C3 — `_build_survival_curve` uses the "credit triangle" approximation  *[APPROXIMATION, documented]*

`hazard ≈ spread / (1 - R)`. Standard quick approximation. Bypasses the proper protection/premium-leg bootstrap. Documented in the docstring.

#### C4 — `tenor_to_years` uses calendar days for sub-month tenors  *[APPROXIMATION]*

`1D=1/365`, `1W=7/365`. Currency-agnostic; for BRL (BUS/252) or any business-day convention this is wrong by ~30%. The function is used only by `build_context` so the blast radius is contained to the demo pipeline.

#### C5 — No `schema_version` on any `to_dict`  *[CONSISTENT WITH B.1 B2]*

Same gap as `DiscountCurve.to_dict` — custom `to_dict` overrides skip the schema-version slot from G1 P3 Slice 2. Fix together with all other custom `to_dict`s in a single generic-helper slice.

### Test coverage

- `test_market_data.py` covers the basic API.
- `tenor_to_years` not tested for unusual strings (`"1.5Y"`, malformed input, etc.).
- `_build_discount_curve` not tested against the proper bootstrap to characterise the approximation gap.

### Slicing items (defer)

- **Retire decision:** Gate 2 — is this module kept (and the new G1 P2 types coexist), or migrated/removed? Either way, document the boundary so callers know which to use.
- **Bridge:** `MarketSnapshot.from_legacy(MarketDataSnapshot)` helper if both coexist long-term.
- **Test gap:** parametrised tenor parsing tests.

---

## B.4 — `core/market_conventions.py`

**Purpose:** Pure-data registry: `EquityIndexSpec`, `CommodityContractSpec`, `LinkerConvention` frozen dataclasses + in-code dicts of canonical instances (SPX, NDX, SX5E, ...; CL, BRN, NG, ...; US TIPS, UK ILG, ...) + lookup functions + one math helper (`index_ratio` for daily-linear CPI interpolation).

**Internal deps:** `core.serialisable`, `core.data_registry` (for the JSON-merge-then-load pattern).

**Size:** 211 lines.

### Status: ✅ Clean

- All three frozen dataclasses decorated with `@serialisable_convention`. Registration verified by the post-fix A.12 auto-discovery (B.1 of this audit prompted me to add a regression test that `EquityIndexSpec` is reachable).
- Hand-curated registries match market reality: TIPS lag = 3 months ✓, UK RPI lag = 8 (old style) ✓, FR/IT/DE HICP lag = 3 ✓; CL/BRN/NG/GC/SI tick values are consistent with the contract sizes (`contract_size × tick_size = tick_value` ✓).
- `index_ratio` daily-linear formula `CPI_ref(d) = CPI(m-1) + (d-1)/D × (CPI(m) − CPI(m-1))` matches the TIPS / UK ILG market convention.
- `_load_reg` overlay pattern (in-code defaults, JSON-file extension) is clean.

### Test coverage

No dedicated test file. The serialisation round-trip is covered by `test_serialization_autodiscovery.py::test_registry_picks_up_module_not_in_old_whitelist` (added in v0.891 — that's how `EquityIndexSpec` is now exercised). Math (`index_ratio`) untested.

### Slicing items (defer)

- Add `test_market_conventions.py` covering `get_equity_index`, `get_commodity_contract`, `get_linker_convention`, and `index_ratio` (especially the day-1 → day-D boundary behaviour). Low priority.

---

## B.5 — `core/rate_index.py`

**Purpose:** Pure-data registry of 26 rate indices: G10 RFRs (SOFR, ESTR, SONIA, TONA, SARON, CORRA, AONIA, NZOCR) + IBORs (EURIBOR 3M/6M, TIBOR 3M) + EM (CDI, KOFR, SORA, HONIA, THOR, TIIE, SHIBOR, DR007, WIBOR, PRIBOR, BUBOR, JIBAR, IBR, TPM, TIPM, BADLAR). Each `RateIndex` carries day count, fixing lag, compounding method, observation shift, payment delay, tenor, and administrator.

**Internal deps:** `core.day_count`, `core.serialisable`, `core.data_registry`.

**Size:** 329 lines.

### Status: ✅ Clean

- All G10 RFR conventions match official sources (FRBNY for SOFR, ECB for ESTR, BOE for SONIA, BOJ for TONA, etc.).
- Day-count conventions correct per market: SOFR/ESTR/SARON = ACT/360; SONIA/TONA/CORRA = ACT/365F; CDI = BUS/252 (per B3 Anbima).
- Compounding methods correct: RFRs = COMPOUNDED, IBORs = FLAT, DR007 = AVERAGED.
- JSON-overlay pattern allows per-environment additions without code changes.

### One data quirk (not a bug)

- **BADLAR** has `is_overnight=False` AND `tenor_months=None`. BADLAR is a 30-35-day rate (large-deposit rate for ARS) so it's not truly overnight, but `tenor_months=None` together with `is_overnight=False` makes it neither category. Should probably be `tenor_months=1` (it's quoted on the ~1-month bucket). Single-line data fix when convenient.

### Test coverage

No dedicated test file. The `@serialisable_convention` round-trip is tested generically. The registry-lookup functions are not directly tested. Low priority.

### Slicing items (defer)

- BADLAR data fix: `tenor_months=1`. One-line + smoke test.
- Add a basic `test_rate_index.py` with: every G10 currency has at least one indexed rate, every overnight index has `compounding=COMPOUNDED`, every IBOR has `compounding=FLAT`. Smoke-shaped tests.

---

## B.6 — `core/notional.py`

**Purpose:** Single helper `normalize_notional(notional, n_periods)`: scalar → replicated list; list → extended/truncated to exactly `n_periods`. Validates positivity.

**Internal deps:** None.

**Size:** 43 lines.

### Status: ✅ Clean

Standalone helper. Validates `notional > 0` (scalar AND elements of a list). When a list is shorter than `n_periods`, extends by repeating the last element — sensible for amortisation tail. Test coverage in `test_notional_schedule.py` is comprehensive (identity for uniform list, factory equivalence for amortising/accreting/rollercoaster patterns, variable-notional consistency).

### Slicing items: none.

---

## B.7 — `core/forward_interpolation.py`

**Purpose:** Build `DiscountCurve` instances by interpolating instantaneous forward rates rather than DFs / zero rates. Three methods: piecewise constant, piecewise linear, *monotone convex* (Hagan-West 2006 inspired).

**Internal deps:** `core.discount_curve`, `core.day_count`.

**Size:** 263 lines.

### Status: 📝 One docstring misnomer, code works

### Findings

#### B1 — `_monotone_convex_forwards` is named for Hagan-West but interpolates linearly  *[DOC]*

**Location:** `forward_interpolation.py:190-252`.

The function implements Step 1 (discrete forwards between pillars) and Step 2 (pillar-value smoothing with monotonicity constraints) of Hagan-West (2006) Algorithm 1. Step 3 — the *quadratic* monotone-convex interpolation between adjusted pillars — is REPLACED by plain linear interpolation:

```python
# Step 3: interpolate using monotone Hermite spline
def f(t):
    ...
    return f0 * (1 - x) + f1 * x  # linear for now — preserves positivity
```

The code comment confesses it (`"linear for now"`). The result still preserves positivity (because adjusted pillar values are clamped above zero) and is C⁰, but it does NOT preserve the C¹ smoothness or the convex shape from the full Hagan-West algorithm.

Practical impact: forward curves built with `MONOTONE_CONVEX` are *better* than plain linear-on-forwards (because Step 2 prevents oscillation) but worse than the published Hagan-West method (forward curve has kinks at pillars instead of being smooth). Not a bug — just shorter of the claim than the name implies.

**Fix shape:**
- Option A: rename the method to `PIECEWISE_LINEAR_WITH_HW_PILLARS` and document explicitly that it's NOT the full HW spline.
- Option B: implement Step 3 properly (quadratic spline per segment with the monotonicity constraint).

Option B is the right long-term call; Option A is a 2-line doc-only fix for now.

### Correctness review (other paths)

- `_piecewise_constant_forwards`: standard. ✓
- `_piecewise_linear_forwards`: places the forward at the period END, then linear-interpolates. Convention; standard. ✓
- `build_forward_curve`: builds a dense weekly grid (52 pts/year, min 50). Integrates forward via trapezoidal rule (n_steps=100). Reasonable for smooth forwards.
- `_integrate_forward`: trapezoidal rule. Adequate.

### Test coverage

`test_forward_interpolation.py` covers all three methods + `monotone_convex_forwards` standalone + `extract_forwards`. Decent.

### Slicing items (defer)

- B1 fix: rename or implement Step 3 properly. Defer to a future curves-quality slice.

---

## Pass C — portfolio constructs

| # | Module | Status | Confirmed bugs | Doc/test gaps |
|---|---|---|---|---|
| C.1 | `trade.py` | 📝 | 1 trivial (dead-code orphan `to_dict`) | direction/notional_scale not validated |
| C.2 | `book.py` | 📝 | 0 (3 `vars(self)` to_dicts; silent 0-notional for instruments lacking `notional` attr) | DV01 timing-by-pillar untested |
| C.3 | `instrument_result.py` | ✅ | 0 (Protocol only) | n/a |
| C.4 | `results.py` | ✅ | 0 (re-exports + 2 dataclasses) | n/a |
| C.5 | `mandate.py` | 📝 | 0 (3 `vars(self)` to_dicts) | rating ladder untested at boundaries |
| C.6 | `daily_pnl.py` | 📝 | 0; sequential bump attribution is order-dependent (standard) | 3 `vars(self)` to_dicts |
| C.7 | `settlement.py` | ⚠️ | **1 — settlement lag uses calendar days not business days** | per-product settlement timing untested |
| C.8 | `greeks.py` | ⚠️ | **1 — `dollar_gamma` formula wrong vs docstring (factor of S² × 1e-4 missing)** | `dollar_delta` formula questionable too |

---

## C.1 — `core/trade.py`

**Purpose:** `Trade(instrument, direction, notional_scale, trade_date, counterparty, trade_id)` + `Portfolio(trades, name)` aggregator.

**Internal deps:** `core.pricing_context`, `core.serialisable`.

**Size:** 137 lines.

### Status: 📝 1 dead code line + 2 robustness concerns

### Findings

#### B1 — Orphan `to_dict` returns mutable `vars(self)`  *[LOW, dead code]*

**Location:** `trade.py:56-57`.

```python
def to_dict(self) -> dict:
    return vars(self)
```

This sits inside the `Trade` dataclass body but is overwritten at line 116 by `Trade.to_dict = _trade_to_dict`. **Dead code** in normal operation. If anyone ever imports the module without triggering the bottom-of-file rebinding (impossible today; defensive against future refactors that split the file), they'd hit the same shared-`__dict__` mutation bug pattern as A.5 / A.7. Easy fix: delete the orphan or make it `return dict(vars(self))`.

### Robustness concerns (not bugs)

- **`Trade.pv` doesn't validate `direction in (+1, -1)`.** `direction=2.0` silently doubles the PV; `direction=0` zeros it. No-op or even useful in some contexts (e.g. fractional positions) but undocumented.
- **`Trade.pv` doesn't validate `notional_scale > 0`.** Negative scale silently flips direction; zero zeros PV. Again, no actual breakage but undocumented.
- **`notional_scale` docstring says "multiplier on the instrument's notional"** — but the code multiplies the *PV* directly. Equivalent for PV-linear-in-notional instruments (the common case) but differs for nonlinear payoffs. Cosmetic.
- **`Portfolio.add` mutates** the trade list — fine for an aggregator, but pricing the same `Portfolio` twice while another caller is `add`-ing is non-thread-safe.

### Test coverage

`test_trade.py` covers the happy paths (long/short PV, cancel, notional_scale, metadata, missing pv_ctx, portfolio aggregate). The `direction` / `notional_scale` value-validation isn't tested because it isn't validated; that's the gap.

### Slicing items (defer)

- Delete the orphan `to_dict` at trade.py:56-57.
- Validate `direction in (-1, +1)` and `notional_scale > 0` in `Trade.__post_init__` (with `raise ValueError`). Test both branches.

---

## C.2 — `core/book.py`

**Purpose:** `Book` (named trade container), `Desk` (collection of books), `Position` / `LimitBreach` / `BookLimits` dataclasses + `tenor_bucket(ref, end)` + `check_limits`. Tenor-bucket DV01 risk decomposition.

**Internal deps:** `core.pricing_context`, `core.trade`. 337 lines.

### Status: 📝 Clean math + 1 silent-zero pattern

- `tenor_bucket` correctly bucketed; boundary semantics match the docstring.
- `Book.pv` / `Book.dv01` / `Book.tenor_dv01` — all standard parallel-bump.
- **Silent zero:** `_instrument_notional` returns 0.0 if the instrument lacks a `notional` attribute. `positions()` then aggregates zero notional silently — that exposure drops out of risk. Should log a warning or raise.
- 3 `to_dict` methods (`Position`, `LimitBreach`, `BookLimits`) return `vars(self)` — same mutation footgun as A.5 / A.7. Fix with the generic `to_dict` cleanup slice.

---

## C.3 — `core/instrument_result.py`

31 lines. `InstrumentResult` Protocol only (price + to_dict contract). No implementation. ✅ Clean.

---

## C.4 — `core/results.py`

60 lines. Re-exports `SolverResult` (eager), `QuadratureResult`/`MCResult`/`OptimizerResult`/`ODEResult` (TYPE_CHECKING). Defines `TreeResult` and `PDEResult` dataclasses. ✅ Clean — no `to_dict` issues here.

---

## C.5 — `core/mandate.py`

**Purpose:** Mandate compliance engine — `PortfolioHolding`, `MandateCheckResult`, `MandateReport`, `Mandate` dataclasses + `check_mandate` driver. Rating ladder via `RATING_ORDER` dict mapping `AAA..D, NR` to integer ranks; `rating_at_least` for comparisons. Pre-built factories: `investment_grade_mandate`, `sovereign_only_mandate`, `balanced_mandate`, `high_yield_mandate`.

279 lines.

### Status: 📝 No bugs

- `RATING_ORDER` ladder is the standard S&P long-term scale (AAA → D), with `NR` mapped to a high (worse) value so unrated entities always fail "at least BBB-" checks. ✓
- `rating_at_least` uses `<=` on the rank — correct (lower rank = better). ✓
- `MandateReport.to_dict` writes a flat representation manually rather than `vars(self)` — clean.
- Other to_dicts use `vars(self)` — same generic pattern.

---

## C.6 — `core/daily_pnl.py`

**Purpose:** Daily official P&L (`compute_daily_pnl`) + bump-and-reprice attribution (`attribute_pnl`).

231 lines. The P&L decomposition (market-move + new-trades + amendments) is correct. Attribution uses **sequential** bumps (rates → vol → theta → unexplained), which is the standard *but* order-dependent approach. Reordering the bumps gives different per-factor splits — by-design but worth documenting prominently.

- 3 `to_dict` methods (`DailyPnL`, `TradeAttribution`, `BookAttribution`) return `vars(self)`. Same pattern.
- No actual bugs.

---

## C.7 — `core/settlement.py`

**Purpose:** `SettlementType` enum, per-product `SETTLEMENT_CONVENTIONS` dict, per-asset settlement-result dataclasses (`CashSettlementResult`, `CDSSettlementResult`, `OptionSettlementResult`, `FuturesSettlementResult`, `SettlementRiskResult`), plus `add_business_days` and `fx_spot_date`.

364 lines.

### Status: ⚠️ Calendar-vs-business-day inconsistency

#### B1 — Settlement lag uses calendar days  *[MEDIUM]*

**Location:** `settlement.py:cash_settlement` (line 75) and similar paths.

```python
settle = date.fromordinal(exercise_date.toordinal() + lag_days)
```

The standard `lag_days = 2` (FX spot) or `lag_days = 1` (US equity option) refers to **business days**, not calendar days. The module HAS `add_business_days` defined right there (lines 230+), but `cash_settlement`/`cds_settlement_physical`/etc. don't use it.

**Impact:** when a Friday trade settles T+2 calendar, the code returns Sunday — silently shifted to a non-business day. Bond settlement, FX spot date, option physical-settlement timing all wrong when the lag spans a weekend or holiday.

**Fix shape:** route every `+ lag_days` through `add_business_days(d, lag_days, calendar)` and require a calendar at the entry point.

- Multiple `to_dict` methods return `vars(self)`. Same pattern.

---

## C.8 — `core/greeks.py`

**Purpose:** `Greeks` dataclass with standard sensitivities + 2 convenience formulas.

37 lines.

### Status: ⚠️ One real formula bug

#### B1 — `dollar_gamma` formula doesn't match docstring  *[LOW-MED]*

**Location:** `greeks.py:32-34`.

```python
@property
def dollar_gamma(self) -> float:
    """Gamma P&L for a 1% spot move: 0.5 × gamma × S² × 0.01²."""
    return 0.5 * self.gamma
```

The docstring claims the formula is `0.5 × gamma × S² × 0.01²`. The code does `0.5 × gamma`. Off by a factor of `S² × 10⁻⁴` — e.g. for S=200, that's 4×. For S=100 the factor happens to be 1 (since 100² × 0.0001 = 1), making the bug invisible at S=100 specifically.

**Why it persists:** for ATM options where S≈100 (test fixtures, demos), the formula is accidentally correct. Any real-spot scenario with S != 100 (FX, S&P at 5000, oil at 80) gives wrong gamma-P&L.

**Compounding issue:** `Greeks` doesn't carry a `spot` field. So even if we wanted to fix the formula, we'd need to thread spot in or change the contract. Cleanest fix: rename to `gamma_pnl_pct_factor` and document explicitly that it must be multiplied by `S² × 1e-4` externally.

#### B2 — `dollar_delta` formula questionable  *[DOC]*

`delta * self.price` is at best an approximation of dollar delta. Real dollar-delta requires either `delta * spot` (per-unit-spot-move PnL) or `delta * notional` (book-level position PnL). The product `delta * price` has the right *order of magnitude* for ATM vanilla options but doesn't have a clean financial meaning.

- `to_dict` returns `vars(self)` — same pattern.

---

## Pass D — engine plumbing

| # | Module | LoC | Status | Confirmed bugs |
|---|---|---:|---|---|
| D.1 | `pricing_context.py` | 282 | ⚠️ | **3 confirmed from prior audit, now verified** |
| D.2 | `data_registry.py` | 150 | ✅ | 0 |
| D.3 | `dependency_graph.py` | 191 | ✅ | 0 (analysis tooling) |
| D.4 | `desk_protocol.py` | 51 | ✅ | 0 (docstring-only contract) |
| D.5 | `numerical_method_map.py` | 230 | ✅ | 0 (recommendation table + helpers) |
| D.6 | `numerical_safety.py` | 289 | ✅ | 0 (CFL / Feller / martingale checks) |
| D.7 | `convergence_framework.py` | 181 | ✅ | 0 (convergence-study runners) |

### D.1 — `core/pricing_context.py`  *(re-audited after G1 P3 + B.1 B2 changes)*

**Purpose:** central pricing-data bundle. Already touched in G1 P3 Slice 1 (`numerical_config`) and B.1 B2 fix (custom `to_dict` migration). Three pre-existing bugs from `MODULE_HEALTH.md` are still active.

#### B1 — Empty-dict fields become `None` on round-trip  *[MEDIUM]*

**Location:** `pricing_context.py:266, 271, 278`.

```python
proj = {n: _fd(c) for n, c in p.get("projection_curves", {}).items()} or None
credit = {n: _fd(c) for n, c in p.get("credit_curves", {}).items()} or None
... vol_surfaces=vols or None, ... fx_spots=fx or None
```

When the dict is empty (no entries or absent from payload), the `or None` collapses to `None`. The dataclass declares `projection_curves: dict[str, DiscountCurve] = field(default_factory=dict)`. Constructing with `None` accepts it (Python doesn't enforce dataclass field types at runtime), so subsequent `ctx.projection_curves["foo"]` raises `TypeError: 'NoneType' object is not subscriptable` instead of the documented `KeyError`.

**Fix:** drop the `or None` — pass empty dicts through.

#### B2 — Several fields silently dropped on round-trip  *[MEDIUM, multi-currency users]*

`_ctx_to_dict` emits `discount_curve, projection_curves, vol_surfaces, credit_curves, fx_spots`. It **silently drops**:

- `discount_curves` (the per-currency dict introduced for multi-currency support)
- `inflation_curves`
- `repo_curves`
- `reporting_currency`
- `stochastic_credit_models`
- `credit_vol_surfaces`
- `credit_correlations`
- `numerical_config` (just added in G1 P3 Slice 1!)

Multi-currency contexts and contexts with stochastic-credit or numerical-config attached lose those fields on every serialisation round-trip.

**Fix:** extend `_ctx_to_dict` / `_ctx_from_dict` to emit and read all dataclass-declared fields. This is non-trivial because each field needs the right serialiser (per-currency dicts need to recurse into curve `to_dict`, `numerical_config` needs its own `to_dict` added, etc.).

#### B3 — `replace()` shares mutable dicts with the parent  *[LOW-MED]*

**Location:** `pricing_context.py:159-176`.

`replace()` passes the original `discount_curves`, `vol_surfaces`, etc. dicts by reference to the new context. The docstring says *"Immutable snapshot of market data"* but `ctx2 = ctx.replace(reporting_currency="EUR")` then `ctx2.discount_curves["USD"] = new_curve` mutates `ctx` too.

**Fix:** `dict(...)` each container in `replace()` to defensively copy. Tiny cost for an actual immutable-snapshot contract.

### D.2-D.7

All clean. Short notes:

- **`data_registry.py`** — defensive path validation (`_validate_filename` rejects `..` and absolute paths), per-row try/except in `load_conventions` with `RuntimeWarning` on failures (not silent), JSON-array-only enforcement. Good shape.
- **`dependency_graph.py`** — analysis tool for the layer graph. Pure traversal logic, no pricing maths.
- **`desk_protocol.py`** — pure documentation file (no executable code beyond the docstring).
- **`numerical_method_map.py`** — recommendation table mapping product features → numerical method. Lookup + comparison helpers; no maths.
- **`numerical_safety.py`** — CFL stability check, Feller condition test, martingale test, convergence-rate estimation from log-error regression. Standard implementations; correctness depends on the inputs the caller passes.
- **`convergence_framework.py`** — runs strong/weak convergence studies with step-halving, fits log-error slopes. Standard.

### Pass D summary

3 confirmed bugs in `pricing_context.py` (all carried over from prior MODULE_HEALTH audit; verified still present). The remaining 6 modules are clean — they're either analysis tooling or framework code with thin layers above standard numerical primitives.

---

## Pass C — summary

8 modules audited. Total **3 confirmed bugs** + a recurring `vars(self)` to_dict mutation pattern.

| Severity | Count | Headline |
|---|---:|---|
| MED | 1 | C.7 B1 — settlement lag uses calendar days, not business days |
| LOW | 2 | C.1 B1 — dead-code orphan to_dict; C.8 B1 — `dollar_gamma` formula doesn't match docstring (off by S²×1e-4) |

**Cross-cutting:** ~15 `to_dict` methods across `book.py`, `daily_pnl.py`, `settlement.py`, `greeks.py`, `mandate.py`, `trade.py` (orphan), `solvers.py`, `approximation.py` return `vars(self)` directly — the same mutation footgun. A single "generic to_dict-cleanup" slice would fix all of them at once.

| Module | LoC | Status |
|---|---:|---|
| C.1 trade | 137 | 📝 |
| C.2 book | 337 | 📝 |
| C.3 instrument_result | 31 | ✅ |
| C.4 results | 60 | ✅ |
| C.5 mandate | 279 | 📝 |
| C.6 daily_pnl | 231 | 📝 |
| C.7 settlement | 364 | ⚠️ |
| C.8 greeks | 37 | ⚠️ |

---

## Pass B — summary

7 modules audited. Total: **4 confirmed bugs** + 1 architectural duplication + 2 minor data quirks.

| Severity | Count | Headline examples |
|---|---:|---|
| HIGH | 1 | B.1 B1 — `DiscountCurve.roll_down` forgets to divide by `P(0, new_ref)`; +1.4 bp/day error on a 5% curve. |
| MEDIUM | 1 | B.1 B2 — custom `to_dict` overrides drop `interpolation` and `schema_version`; wider pattern across the library. |
| LOW | 2 | B.1 B3 — `bumped_at` no bounds check; B.7 B1 — Hagan-West "monotone convex" actually piecewise linear. |
| ARCH | 1 | B.3 C1 — legacy `core.market_data` vs new `pricebook.market_data` (G1 P2). Decision deferred to Gate 2. |
| DATA | 2 | B.5 BADLAR `is_overnight=False, tenor_months=None`; layer inversion in `SurvivalCurve.bumped` → credit. |

**Cross-cutting finding:** the `schema_version` work from G1 P3 Slice 2 doesn't reach **any** custom `to_dict` override. Every model with a hand-written `to_dict` (`DiscountCurve`, `SurvivalCurve`, legacy `MarketDataSnapshot`, `PricingContext`, several option types) is invisible to the schema-version contract. A single "wrapper helper" slice would close this gap once for the whole library.

Layers complete: L0 Pass A (foundations) and L0 Pass B (simple composites). 13 + 7 = 20 modules audited; **24 confirmed bugs** + 1 architectural duplication catalogued.

---

## Pass A — summary

13 modules audited. Total: **20 confirmed bugs** (mostly Low/Medium) + significant test gaps in calendar / interpolation / FX-forward / Serialisable-edge-shapes.

| Severity | Count | Headline examples |
|---|---:|---|
| HIGH | 4 | A.1 B1 ICMA silent fallback (UST mispricing); A.2 B1 wrong Sat-substitute for UK/AU/NZ/CA; A.4 B1 EOM anchored to end in front-stub; A.12 B1 curated import whitelist |
| MEDIUM | 4 | A.2 B2 Tokyo no substitute; A.11 B1 list[T] dispatch; A.11 B2 Union[A,B,None] dispatch; (others) |
| LOW | 12 | to_dict mutation × 4, schedule heuristics, calendar n=5/n=0, etc. |

Now move to **Pass B** — simple composites that depend on Pass A's primitives: `discount_curve`, `survival_curve`, `market_data` (old), `market_conventions`, `rate_index`, `notional`, `forward_interpolation`.

---

## Aggregate slicing queue (will work after audit pass)

From A.1 (`day_count`):
1. Test gap fill: ACT/ACT ICMA + ACT/ACT ISDA + 30E/360 + 30/360-EOF + `date_from_year_fraction`.
2. B2 fix: validate `frequency > 0` in ICMA (single slice).
3. D1: rename `_thirty_360` docstring "ISDA 2006" → "30/360 US (Bond Basis)".
4. D2: clarify `date_from_year_fraction` calendar-year semantics.
5. B1 (multi-slice): introduce `strict_icma` flag; per-caller migration; flip default. Sized as its own mini-roadmap inside Gate 2.

From A.2 (`calendar`):
6. Calendar test scaffolding: parametrised holiday-truth tests for every cal class (one slice — pure tests, no source changes; will start failing for B1/B2 calendars, which is the point).
7. B3 fix: validate `1 <= n <= 5` in `_nth_weekday`; raise on out-of-range.
8. B5 fix: stop mutating components' `_holiday_cache` from `JointCalendar`.
9. B4 fix: widen `is_holiday` window to `(Y-1, Y, Y+1)`.
10. B1 fix (multi-slice, one per calendar): override `_observe` for London → AUD → NZD → CAD with UK 1971-style "next available working day" rule. One slice per calendar so failing-test surface is clean.
11. B2 fix: implement *furikae kyūjitsu* on `TokyoCalendar`.

From A.3 (`currency`):
12. Tests for `settlement_lag` (USD/CAD T+1; others T+2), `forward_rate`/`forward_rate_from_curves`/`forward_points`, `is_ndf`, `all_g10_pairs` (count = 45, all market convention). Single slice.

From A.4 (`schedule`):
13. B1 fix: anchor EOM on `start` regardless of generation direction. Add regression test with `start=2024-01-31, end=2025-04-15, semi-annual` to catch the front-stub case.
14. Test gap fill: WEEKLY, LONG_BACK, post-adjust dedup/ordering. Single slice.

From A.5 (`solvers`):
15. B1 fix: `SolverResult.to_dict` returns a copy (one-line fix + test).
16. B2 fix: honour `maxiter` in ITP and report actual iteration count.
17. Test gap fill: NaN/Inf, near-zero-derivative, maxiter-exhausted return paths. Single slice.

From A.6 (`interpolation`):
18. Test gap fill: `TestAkima` class (knot recovery, smoothness, ghost-boundary 2-point), `LogLinear` right-extrap slope continuation, Hyman α²+β²>9 clipping path. Single slice.

From A.7 (`approximation`):
19. Generic `to_dict` mutation fix — patches A.5 `SolverResult` + A.7 `ChebyshevInterpolant`/`PadeApproximant`/`RichardsonTable` in one slice (`return dict(vars(self))`). Add a regression test that mutates the dict.

From A.10 (`fixings`):
20. Clarify `get_with_lag` semantics — either require `calendar` (raise on None) or rename / split. Single slice.

From A.11 (`serialisable`):
21. B1: dispatch `list[SomeSerialisable]` via recursive `from_dict`. Round-trip test.
22. B2: handle `Union[A, B, None]` (3+ args) by dispatching the dict's `"type"` via registry.
23. B4: narrow numpy `.item()` duck-test to `isinstance(v, np.generic)`.
24. B5: graceful `CurrencyPair` parse with split-limit + length-check + clear error. (Bundle with A.12 B2.)
25. B6: structured `ValueError` (not bare `KeyError`) when `params` is missing from payload.
26. B7: int-valued Enum from string fallback.
27. B3: registry re-register emits `DeprecationWarning` and overwrites. (Dev-workflow ergonomics only — defer to lowest priority.)
28. Doc the v<expected migration contract before bumping any class to v2.

From A.12 (`serialization`):
29. B1: replace curated `_ensure_loaded` list with `pkgutil.walk_packages` auto-discovery. Add a test that imports the package, calls `registered_types`, asserts every `@serialisable*` site has a registered key. (Bigger slice — gates Pass B audits where some Pass-B classes use @serialisable and might be missing from the curated list.)

(More entries will arrive as the audit walks through Pass A.)
