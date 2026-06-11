# L0 Foundations — Layer Audit

**Started:** 2026-06-11 (Gate 2 prep)
**Scope:** `pricebook.core.*` — 36 modules organised into four passes by dependency depth.
**Method:** Fresh read per module; cross-reference existing `MODULE_HEALTH.md` findings; document status, real bugs, doc gaps, test gaps, slicing proposals.

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
| A.4 | `schedule.py` | ❓ | | |
| A.5 | `solvers.py` | ❓ | | |
| A.6 | `interpolation.py` | ❓ | | |
| A.7 | `approximation.py` | ❓ | | |
| A.8 | `caching.py` | ❓ | | |
| A.9 | `protocols.py` | ❓ | | |
| A.10 | `fixings.py` | ❓ | | |
| A.11 | `serialisable.py` | ❓ | | |
| A.12 | `serialization.py` | ❓ | | |
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

(More entries will arrive as the audit walks through Pass A.)
