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
| A.2 | `calendar.py` | ❓ | | |
| A.3 | `currency.py` | ❓ | | |
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

## Aggregate slicing queue (will work after audit pass)

From A.1:
1. Test gap fill: ACT/ACT ICMA + ACT/ACT ISDA + 30E/360 + 30/360-EOF + `date_from_year_fraction`.
2. B2 fix: validate `frequency > 0` in ICMA (single slice).
3. D1: rename `_thirty_360` docstring "ISDA 2006" → "30/360 US (Bond Basis)".
4. D2: clarify `date_from_year_fraction` calendar-year semantics.
5. B1 (multi-slice): introduce `strict_icma` flag; per-caller migration; flip default. Sized as its own mini-roadmap inside Gate 2.

(More entries will arrive as the audit walks through Pass A.)
