# Report — foundation audit (fix/foundation-audit)

**Branch:** `fix/foundation-audit` (off main). **Version:** `0.74.2` (PATCH). **136 L0 tests,
`verify.py all` green.** Deep-read verification of `calendar · market_calendars · rate_index ·
schedule · solvers`, plus housekeeping and design-doc landing.

---

## Part 0 — working tree cleaned
- **(a) style:** the uncommitted diff across all 16 foundation modules was **100% ruff-format reflow**
  — proven format-invariant (formatted-base ≡ formatted-dirty, byte-identical; 0 substantive changed
  lines; 38 calendar registrations unchanged). Landed as one `style:` commit, ruff-format-stable.
- **(b) substantive:** **none.** The flagged `market_calendars +929 · rate_index +109 · day_count +52 ·
  schedule +44 · solvers +39` are entirely one-arg-per-line reflow (the committed code was hand-compact;
  format-on-save exploded it). No real work was sitting uncommitted — nothing to commit blind.

## Part 1 — design docs landed
`docs(design):` — `17_quarry_L0_classification` · `18_topic1_yield_curve` · `19_market_data_design` ·
`20_foundation_contracts` + README forward scope (F1 market-data ✅, F2 model/calibrator next, T1
multicurve ✅). **`.claude/` ruling:** personal + transient state (`settings.local.json`, `worktrees/`)
is **gitignored** (made explicit in the repo `.gitignore`, not relying on a contributor's global
ignore); **shared** project config (`.claude/settings.json`, `commands/`, `agents/`, `skills/`) stays
**trackable** for the team. Currently only the personal file exists, so nothing is tracked — correct.

## Part 2 — verification pass (findings → red → green, severity order)

| # | finding | severity | fix | oracle |
|---|---|---|---|---|
| **F2** | `accrued_rate` obs-shift: numerator over shifted window, denominator over interest period → silently wrong rate | **silent-wrongness** | normalise by the numerator window's own days (`total/basis`) | Jun 21–28 2024 (shift crosses Juneteenth), flat 5% → 0.05 not 0.0572 |
| **F1** | degenerate window: COMPOUNDED returns `0.0`, AVERAGED/EXPONENTIAL `ZeroDivisionError` | **silent-wrongness + crash** | guard → `ValueError` (S14) | accrual [Sat, Mon) with no business day raises for all 3 methods |
| **F3** | `add_business_days(d, 0)` returns non-business `d` | **wrong-answer (latent)** | `n==0` requires a business day, else raise; snap via `adjust()` | `add_business_days(Sat, 0)` raises; on a business day returns it |

**F2 is a no-op for the lookback/plain path** (`total/basis` ≡ the old `year_fraction` there) — the SOFR
1e-12 compounded oracle passes unchanged; only the previously-wrong observation-shift path moves.

### Systematic checks — clean bill (published references)
Gregorian + Orthodox Easter 2020–2035 · US 5 U.S.C. 6103 (Sat→Fri, Sun→Mon) · ANZAC no-shift vs
Christmas shift (Sydney) · Tel Aviv Fri-Sat weekend · year-boundary spill (New Year Sat→Fri prior
Dec 31) · `since=2021`/`until=2023` boundaries · unmapped currency raises · half-days are business days ·
weekend 3-day accrual · IMM 2024 (Mar 20/Jun 19/Sep 18/Dec 18 vs CME) · CDS unadjusted-20th · EOM
anchored once from start · obs_shift ≠ lookback · lockout freezes the cutoff rate · payment_delay
independence · solvers pass `tol`/`maxiter` to scipy and raise on non-bracket/non-convergence.

---

## Review inputs (redesign/11)
1. **Oracle quality:** every fix is proven by a red test against a published RFR/ISDA convention; no
   quarry-cross-check reliance. Systematic checks anchored in ISDA 2006, ICMA 251, CME IMM, NY Fed SOFR,
   national statutes.
2. **Drawdown:** unchanged (13/793 parked; this is a correctness pass on live L0, no parking).
3. **Challenge me:** (a) **F3 chose raise over snap** — `n==0` on a non-business day is undefined and
   fails loud; if the RFR fixing use-case later wants a snap, it should `adjust()` with a *stated*
   convention (preceding for a published fixing), not overload `add_business_days`. (b) **F2 denominator
   = `total/basis`** assumes the compounding-day count equals Σ overnight weights; correct for ACT/360 &
   ACT/365 (the only RFR bases). (c) the style reflow inflates the calendar tables ~740 lines — if the
   compact form is preferred, raise ruff `line-length` or skip format-on-save (the project gates lint
   only, not format).
4. **Smell + debt:** 0 suppressions; removed the now-unused `year_fraction` import; ruff + all gates
   green; format-stable.

## Next
Merge `fix/foundation-audit`. L0 is error-free on the audited surface. Then **Topic 1** (per `18`) after
the F2 model/calibrator foundation (per README forward scope) — or as the ratified sequence directs.
