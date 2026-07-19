# Foundation Audit Report

**Date:** 2026-07-19
**Scope:** `/Users/bernardocohen/work/analysis/foundation/` — 16 modules, ~3,215 lines
**Context:** Foundation layer of a derivatives pricing library intended for multiple users, competing with QuantLib. Audited before building the next layers (market-data, models, trade, product).
**Method:** Four independent adversarial reviews run in parallel — numerical critic (11-lens methodology with hand calculations), correctness-bug critic, architecture review, and cross-asset market-practice review — then merged and deduplicated.

---

## Verdict

**The design is genuinely good — better than QuantLib on the axes it targeted — but it is not ready to build layers on.**

All four auditors independently praised the same things: no global evaluation date, frozen value semantics, a clean dependency DAG, explicit conventions with ISDA/statute citations, honest coverage flags (`SECULAR_ONLY`), and RFR lookback/lockout/observation-shift mechanics that are actually correct (rare). Nothing needs a redesign.

But there are **5 confirmed computational bugs, 4 market-data/convention bugs, and a set of structural gaps** that become expensive the moment upper layers depend on them. One round of fixes first.

---

## Tier 1 — Confirmed bugs (hand-verified with counterexamples)

### 1.1 ACT/ACT AFB undercounts leap-to-leap spans

- **Where:** `day_count.py:211-226` (`_sub_one_year` + `_act_act_afb`)
- **Counterexample:** `year_fraction(date(2004,2,29), date(2008,2,29), ACT_ACT_AFB)` returns **3.9972677…** instead of **4.0**. (Equivalently 2016-02-29 → 2020-02-29.)
- **Cause:** `_sub_one_year` clamps 29-Feb to 28-Feb of the prior year; the whole-year loop's `>= start` comparison then drops the last whole year, leaving `stub_end` *before* `start` — a literal negative-day stub folded into the result.
- **Impact:** ~1 day of accrual error on a 4Y AFB span; silently violates the exact-whole-years property AFB is defined by. Found and executed independently by both critics.
- **Fix:** count whole years by comparing (year, month, day) with an explicit 29-Feb ↔ 28-Feb equivalence, rather than via `date` subtraction that loses the leap day. At minimum, guard the stub against negative days and add the missing whole year.

### 1.2 `LOG_LINEAR` + `CONTINUE_SLOPE` extrapolation produces negative discount factors

- **Where:** `interpolation.py:80-90` (`_boundary_slope`), `interpolation.py:107` (`_extrapolate`)
- **Counterexample:** DF nodes `xs=[1,2,5]`, `ys=[0.97,0.94,0.80]`, right-end extrapolation to t=30: code gives **DF(30) = −0.275**; correct log-linear extrapolation (constant continuously-compounded forward) gives **+0.209**.
- **Cause:** the boundary slope is computed in value space and extended linearly in value space, breaking the log-space invariant that is the entire point of `LOG_LINEAR`.
- **Impact:** the natural configuration for extending a rates curve past its last pillar yields arbitrage-violating (negative) DFs and NaN forward rates.
- **Fix:** when `method is LOG_LINEAR`, compute the slope of `log(y)` vs `x` and extrapolate as `exp(log(y_end) + slope_log·(x − x_end))`.

### 1.3 BUS/252 and CDI accrual count business days over different intervals

- **Where:** `day_count.py:229-240` (`business_days_between`, counts `(start, end]`) vs `rate_index.py:146-159` (`_overnight_days`, counts `[start, end)`)
- **Counterexample:** any accrual period whose start or end is a São Paulo holiday — the two paths disagree by exactly one business day (1/252 ≈ 0.4% of a year).
- **Impact:** a CDI swap's accrual exponent and its BUS/252 discount-factor year fraction are computed on different day counts; the BR curve is internally inconsistent. ANBIMA standard is `[start, end)`, so the day-count side is also non-standard.
- **Fix:** make `business_days_between` count `[start, end)`, or route BUS/252 through the same primitive `_overnight_days` uses. Note: the S16 "recorded invariant" claiming no consumer needs the alternative convention is contradicted by the CDI consumer.

### 1.4 Backward schedule generation: EOM keyed on the wrong end + roll-day drift

- **Where:** `schedule.py:163-199`
- **Two defects:**
  1. The EOM snap tests `start == _end_of_month(start)` even when generating backward from `end`. Market practice (ISDA §4.10; QuantLib) keys the EOM decision on the generation anchor — maturity, for backward generation. A swap maturing 2030-06-30 with a front stub generates interior dates by clamping accident, not by rule.
  2. Iterative stepping from clamped dates loses the roll day: end = May 31, quarterly backward, no snap → Feb 28 → **Nov 28** (should be Nov 30), and every earlier date inherits the 28th.
- **Fix:** generate as `anchor + k·tenor` from the seed date (QuantLib-style), or carry the roll day explicitly.

### 1.5 Japanese furikae substitution is non-deterministic

- **Where:** `calendar.py:369-378` (`_furikae_substitutes`)
- **Cause:** iterates a `set[date]` while the inner walk consults the mutable substitute set, so overlapping substitute walks depend on set iteration order. The `lru_cache` at `calendar.py:351` then freezes whichever answer came first, per process.
- **Impact:** two processes can disagree on whether a given Tokyo date is a business day.
- **Fix:** iterate `sorted(holidays)` — furikae is defined to walk forward in date order anyway.

---

## Tier 2 — Market-data / convention bugs (wrong answers on real dates)

### 2.1 SOFR declared with a non-standard 2-day observation shift

- **Where:** `rate_index.py:268-276` — `SOFR = ... RfrConvention(observation_shift=2, payment_delay=2)`
- **Issue:** the ISDA-standard SOFR OIS is plain compounded-in-arrears over the calculation period — no observation shift, no lookback — with a 2-business-day payment offset. `observation_shift=2` is the loan-market/FRN convention. As declared, every SOFR swap coupon compounds over the wrong window (sign-varying PV error, wrong DV01 bucket at window edges).
- **Note:** the engine mechanics in `accrued_rate` (`rate_index.py:178-197`) are *correct* per the ARRC user's guide — this is a one-line declaration bug: `RfrConvention(payment_delay=2)`. The LIBOR-fallback index (`rate_index.py:298-310`) correctly uses shift=2 per the Bloomberg/ISDA fallback methodology — keep it.
- **Urgency:** must land before any fixings or curves are stored against the wrong window.

### 2.2 The USD calendar is wrong for SOFR

- **Where:** `market_calendars.py:53-74` (`NEW_YORK_SIFMA` with `Observance.US`)
- **Three problems:**
  1. **Good Friday is missing.** SOFR publishes on U.S. government-securities business days; there is no SOFR fixing on Good Friday. Every SOFR compounding window spanning Easter picks up a phantom fixing, and `FixingHistory.rate` (`rate_index.py:114-118`) will raise on real data — breaks annually.
  2. The Sat→Fri observance marks **2021-12-31** a holiday, but SIFMA/Treasury/Fed were open and SOFR was published that day.
  3. One USD calendar cannot serve rates (SIFMA), equities (NYSE — different Good Friday/half-day/observance behaviour), and EFFR (Fed bank days — open Good Friday).
- **Fix:** add `US_GOVERNMENT_SECURITIES` (SIFMA, with Good Friday), `NYSE`, and a Fed-bank calendar; bind SOFR to the SIFMA one.

### 2.3 LONDON missing 2022–23 one-off holidays

- **Where:** `market_calendars.py:91-105` — only recurring rules; no 2022 Platinum Jubilee (Jun 2–3), no state funeral (Sep 19 2022), no coronation (May 8 2023), no 2020 VE-Day bank-holiday move.
- **Impact:** historical SONIA compounding through those windows is wrong (missing fixing dates, wrong day-weights). Past fixings enter live coupons — historical accuracy is not optional for RFR indices.
- **Fix:** the rule DSL (`calendar.py:110-198`) has no one-off-date combinator. Add `dates(*ds)` (~5 lines) and backfill (also US days of mourning and similar one-offs).

### 2.4 Tokyo equinoxes hardcoded — wrong this year

- **Where:** `market_calendars.py:116,122` — `fixed(3,21)` / `fixed(9,23)`
- **Issue:** Vernal Equinox Day is Mar 20 in 2024/2025/2026; Autumnal was Sep 22 in 2024. TONA fixing dates are wrong in the current year. Standard astronomical approximation exists (QuantLib's `Japan` calendar implements it).
- **Also missing:** the sandwiched-holiday rule (*kokumin no kyūjitsu*, Silver Week), the Emperor's Birthday move (`fixed(2,23)` needs `since=2020`; Dec 23 until 2018), the 2020/2021 Olympic holiday shifts.

---

## Tier 3 — Structural: fix before building upper layers

### 3.1 It is not a package and does not import

- Every internal import is `from pricebook_ng.foundation...` but there is no `pricebook_ng/` package, no `pyproject.toml`, no `__init__.py`, no `__version__`, no `py.typed` (fully typed code that downstream mypy will see as `Any`), no `LICENSE`, and the oracle tests cited throughout the docstrings are not in the tree. `import foundation` fails on the first internal import.
- **For a multi-user library:** define the public API surface in `__init__.py` (explicit exports / `__all__`) **before** users freeze it for you; bring the tests in-tree — they are part of the foundation contract.
- **Rename `calendar.py` → `calendars.py` now.** `tenor.py:17` imports stdlib `calendar`; the moment the `foundation/` directory itself lands on `sys.path` (test runner, notebook), the local module shadows stdlib and breaks with a confusing error. Cheap now, expensive after four layers import it.

### 3.2 `JointCalendar` cannot be used where a calendar is required

- **Where:** `calendar.py:381-398` — implements only `is_weekend`/`is_holiday`/`is_business_day`; lacks `adjust`, `add_business_days`, `day_type`, `identity`. Every consumer slot is typed `Calendar`: `RollRule.calendar` (`schedule.py:86`), `year_fraction` (`day_count.py:65-66`), `settlement_date` (`settlement.py:89-93`), `accrued_rate` via `index.accrual.roll.calendar` (`rate_index.py:167`).
- **Impact:** cross-currency swap schedules (TARGET+NY), FX dates (joint by definition), NY+LON payment calendars — day-one trade-layer needs — are a type error and a runtime `AttributeError` today. Flagged independently by both the architecture and market-practice reviews as **the single most expensive thing to retrofit** once `RollRule` has dependents.
- **Fix:** define a `CalendarProtocol` (`is_business_day`, `adjust`, `add_business_days`, `identity`) in the calendar module; implement `adjust`/`add_business_days` on `JointCalendar` (trivial — the algorithms only call `is_business_day`); retype every consumer slot to the protocol. Keep the `lru_cache` on the concrete type.

### 3.3 Registry inconsistency — calendars and rate indices are closed

- Currencies and units have public `register_*` (`money.py:91,147`); calendars (`market_calendars.py:48,772`) and rate indices (`rate_index.py:313-315`) are closed — yet these are the *most* open-ended domains. The docstrings cite TIIE 28D as a motivating case, and TIIE is not even registered.
- **Fix:** public `register_calendar(cal)` and `register_rate_index(idx)`. Make **all** `register_*` raise on conflicting re-registration — `register_currency` currently overwrites silently (`money.py:94`), which also breaks its "interned so `is` holds" claim. Add an unregister/context helper for test isolation.

### 3.4 No serialization convention for registry identities

- `Calendar` holds rule closures (`calendar.py:33,242`), so equality is object identity — it can only serialize **by name**; atoms (Money, Accrual, Cashflow) serialize by value. Half the codebase has `to_dict`/`from_dict`; `Calendar`, `RateIndex`, `Tenor`, `Schedule`, `SettlementTerms` have none.
- **Impact:** the trade layer must serialize a trade referencing an index and a calendar on day one. If the by-name-vs-by-value convention is not fixed in foundation (which owns the identities), each layer invents its own.

### 3.5 `Schedule` discards its provenance

- **Where:** `schedule.py:104-111` — two bare date tuples; no stub flags, no ICMA reference periods, no `terms`.
- **Consequences:** ACT/ACT ICMA on a stub needs the regular reference period — exactly what generation knew and threw away; a leg builder cannot recover it without re-deriving the generation logic. `ScheduleTerms` also cannot express explicit stub boundary dates (`firstRegularPeriodStartDate`/`lastRegularPeriodEndDate` — needed for any trade booked mid-life; the four-member `StubType` at `schedule.py:63-67` cannot represent them). `RfrConvention.payment_delay` (`rate_index.py:78`) is dead code because `Schedule` has no payment-date column — decide where pay-lag applies (often on a *different* calendar than fixing, for XCCY).
- **Related:** ACT/ACT ICMA (`day_count.py:167-180`) supports one reference period, so **long stubs are not computable** (Rule 251 sums over multiple quasi-coupon periods) — raise loudly on long stubs until supported.
- **Fix:** emit per-period records (start, end, is_stub, reference anchors, payment date), and add optional explicit stub dates to `ScheduleTerms`.

### 3.6 FX spot machinery is absent

- **Where:** `money.py:263-274` (`CurrencyPair.spot_lag: int`), `settlement.py:89-93` (single-calendar T+lag)
- **Issue:** the market FX spot algorithm requires per-currency intermediate-day checks, joint validity of the candidate date in both centres, and USD-holiday involvement for crosses. Single-calendar T+n is not it. Forward points are quoted *off spot* — the FX market-data layer, FX options (expiry → delivery), and XCCY spot-start conventions all depend on a correct `fx_spot_date(pair, trade_date)` primitive. "USD/CAD is 1" lives in a docstring, not in code — no pair-conventions registry (quote order, standard lags, cross triangulation).
- **Also:** `spot_lag` participates in `CurrencyPair` equality, conflating identity with convention — `CurrencyPair(USD, JPY)` ≠ `CurrencyPair(USD, JPY, spot_lag=1)` fragments dict keys (`FxFixing` keys on the pair, `underlying.py:87`). Move it to a convention lookup or exclude from equality.

### 3.7 Smaller must-decides before dependents accrete

- **`_denominator` silent fallback** (`rate_index.py:142-143`): any day count other than ACT/365F silently gets 360 — including nonsense inputs. Violates the library's own no-silent-fallback brand (the ICMA fallback deletion at `day_count.py:7-9` exists precisely because of such a bug). Raise on unsupported conventions.
- **`PricingResult.sensitivities: Mapping[str, float]`** (`results.py:48`): a stringly-typed greek bag — QuantLib's `results_` dictionaries in different clothes. Real risk output is curve × pillar × bump type. Define a `RiskKey` type or delete the field until the models layer defines it. `cashflow_breakdown: tuple[Money, ...]` (`results.py:47`) is a parallel array with no dates or flow identity — pair it with flows or drop it.
- **Delete the four unpopulated `Underlying` siblings** (`underlying.py:53-118`): `ReferenceEntity`, `InflationIndex`, `FxFixing`, `EquityUnderlying`, `CommodityUnderlying` have zero consumers and guessed fields (`fixing_time: str` = "16:00 London"; required `grade: str`). Keep the `Underlying` protocol and `AssetClass` enum. Each sibling costs one file-touch to reintroduce correctly; a breaking change if built upon.
- **`FixingSource` protocol**: `FixingHistory` (`rate_index.py:107-118`) is a concrete fixings store — market-data-layer property. Have `accrued_rate` accept a `FixingSource` protocol (`rate(index_name, on) -> float`); keep `FixingHistory` as the trivial implementation. One line now; a two-truths migration later.

---

## Tier 4 — Scheduled with their asset-class topics (not urgent, not optional)

| Item | Where | Note |
|---|---|---|
| Index registry too thin | `rate_index.py:267-310` | No **EURIBOR_6M** (the standard EUR vanilla floating leg), EFFR (`AccrualMethod.AVERAGED` exists for it), BBSW, AONIA, CORRA, TIIE 28D, SELIC, WIBOR/PRIBOR/BUBOR/JIBAR |
| Zero/negative tenor → infinite loop | `tenor.py:42-44`, `schedule.py:175,190` | `Tenor.parse("0D")` / `"-3M"` accepted; a zero step never advances the schedule loops. Reject non-positive counts |
| `distributions.py` too thin | `distributions.py:19-33` | Models layer needs bivariate normal CDF (Genz/Drezner) and non-central χ² on week one, or engines violate the scipy rule |
| `least_squares` cannot bound | `solvers.py:59-70` | `method="lm"` hardcoded; every stoch-vol calibration needs bounds (Feller, \|ρ\|<1). Add `trf` variant while API is unfrozen |
| No `TimeMeasure` concept | (absent; invariant recorded at `rate_basis.py:8-11`) | A frozen `TimeMeasure(anchor, day_count)` with `t(d) -> float` — the one thing everything above touches; painful to retrofit |
| CDS maturity roll pre-2015 | `schedule.py:234-246` | Since CDS2015, maturities roll semiannually (Jun/Dec 20) while coupons stay quarterly. Add `standard_cds_maturity` before the credit layer |
| `is_holiday` forward year-spill | `calendar.py:290-292` | Checks `year`/`year+1` but never `year−1`; a Dec holiday observed forward into January is missed. Latent (no current declaration triggers it) — trap for the next calendar |
| `observe()` hardcodes Sat/Sun | `calendar.py:268-285` | Wrong for any future mondayising FRI_SAT calendar (all current FRI_SAT markets use `Observance.NONE`). Derive shift days from `weekend.value` or assert incompatibility |
| `NEAREST` tie-break | `calendar.py:311-313` | Ties roll backward; QuantLib and common practice roll forward. Undocumented either way — decide and document |
| `log(y≤0)` unguarded | `interpolation.py:74` | Negative rates/spreads or an underflowed DF raise a bare `math domain error`; add a curve-level guard message |
| `convert_rate` on negative growth factor | `rate_basis.py:48-64` | `math.log(−x)` crash; negative `factor ** (1/(m·t))` NaN path. Decide reject-vs-define |
| CDI rate application trap | `rate_index.py:199-205` | Returns an annualized exponential rate; a consumer applying `r·yf` (simple) is silently wrong by convexity. Consider returning a growth factor or tagged result |
| Interpolators rebuilt per call | `interpolation.py:76,88` | O(N·M) vs O(N+M); already ponytail-flagged as a Topic-1 deferral. `_boundary_slope` doubles it |
| `Money` is unrounded float | `money.py:186` | Deliberate and correct for pricing — but `minor_units` is decorative. Document so no ledger/settlement code assumes rounding |
| Two `frequency` concepts, no bridge | `schedule.py:33` vs `day_count.py:57` | `Frequency` (tenor-step) vs ICMA `frequency: int` (per-year). Decide the 28D/TIIE answer before the trade layer guesses |
| No time-of-day/timezone story | `underlying.py:88-90` | `fixing_time: str` is a placeholder; expiry cuts, equity closes need `datetime.time` + IANA zone. Decide before FX options |
| `Weekend` is time-invariant | `calendar.py:45-48` | Saudi 2013 change; Israel moves to Mon–Fri effective 2026 (`TEL_AVIV` wrong going forward). Record as approximation or add `since=` support |
| Month-arithmetic triplication | `tenor.py:54-60`, `schedule.py:115-131`, `day_count.py:109-112` | Three near-duplicate add-months/EOM helpers. DRY debt, fine to leave |

---

## Verified clean (credit where due)

Hand-verified against ISDA worked examples, published dates, and known benchmarks:

- **Day counts:** 30U/360 with full SIA February rules (5 ISDA table cases), 30E/360, 30E/360 ISDA final-Feb termination rule, ACT/ACT ISDA (matches the ISDA worked example to 1e-15), ACT/ACT ICMA (strict, divide-by-zero guarded), ACT/360, ACT/365F, ACT/365L, NL/365, 1/1. `Accrual.__post_init__` rejects reversed/zero spans.
- **Rate basis:** conversions round-trip to ~1e-15; semi→annual and semi→continuous match hand calculations.
- **RFR mechanics:** lookback (rate-date shift, unshifted weights) vs observation shift (shifted window and weights) correctly distinguished per the ARRC user's guide; day weights sum exactly (Friday carries 3); lockout indexing verified for lockout 0 and >0.
- **Calendar machinery:** Gregorian and Orthodox Easter (verified 2020–2025 against published dates), IMM 3rd-Wednesday (2024–2025), CDS 20th, Victoria Day (including the May-25-is-Monday tie), Midsummer Eve, Ley Emiliani mondayisation, Christmas/Boxing collision bump, `add_business_days` sign/termination. `add_business_days(d, 0)` raising on a non-business day is good design.
- **Value types:** `money.py` equality/hash consistency, currency-mixing `TypeError`, `MappingProxyType` registry views; `results.py` `clean` None-handling; `numerical_config.py` validators and schema-version guard; `cashflow.py` discriminated dispatch; `settlement.py` invariants.
- **Architecture:** no global evaluation date anywhere (checked every file); no Handles/observer graph; frozen dataclasses and value semantics throughout; clean dependency DAG with no cycles; `Schedule` carrying both unadjusted and adjusted dates; `Compounding` vs `AccrualMethod` split; layering discipline in the docstrings genuinely well held. Correctly deferred: curve/term-structure abstractions, Hagan–West, CSA mechanics, credit-event mechanics.

---

## What was NOT verified

- Individual national holiday **lists** in `market_calendars.py` against official exchange calendars (mechanisms were verified; the 37 markets' data content is oracle-test territory — and the oracle tests are not in the tree).
- ACT/365L freq>1 semantics against a primary ICMA/ISDA worked example (code matches the auditors' reading of ISDA §4.16(i)).
- End-to-end LONG_FRONT/LONG_BACK stub schedules (merge logic read as correct; recommend a targeted hand-calc).
- scipy version-dependent behaviour of `brentq`/`newton`/`least_squares`/`norm` (thin adapters assumed correct per contract; check whether `brent`'s `tol=1e-14` is clamped).
- The project's own test suite — it could not be run because the package does not import (see 3.1).

---

## Recommended execution order

1. **Package it** (3.1): `pricebook_ng/foundation/`, `pyproject.toml`, `__init__.py` exports, `py.typed`, version, tests in-tree, `calendar.py` → `calendars.py`. Nothing is verifiable until it imports.
2. **Tier 1 bugs** — each counterexample above is a ready-made regression test.
3. **Tier 2 convention fixes** — before any fixing/curve data is stored against wrong windows or wrong calendars.
4. **Tier 3 structural items** — `CalendarProtocol` first (most expensive to retrofit), then registries, serialization convention, schedule metadata, FX spot, and the small must-decides.
5. **Tier 4** rides along with its respective asset-class topic — scheduled, not discovered.

---

## Closure disposition (2026-07-19)

Closed on branch `fix/foundation-audit-closure` (commits `main..` = v0.75.0–v0.80.0). Method: red→green
throughout — every Tier-1/2 counterexample landed as a **failing** test *before* its fix. Rule: this
report is prefixed `closed_` only because every finding below is **fixed-with-a-test** or **ledgered**
with a named re-open trigger (`OPEN.md` → "Foundation audit closure — Tier-4 & deferred-scope ledger").

| finding | disposition | where |
|---|---|---|
| 1.1 AFB leap-to-leap | **FIXED** v0.76.0 (Phase 1) — direct year-shift `_shift_years`, leap-exact | `test_audit_closure` 1.1 |
| 1.2 LOG_LINEAR CONTINUE_SLOPE neg-DF | **FIXED** v0.76.0 — log-space extrapolation (also A1 doc, v0.80.0) | 1.2 |
| 1.3 BUS/252 vs CDI interval | **FIXED** v0.76.0 — one half-open `[start,end)` primitive (S16 withdrawn, A2) | 1.3 |
| 1.4 backward-schedule EOM/roll drift | **FIXED** v0.76.0 — anchor-based `_step_k`, EOM keyed on generation seed | 1.4 |
| 1.5 furikae non-determinism | **FIXED** v0.76.0 — iterate `sorted(holidays)` | 1.5 |
| 2.1 SOFR 2-day obs-shift | **FIXED** v0.77.0 (Phase 2) — `RfrConvention(payment_delay=2)`, no shift | 2.1 |
| 2.2 USD calendar | **FIXED** v0.77.0 — `US_GOVERNMENT_SECURITIES` (Good Friday, `SUNDAY_ONLY`); SOFR bound. **Deferred:** NYSE + Fed-bank/EFFR → `OPEN.md` **AC-2.2b** | 2.2 |
| 2.3 LONDON one-offs | **FIXED** v0.77.0 — `dates()` combinator + gated `nth` | 2.3 |
| 2.4 Tokyo equinoxes | **FIXED** v0.77.0 — astronomical `equinox()`, Emperor moves. **Deferred:** Silver Week + Olympic shifts → `OPEN.md` **AC-2.4b** | 2.4 |
| 3.1 not a package | **FIXED** v0.75.0 (Phase 0) — `__init__`/`__all__`, `py.typed`, root pyproject, `calendar.py`→`calendars.py`, tests in-tree | — |
| 3.2 JointCalendar unusable | **FIXED** v0.78.0 (Phase 3) — `CalendarProtocol`, `adjust`/`add_business_days`/`identity`, all slots retyped | — |
| 3.3 closed registries | **FIXED** v0.78.0 — public `register_calendar`/`register_rate_index`; all raise on conflict; `temporary_*` | — |
| 3.4 no serialization convention | **FIXED** (commit `d7fc23c0`, v0.79.0 window) — identities by name, atoms by value | `test_audit_closure` 3.4 |
| 3.5 Schedule discards provenance | **FIXED** v0.79.0 (Phase 3b) — `SchedulePeriod`, `RegularPeriod`, `PaymentRule`; ICMA 251.2 (long stubs computable) | 3.5, 3b tests |
| 3.6 FX spot absent | **FIXED** v0.78.0 — `fx_spot_date` (joint count + USD-holiday-for-cross); `spot_lag` out of `CurrencyPair` identity into a pair-conventions lookup. **Deferred:** FX quote-order/triangulation (L1) → `OPEN.md` **AC-3.6b** | settlement tests |
| 3.7 `_denominator` silent 360 | **FIXED** v0.78.0 — raises on unsupported conventions | 3.7 |
| 3.7 `sensitivities`/`cashflow_breakdown` | **FIXED** v0.80.0 (Phase 4) — deleted (A3); return with the L4/L5 layer | — |
| 3.7 four `Underlying` siblings | **FIXED** v0.80.0 — deleted (A3); protocol + `AssetClass` kept; each returns with its asset class | — |
| 3.7 `FixingSource` protocol | **FIXED** v0.78.0 — `accrued_rate` accepts `FixingSource`; `FixingHistory` is the trivial impl | — |
| **Tier 4** (18 items) | **LEDGERED** — `OPEN.md` **AC-T4.1 … AC-T4.18**, each with its asset-class re-open trigger | — |

**Not-verified items (report §"What was NOT verified")** remain oracle-test territory for the market-data
topic (national holiday lists) and later layers (stub end-to-end, scipy version behaviour); tracked
implicitly by those topics' own oracle gates, not re-listed here.
