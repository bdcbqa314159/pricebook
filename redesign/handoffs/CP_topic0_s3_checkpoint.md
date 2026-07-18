# Checkpoint — Topic 0 (Foundation), after Slice 3  (cadence ≤6; stop + report)

Covers **Slices 1–3** of Topic 0 (the cross-asset foundation, L0): `calendars`, `daycounts`,
`schedules`. Slice 0 (`ng-parking`) parked the whole prior ng tree to `ng_parked/` and reset the tree
clean. **Nothing above L0 has been built** (Topic 0's gate is not yet green — Slices 4–6 remain).

**Versions:** v0.55.0 (parking) → v0.58.0. **Tests:** 36 L0 oracles green. **Gates:** `acyclic ·
layers · fields · debt · version · provenance` all green; `ruff` (PLR0913) green.

---

## 1. Oracle-quality audit

Every Topic-0 oracle so far is a **published external reference**, not self-consistency — the strongest
tier available for conventions (there is no "closed form" for a holiday calendar; the authority *is* the
statute / ISDA rule).

| slice | oracle | tier |
|---|---|---|
| calendars | published holiday & observance dates; US-vs-UK Saturday divergence; Juneteenth `since=2021`; Store Bededag `until=2023`; Christmas/Boxing cascade; Tokyo furikae; Fri–Sat weekend | **published standard** (5 U.S.C. §6103, national statutes) |
| daycounts | **ISDA 2006 §4.16 worked examples**; **ICMA Rule 251 (UST semi-annual coupon = exactly 2.0000)**; 30U/360 Feb edges; ACT/365L & NL/365 leap handling; 30E/360-ISDA termination | **published standard** (ISDA/ICMA) |
| schedules | ISDA §4.10 EOM anchoring; all four stubs; adjusted ≠ unadjusted under a holiday; **published IMM (3rd Wed) & CDS (20th) roll tables** | **published standard** (ISDA/CME) |

No self-consistency-only oracle in the topic. The UST-2.0000 case is a genuine **regression oracle**:
the deleted `strict_icma=False` used to price it at 1.9836; ICMA now reprices to exactly 0.5 or raises.

---

## 2. Quarry-drawdown reconciliation

Topic 0 **supersedes but does not yet park** — the set is `git mv`'d to `parked/topic-00-foundation/`
only at Topic 0 close (handoff). So far the ng foundation supersedes three quarry `core/` modules:

| quarry module | superseded by (ng) | status |
|---|---|---|
| `core/calendar.py` (943 LOC) | `foundation/calendar.py` + `market_calendars.py` | **covered** (park at close) |
| `core/day_count.py` (274 LOC) | `foundation/day_count.py` | **covered** (park at close) |
| `core/schedule.py` (143 LOC) | `foundation/schedule.py` | **covered** (park at close) |

Topic-0 module set still to cover (Slices 4–6): `currency`, `notional`, `rate_index`, `fixings`,
`interpolation`, `solvers`, `data_registry`, `serialisable`, `serialization`. (`core/numerical_config`
and `fixed_income/fixed_leg` were retired pre-topic and stay ticked.)

**Global roll-up: files parked / 768 unchanged this checkpoint** (Topic 0 parks at close). Drawdown is
reporting, not steering (redesign/12) — the honest denominator for Topic 0 is its own module set.

---

## 3. Design choices to challenge ("challenge me")

1. **`Calendar` = one frozen value + 37 declarations, identity-keyed.** The quarry's ~38 subclasses +
   currency-keyed registry collapsed to data; currency→calendar is a lookup (C1). The per-rule `observe`
   flag was **lifted** to a calendar-level `Observance` regime (I verified every calendar observes its
   fixed holidays uniformly, so the lift is behaviour-preserving). Challenge: is lifting `observe` a
   faithful ADAPT, or did it lose a per-holiday case I didn't spot?
2. **Furikae as an `Observance` value handled at set-level** (not per-date), because the Japanese
   substitution must walk forward past consecutive holidays. Clean, or should Tokyo stay bespoke as the
   quarry had it?
3. **ACT/365L definition.** A gap convention (not in the quarry) — I implemented **366 iff a 29 Feb lies
   in `[start,end)`, else 365**. There are ISMA variants keyed to coupon frequency / the period-end
   year. Is the leap-day-in-period rule the one we want, or a frequency-keyed variant?
4. **`CouponPeriod` carries `is_final`** (the 30E/360-ISDA termination flag) alongside the ICMA
   reference anchors. Two conventions share one context type, each using different fields — mild smell,
   but it keeps `year_fraction` at ≤5 args. Accept, or split into two types?
5. **`year_fraction` is a 5-arg primitive** (at the ceiling). Ruled with the user: keep the primitive;
   the ergonomic entry point is `Accrual.year_fraction(...)` in Slice 4 (Accrual bundles start+end+
   day_count). Not a `Period` type now (one consumer = premature). Confirm this is the intended shape.
6. **"BusinessDayConvention += the EOM variant"** (handoff): I read this as the **`RollRule.eom`**
   schedule-level mechanism (ISDA §4.10 anchoring), not a new `BusinessDayConvention` enum value.
   `NEAREST` was added. Is a distinct `BDC.EOM` value wanted, or is `RollRule.eom` the intended home?
7. **Long stubs merge by construction** (the `first_gap < months*30*0.5` heuristic is shed): LONG always
   merges a genuine stub with its neighbour, SHORT always keeps it. Confirm this matches desk practice.

---

## 4. Smell + debt scan

- **`verify.py debt` green — zero suppressions** (removed two `# pragma: no cover` markers that would
  have tripped it; the defensive `raise AssertionError` stay). No `# type: ignore`, no skips, no
  load-bearing TODOs.
- `fields` green (largest L0 dataclass = `Calendar`, 4 fields; `CouponPeriod` 4; `RollRule`/`ScheduleTerms`
  3). `PLR0913` green (`year_fraction` at the 5-limit, `build_schedule` 3).
- **No `dateutil`** — schedule month arithmetic is stdlib (day clamped to month length), so the clean
  tree gains no runtime dependency.
- **Content gap to flag (not code debt):** several EM calendars are **fixed/secular-only** (Riyadh,
  Cairo, Istanbul, Tel Aviv, Beijing, Seoul, Mumbai, Bangkok) — they omit **lunar/religious holidays**
  (Islamic Eid, Hebrew, Chinese New Year, Hindu, Thai Buddhist). Inherited verbatim from the quarry
  (which labels them "approximate"/"secular only"). Real, and a future data slice — flag for Cowork
  whether Topic 0 owes lunar-calendar support or defers it with the EM-rates topic.

---

## 5. Spine-conformance audit (5th input, post-`black.py`)

Every module placed by what it **is**, verified module-by-module:

| module | layer | finance-free? | notes |
|---|---|---|---|
| `foundation/calendar.py` | L0 ✓ | yes (`verify.py layers`) | pure date/observance logic; no pricing vocabulary |
| `foundation/market_calendars.py` | L0 ✓ | yes | 37 declarations; imports only `calendar` (same layer) |
| `foundation/day_count.py` | L0 ✓ | yes | conventions + `year_fraction`; "coupon"/"UST" only in prose |
| `foundation/schedule.py` | L0 ✓ | yes | schedule/roll math; imports only `calendar` |

`verify.py layers` is **wired into CI** and green. No module reaches upward; `market_calendars` →
`calendar`, `schedule` → `calendar`, `day_count` → `calendar` (TYPE_CHECKING only) are all equal-rank
L0 edges. No `Money`/pricing type has leaked into L0 (those arrive as L0 *value types* in Slice 4).

---

## 6. Ready-for-next / named next checkpoint

**Topic 0 remaining (Slices 4–6), before the Topic-0 gate:**
- **S4 `money-quantity`** — `Currency`/`CurrencyPair` · `Money` · **`Quantity`** (commodities) ·
  `Cashflow` · `Leg` · `Accrual` (and `Accrual.year_fraction`, per §3.5).
- **S5 `index-identity`** — declarative `RateIndex` (full RFR set) · generic `FixingHistory`.
- **S6 `numerics-config`** — `Interpolator` · solvers · distributions · **complete `NumericalConfig`** ·
  `PricingResult`/`PricingFailure` · the serialisation pattern.

**Named next checkpoint — Topic 0 GATE** (after Slice 6): the full foundation green, then park the
Topic-0 quarry set to `parked/topic-00-foundation/` and refresh the reconciliation roll-up. Topic 1
(yield curves) does not begin until that gate is green.

**Ask for Cowork:** rule the §3 challenge list — especially (3) the ACT/365L variant, (6) whether a
`BDC.EOM` value is wanted, and (§4) whether Topic 0 owes lunar-calendar support or defers it.
