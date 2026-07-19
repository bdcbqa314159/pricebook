> # ⛔ SUPERSEDED — DO NOT USE
> Replaced by **`handoff_topic0_foundation.md`**.
> This version scoped the foundation to what *multicurve* needed (EUR/USD/GBP calendars, rate
> indices, 3 `NumericalConfig` knobs). That is the retrofit trap: the foundation serves *every*
> topic, so it must be designed **cross-asset complete** first — see `redesign/16_topic0_foundation.md`.
> Retained only as history. Its per-slice mining detail (ICMA, EOM, RFR conventions) was carried
> forward into the new hand-off.

# Hand-off — Topic 1, cluster 1: parking + conventions  (SUPERSEDED)

Read first: `redesign/12_domain_build_order.md` · `13_topic_migration_and_parking.md` (esp. §3 tick
mechanism, §5 target/use/apply) · `14_topic1_object_model.md` (incl. §6b corrections) ·
`handoffs/rulings_ng_parking_and_topic1_design.md` · `parked/topic-01-yield-curve/MANIFEST.md`.

**Standing rule for every slice below:** *mine the quarry for CONTENT, never for STRUCTURE*
(`CLAUDE.md §4`). The transformation gate (#13 §5.3) applies at write time. Five slices → checkpoint.

---

## Slice A — `slice/ng-parking` (housekeeping, no behaviour)

```
git mv the ~30 out-of-topic ng modules to ng_parked/ (list in rulings_ng_parking_and_topic1_design §1).
Their tests move with them and are EXCLUDED from the active suite but RETAINED (re-base oracles).
Write ng_parked/MANIFEST.md: module | topic it belongs to | its re-base oracle | parked-at version.
Active suite + verify.py all green after the move. Version MINOR. No behaviour change.
```
This also disposes of `foundation/black.py` — it parks with the options material.

---

## Slice B — `slice/conventions-daycount`

**Mine** `core/day_count.py` (274 LOC). Carry all 7: ACT/360 · ACT/365F · 30/360 (**30U/360 US SIA
*with* the end-of-February rule — explicitly NOT ISDA-2006 Bond Basis; 4 ordered rules**) · 30E/360
(unconditional `min(day,30)` both ends) · ACT/ACT ISDA (year-boundary split) · **ACT/ACT ICMA** ·
**BUS/252**.

**Carry these specifics — they are the hard-won part:**
- **ACT/ACT ICMA** = `(end−start) / ((ref_end−ref_start) × frequency)`, requiring coupon anchors.
  ng takes them as a **`CouponPeriod`/`Accrual` value object**, never 3 loose args (§3b).
  **Drop the quarry's `strict_icma` dual-mode entirely** — the non-strict path silently fell back to
  ACT/365F and priced a UST coupon at 1.9836/2.0164 instead of 2.0000. Missing anchors ⇒ **raise**.
- **BUS/252** = business days / 252. **Never default the calendar** (quarry silently instantiates São
  Paulo — a hidden wrong default). Calendar is required. Counting is **half-open `[start, end)`**
  (start-inclusive, end-exclusive) — the one `business_days_between` primitive, matching ANBIMA/CDI
  (audit closure A2; the earlier "start-exclusive / end-inclusive" S16 record is **withdrawn**).
- `date_from_year_fraction` — quarry uses 365.25 rounding and is lossy; decide deliberately.

**Oracles (published, not self-consistency):**
1. **ISDA 2006 §4.16 worked examples** for each convention.
2. **ICMA Rule 251 examples**; plus the regression: a UST semi-annual coupon period must give
   **exactly 2.0000** (the documented failure value is 1.9836/2.0164).
3. 30U/360 February edge cases (last-day-Feb both ends; d1=31; d2=31 ∧ d1=30).
4. Cross-check against the quarry implementation on a date grid (both trees importable).

---

## Slice C — `slice/conventions-calendar`

**Mine** `core/calendar.py` (943 LOC — the largest and richest). Carry the **holiday-rule DSL**
(`fixed` / `easter` / `orthodox` / `nth` / `monday` modifiers, `since=`/`until=` year-gating) and the
**three distinct observance rules**: US 5 U.S.C. §6103 (Sat→prev Fri, Sun→next Mon), Commonwealth
(Sat→Mon+2, Sun→Mon+1), Johannesburg (Sunday-only). Plus `JointCalendar` and the Christmas/Boxing
collision resolution.

**Apply C1 — key calendars by IDENTITY, not currency.** `TARGET`, `LONDON`, `NEW_YORK_SIFMA`, … ;
**currency → calendar is a lookup**, never an identity. This is the quarry's own admitted flaw (it
cannot express USD+GBP joint or NY-vs-SIFMA).

**Scope now:** the mechanism + the calendars for **EUR (TARGET), USD (SIFMA/Fed), GBP (London)**.
The remaining G10 are cheap once the DSL exists — completing them is what lets `core/calendar.py`
tick deletable. EM calendars are self-declared placeholders (no Islamic/lunar/CNY holidays) →
`deferred→EM topic`, forward-linked.

**Oracles:** known holiday dates per calendar (incl. Juneteenth `since=2021`, Danish Store Bededag
`until=2023`); observance-shift cases (holiday on Sat vs Sun, US vs UK divergence); the
Christmas-Sunday → Boxing-to-27th cascade; joint-calendar union.

---

## Slice D — `slice/conventions-schedule`

**Mine** `core/schedule.py`. Carry `Frequency`, the 4 stub types, forward/backward generation, and
critically the **EOM anchoring rule (ISDA 2006 §4.10): the EOM decision is made ONCE per schedule
from `start`, not per step** — the quarry's documented fix, because backward generation would
otherwise anchor EOM to `end`.

**Apply C2 — the result carries BOTH adjusted and unadjusted dates** (per period: unadjusted
start/end, adjusted start/end, payment date). Accrual uses unadjusted; payment uses adjusted. The
quarry adjusts in place and loses the accrual dates — do not inherit that.

**Shed:** the long-stub merge heuristic (`first_gap < months*30*0.5`) is a guess, not a rule — replace
with an explicit stub specification.

**Oracles:** ISDA schedule examples; EOM anchoring (31-Jan start, monthly, must roll to month-ends);
short/long front/back stubs; adjusted≠unadjusted under a holiday; roundtrip period count.

---

## Slice E — `slice/rate-index`

> **The governing rule for this slice: adding a new index must be a DECLARATION, never a code
> change.** If supporting a new RFR requires touching a function, the design has failed.
> `RateIndex` is **pure declarative metadata** (no computing methods); `get_rate_index("SOFR")`
> returns the populated metadata; **one generic** `accrued_rate(index, accrual, fixings)` reads the
> fields. The only branching is on the declarative `CompoundingMethod`
> (COMPOUNDED ∏(1+rᵢδᵢ)−1 · AVERAGED weighted mean · FLAT single fixing at start−lag).
> **No `SOFRIndex` subclass, no `if index.name == …`** — same anti-pattern as the `isinstance`
> ladders we removed from risk. ESTR vs SONIA differ *only* in their declarations.
>
> **RATIFIED — carry the FULL RFR convention set as fields, now:**
> **observation_shift** (shifts fixing *and* accrual weighting) · **lookback** (shifts the fixing
> date, keeps accrual weighting on the original period — *a different number*, and the quarry cannot
> express it) · **lockout / rate cut-off** (freeze the rate for the final N days — not modelled at
> all in the quarry) · **payment_delay**. Topic 1's OIS pillars only exercise compounding + payment
> delay, but this is *metadata* — precisely the thing that must be declarative — and retrofitting a
> convention distinction later means revisiting every index declaration **and** the accrual function.
> Oracle each field's effect independently (a lookback and an observation-shift index over the same
> period must give **different** rates).

**Mine** `core/rate_index.py` (28 indices). Carry the RFR conventions table — **SOFR** (ACT/360,
shift 2, delay 2), **ESTR** (ACT/360, 2/2), **SONIA** (ACT/365F, **0/0**), plus **EURIBOR 3M/6M**
(ACT/360, lag 2, FLAT compounding). `CompoundingMethod`: COMPOUNDED / AVERAGED / FLAT.

**ng improvement (already ratified):** `RateIndex` carries a **`RollRule` (hence a calendar)** —
the quarry has no calendar field and infers it from currency. Keep our shape.

**Fix, do not inherit:** the module-level `_REGISTRY` is rebound at import by a JSON load that
**replaces the whole dict** (one bad entry silently drops the other 27). ng: explicit construction,
no import-time I/O, no all-or-nothing replace.

**Scope now:** EUR/USD/GBP indices. The other 24 are `deferred→` their currency topics.

**Oracles:** compounded-RFR period accrual against a hand-computed fixing series; observation-shift
vs payment-delay applied to the correct dates; SONIA 0/0 vs SOFR 2/2 divergence on the same period.

---

## Checkpoint
After Slice E: **CP-5 checkpoint** per cadence #11 with all five review inputs (incl. spine
conformance) + the Topic-1 manifest refreshed (rows ticked `covered`/`dead`/`reassigned`, with
evidence). Name the next cluster (expected: curves — `YieldCurve`, `Interpolation`, `CurveSet`).
