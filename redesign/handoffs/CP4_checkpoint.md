# Checkpoint CP-4 — CP-3 tail + the swap decision (drawdown 5 → 7/768; swap + inflation for Cowork)

Closes the **fixed-income vanilla cluster**. Since CP-3 (v0.51, 5/768): two clean tail retires (bond,
fixed_leg → 7/768), one partial-cross held (inflation), the property-based oracle, and the **swap
retire-read** — which resolves Cowork's flagged "is multi-curve owed?" question. **Two decisions are
routed here for Cowork**, not taken unilaterally: the **swap tick** and the **multi-product-module
retire pattern** (inflation).

**Versions:** v0.52.0 → v0.54.0. **Tests:** 286 ng green. **Gates:** `acyclic / debt / version /
provenance / fields` green.

---

## 1. Slices landed (CP-3 tail)

| slice | version | outcome | drawdown |
|---|---|---|---|
| property-serialisation-oracle | (chore) | hypothesis round-trip sweep over all serialisable types | — |
| serialisation-bond | 0.52.0 | **RETIRED** `fixed_income/bond.py` (heaviest; large deferred analytics) | 5 → 6 |
| retire-fixed-leg | 0.53.0 | **RETIRED** `fixed_income/fixed_leg.py` (no-code tick) | 6 → 7 |
| serialisation-zcis | 0.54.0 | ZCIS serialised; `inflation.py` **held partial-cross** (not ticked) | 7 (unchanged) |
| swap analysis | (this doc) | retire-read only — **swap deletable, routed to Cowork** | 7 (pending) |

---

## 2. Oracle-quality audit

- **bond / ZCIS:** round-trip + version-safety self-consistency oracle (same shape as the CP-3 cluster),
  plus the pre-existing closed-form pricing oracles (untouched). bond added to the property sweep; ZCIS
  added to the property sweep.
- **fixed_leg:** **no-code retire** — no new oracle; the tick rests on ng's already-green
  leg/cashflow/swap/bond oracles (which exercise `fixed_coupon_cashflows` + `Cashflow` round-trip) +
  the consumer analysis. This is the honest "several may tick immediately" (§4.5) case.
- **property oracle:** strengthened all round-trips from one-example to generated-instance (hypothesis);
  it immediately earned its keep by surfacing the builder fail-fast on `ACT_ACT_ICMA`/`BUS_252`.
- **No tolerance issues** (dict round-trip is exact float identity). CI gained `hypothesis`.

---

## 3. Quarry-drawdown reconciliation — **7 / 768 (0.91%)**

Retired: `core/numerical_config`, `fixed_income/{deposit, fra, ois, zero_coupon_bond, bond, fixed_leg}`.
Held partial: `fixed_income/inflation.py` (ZCIS + curve superseded; YoY dead; linker deferred).

**The §4 "gaps are overstated" pattern held all the way through — including the two modules everyone
expected to be hard:**
- **bond** — looked load-bearing (rich analytics, 8 instantiations); consumer analysis: all instantiations
  un-crossed, ng supersedes the product + A2 accrued, analytics deferred. Deletable.
- **swap** — Cowork flagged it as "genuinely load-bearing, real residual = multi-curve/curve-pillar."
  Consumer analysis says otherwise (see §4).

---

## 4. Decisions for Cowork

### 4a. Swap — recommend TICK (the flagged concern is a phantom)
The retire-read of `fixed_income/swap.py` (`InterestRateSwap`, 29 production instantiations):
- **Curve-pillar role is a phantom.** The **quarry's own** bootstrap (`curves/bootstrap.py`) takes
  `(maturity, par_rate)` tuples — **never `InterestRateSwap` objects**; ng uses `ParSwapQuote`. The swap
  *product* is a bootstrap pillar in **neither** tree. (Same shape as deposit's "conventions" phantom.)
- **All 29 instantiations are in un-crossed modules** (options/desks/structured/equity/zc_swap/
  curve_trading). The **crossed** consumers that matter — `risk/exposure`, `risk/saccr`,
  `shell/xva_report` — use the **ng `VanillaSwap`**, not the quarry one.
- **Superseded:** product (fixed+float legs) → ng `VanillaSwap`; single-curve pricing → `SwapEngine`.
- **`deferred→` un-crossed:** `par_rate`/`annuity`/`dv01`/`notional_schedule`/`cashflow_schedule`/
  `from_convention` (→ swaption/cms/desks/curve_trading); **multi-curve pv** (`projection_curve`) →
  the multi-curve slice/desks. **No crossed module needs multi-curve swap pricing** (ng risk simulates
  single-curve `VanillaSwap` under Hull-White), so **multi-curve is not a build owed to retire swap.**
- **`dead`:** `amortising`/`accreting`/`roller_coaster` (variable-notional) — consumed only by the
  quarry module's own docstring examples.
- **Recommendation:** tick swap (→ 8/768), add `VanillaSwap` serialisation (build-early). **Held for
  Cowork because Cowork explicitly flagged it** — not because the evidence is weak.

### 4b. Multi-product-module retire — the inflation pattern (needs a rule)
`fixed_income/inflation.py` bundles **four** things; ng built **one** (ZCIS). `YoYInflationSwap` is
`dead` (test-only); `CPICurve` is superseded by the A5 `MarketKey(INFLATION,…)` real curve;
**`InflationLinkedBond` is a whole *unbuilt* product** whose 2 consumers are un-crossed
(`desks/api`, `inflation_indices`) → `deferred→` them. By the letter of `deferred→X`, the module is
deletable — but ticking it **deletes the definition of a product ng never built**. That is a new
pattern, beyond bond's deferred *methods*. **Held partial-cross pending a Cowork rule:** *can a
multi-product module tick when ng has built only some of its products and defers the rest to their
un-crossed consumers, or must every product be built/dead first?*

### 4c. Bond spot-check (from CP-3 §5 / v0.52)
Bond was the first tick resting on a large deferred analytics surface. Flagged then for CP-4 spot-check.
Standing by the tick: no crossed consumer needs bond yield-analytics; they are L4/L5 engine math per the
spine, built when a consumer crosses. Cowork to confirm or un-tick.

---

## 5. Smell + debt scan
- `verify.py debt` green — zero ng suppressions/ignores/skips/TODOs across the tail.
- Rule-of-two lifts remain correctly placed (`Money`/`Accrual`/`Cashflow`). `FixedLeg`/`FloatLeg`
  encoding still inlined in OIS (one serialising consumer); trigger detached from swap per CP-3 §4.4 —
  lifts at the next serialising leg consumer.
- `fields` / `provenance` green; no new exemptions.
- The two judgment calls (bond tick, inflation hold) were surfaced to the user, not taken silently.

---

## 6. Ready-for-next / named next checkpoint

Pending Cowork rulings on §4a/§4b:
- **If swap ticks:** `fixed_income/` vanilla + core cluster is essentially retired (8/768); the natural
  next cluster is either **multi-curve/projection** (unlocks the deferred swap/FRA/OIS multi-curve pv
  and a batch of un-crossed consumers) or **breadth resumes** (options/credit/structured).
- **If the multi-product rule lands:** apply it to `inflation.py` and the other bundled modules.

**Named next checkpoint — CP-5:** at the first of (a) the multi-curve cluster landed, (b) 6 slices
since CP-4, (c) a new cross-cutting abstraction introduced (immediate-stop trigger).

**Ask for Cowork:** rule §4a (swap tick), §4b (multi-product-module retire pattern), confirm §4c (bond
spot-check), and set the post-vanilla direction (multi-curve next, or breadth).
