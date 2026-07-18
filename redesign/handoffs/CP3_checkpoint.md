# Checkpoint CP-3 — serialisation cluster (first honest drawdown: 0 → 5/768)

**Cluster:** CP-3, the *cross-cutting-to-retire* pivot. Goal was **drawdown > 0** by building the
serialisation capability that the retire-reads kept surfacing as the production-reachable residual.
**Result: drawdown 0 → 5/768** across five oracle-gated retire slices.

**Versions:** v0.47.0 → v0.51.0. **Tests:** 276 ng green (full `tests_ng` sweep). **Gates:**
`acyclic / debt / version / provenance / fields` all green. **Branch discipline:** one `slice/*` per
retire, RED→GREEN→release→docs; two governance docs (`docs/ratify-*`) merged mid-cluster.

---

## 1. Slices landed (CP-3 cluster)

| # | slice | version | quarry module RETIRED | genuine residual closed | drawdown |
|---|---|---|---|---|---|
| 1 | serialisation-numerical-config | 0.47.0 | `core/numerical_config` | `to_dict/from_dict` + `replace` | 0 → 1 |
| 2 | serialisation-deposit | 0.48.0 | `fixed_income/deposit` | product `to_dict/from_dict` | 1 → 2 |
| 3 | serialisation-fra | 0.49.0 | `fixed_income/fra` | `to_dict/from_dict` (+ lift `Money`) | 2 → 3 |
| 4 | serialisation-ois | 0.50.0 | `fixed_income/ois` | `to_dict/from_dict` (+ lift `Accrual`, `Cashflow`) | 3 → 4 |
| 5 | serialisation-zcb | 0.51.0 | `fixed_income/zero_coupon_bond` | `FixedCashflow.to_dict/from_dict` | 4 → 5 |

Mid-cluster governance (Cowork-ratified, merged): `rulings_spotcheck_retire_1.md` (retire #1 tick
confirmed + evidence protocol); `rulings_CP3_correction.md` (serialisation through-line ratified,
**§4 phantom-residual rule** added).

**Serialisation encoder stack built (rule of two, bottom-up):** `Money` (deposit + FRA) → `Accrual`
(FRA + coupon cashflows) → `Cashflow` (deposit + OIS legs), all shared in `foundation/`. `FixedLeg`/
`FloatLeg` encoding stays **inlined in OIS** (its only serialising consumer) — lifts when the vanilla
swap serialises. No serialisation framework; per-class `to_dict/from_dict` with a `schema_version`
(absent = legacy v1, newer = loud reject).

---

## 2. Oracle-quality audit

Every retire closed on the **same oracle shape**: a `to_dict → from_dict` **round-trip equality** on a
representative instance, plus three guards — `schema_version` present, absent-reads-as-v1,
future-version-rejected-loudly. This is a **self-consistency oracle** (round-trip identity), which is
below closed-form on the hierarchy — *but* it is the correct and complete oracle for the capability
being added: serialisation's contract **is** round-trip fidelity + version safety. The instruments'
*pricing* oracles (closed-form par→0, closed-form off-par, statelessness) were already green from the
originating slices and are untouched — serialisation adds no pricing claim.

- **No tolerance issues:** round-trip is exact equality on frozen dataclasses (no transcendental
  compare), so the cross-OS ULP rule (§7c) does not bite.
- **Refactors guarded:** the `Money`/`Accrual`/`Cashflow` lifts refactored already-green producers
  (deposit, FRA) under their existing round-trip oracles (§8 pure-refactor carve-out) — full L4+L0
  re-run green after each.

**Honest limit:** these are self-consistency oracles. They prove ng round-trips *itself*, not that ng's
wire format matches the quarry's. That is intentional — ng is a copy-ADAPT supersede, not a
byte-format clone (the quarry's `_serialisable` envelope is deliberately not carried forward), so a
cross-format oracle would be testing a compatibility we explicitly do not owe.

---

## 3. Quarry-drawdown reconciliation — **5 / 768 (0.65%)**

Retired: `core/numerical_config`, `fixed_income/{deposit, fra, ois, zero_coupon_bond}`. Living map:
`quarry_reconciliation.md` (RETIRED section carries per-module consumer-evidence + shed lists).

**The cluster's central finding (already ratified as CLAUDE.md §4):** the reconciliation map's parity
gaps were **feature-diffed** (quarry-has vs ng-has) and are **systematically overstated**. Under
*consumer analysis*, residuals shrank or vanished:
- **deposit** — the "conventions" residual was a **phantom**: the quarry `Deposit` class has **zero
  production instantiations**. Real residual = serialisation only. No conventions/RateIndex built.
- **fra / ois** — one-or-two production instantiations, all in the un-crossed `desks/` layer
  (`deferred→desks/api`, multi-curve) — not owed now.
- **zcb** — no external production instantiation; its money-market analytics are `dead` duplicates
  (quarry `tbill.py` is their real home); `from_convention` `deferred→sovereign_bonds`.

**Consequence:** drawdown should keep moving **faster than the map's gap lists suggest**. Re-derive
each module's residual by consumer analysis at pickup; do not plan from the feature-diffed gaps.

**Shed-evidence ledger (this cluster):**
- `dead` (test-only / no-consumer): numerical_config's 4 truly-unconsumed knobs; deposit/fra/ois
  `from_convention`; zcb's yield analytics surface.
- `deferred→X` (real un-crossed consumer, forward-linked): numerical_config's 8 toolkit knobs
  (→ `numerical/_fourier/_pde/_trees/_integrate/_rootfinding`); zcb `from_convention`
  (→ `sovereign_bonds`).

---

## 4. Design choices to challenge

1. **Retiring `zero_coupon_bond.py` (a ~200-line module) against `FixedCashflow` (a ~45-line product).**
   Justified: ZCB's *live* role is `Face·DF(T)` + serialise, which FixedCashflow covers; the rest is
   `dead` duplicates of `tbill.py` or `deferred→sovereign_bonds`. Is "a ZCB *is* a FixedCashflow" the
   modelling we want long-term, or should ng eventually carry a named `ZeroCouponBond` (even if thin)
   for legibility/provenance? Current call: no — a named wrapper with no distinct behaviour is the
   speculative-abstraction smell (§6b).
2. **Self-consistency oracle for the whole cluster.** Acceptable for serialisation (round-trip *is* the
   contract), but it is the weakest oracle tier. Flagging per cadence ("any slice whose best oracle is
   only self-consistency" is a checkpoint trigger — this is that flag, cluster-wide).
3. **`schema_version` per class, no central registry.** Each type owns its version integer. Legible and
   local, but there is no global "wire schema version." Fine while there is no cross-type migration;
   revisit if/when a persisted-portfolio format needs one envelope version.
4. **Leg encoding inlined in OIS.** Deliberate rule-of-two hold (one serialising consumer). The bet is
   that the vanilla swap serialises "soon enough" to lift it — but swap is load-bearing (29 prod
   instantiations) and its retire is *not* serialisation-only, so the lift may wait. Acceptable?
5. **`db.py from_dict` dispatcher as the residual's justification.** The whole cluster treats "the DB
   dispatcher reconstructs it" as making serialisation a production-reachable residual, even though
   ng's persistence/data-spine layer is **not yet crossed**. Is serialisation genuinely owed now, or is
   it `deferred→persistence`? Cowork's #1 ruling said build it; this cluster followed that. Worth an
   explicit re-affirmation that we keep building serialisation ahead of the persistence consumer.

---

## 5. Smell + debt scan

- **Debt:** `verify.py debt` green — **zero ng suppressions, zero `# type: ignore`, zero skipped
  tests, zero load-bearing TODOs** in the CP-3 code. (OPEN.md's serialisation-typing debt is **quarry-
  side**, a separate refactoring effort on the old tree — not ng.)
- **Duplication:** the intended rule-of-two lifts were taken (`Money`/`Accrual`/`Cashflow`). Remaining
  known duplication: FixedLeg/FloatLeg encoding inlined once in OIS (tracked, §4.4).
- **Signature/field hygiene:** `verify.py fields` green; no product exceeded ≤5 fields; no new
  `fields-exempt` markers.
- **No speculative infra:** the headline discipline win — *declining to build conventions/RateIndex*
  against a false premise (deposit) held §6b even when the ruling was wrong.
- **Provenance:** every touched module's four-line header updated (`verify.py provenance` green).

---

## 6. Ready-for-next / named next checkpoint

**Cluster boundary reached** (fixed-income vanillas + the config). Recommended continuation, all by
consumer-analysis retire-read (expect small residuals):

- **CP-3 tail:** `fixed_rate_bond`, `leg`, `inflation` — likely serialisation-only or already
  superseded. Each its own thin retire slice.
- **Then the hard one:** the **vanilla swap** — *not* a serialisation-only retire. 29 production
  instantiations (par-swap curve pillars, XVA exposure). Its real residual is the **multi-curve /
  curve-build role**; needs its own analysis (does ng's `ParSwapQuote` + bootstrap already supersede
  the curve-pillar use?). Candidate boundary for **CP-4 = multi-curve / projection**.

**Named next checkpoint — CP-4:** at the first of (a) the fixed-income vanilla cluster fully retired
(bond/leg/inflation + a swap decision), or (b) 6 slices since CP-3, or (c) the multi-curve abstraction
is introduced (immediate-stop trigger). 

**Ask for Cowork:** ratify the §4-driven expectation (gaps overstated → keep retiring by consumer
analysis, not the map), and rule on the five challenge items above — especially #5 (do we keep building
serialisation ahead of the persistence consumer?).
