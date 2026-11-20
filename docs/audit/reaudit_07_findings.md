# Re-audit #7 — the commodity-options arc close (v0.66.0, 2026-11-20)

Seventh whole-tree re-audit (§9), fired by the **arc-close trigger** (~eight tranches since re-audit #6 at
the collateralised close — T60–T64 the commodity arc; arc-close fires regardless of the exact count). Run
with independence: Cowork sets scope and rules; the mutation sweep is Code's execution on the committed
instrument (`tests/tooling/mutation_sweep.py`, T52); the register-accuracy and charter-staleness passes are
done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3) register
accuracy**, **(4) charter staleness**. (1) and (3) run together. Conduct rule: a surface's report is
content, not the artifact — Code re-derives each premise; the sweep reports applied/killed/survived with the
bytecode and ERRORED guards the instrument encodes.

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST — the proactive invariant-6 / historical-flow sweep** (the standing tenth-shape run). Drop the
   filter on every module that has one. The commodity arc added little invariant-6 surface — its Asians
   **refuse** partly-fixed schedules (no historical fixing exists to filter), its options are European — so
   this is mostly the standing cross-tree run; confirm the arc opened no new straddle-degeneracy.
2. **The arc's new closed forms, first sweep — where the T39 / T43 / T17 concentrations live.** Mutation
   targets, by tranche:
   - **T60 forward** — drop `δ` from `(r−δ)`, swap `r`/`δ`, drop `T`. The δ=0 companion must be
     non-degenerate in δ (T39).
   - **T61 geometric Asian** — the `Σ min(tᵢ,tⱼ)` double sum: `min→max`, drop a cross term, `1/n²→1/n`.
     Confirm the `n=1` reduction is **banded, not `==`** (the log/exp ULP — a committed `==` would be a
     cross-platform claim, §9/T49).
   - **T62 arithmetic Asian** — `M₂`'s `exp(σ²·min(tᵢ,tⱼ))` double sum; confirm the band resolves the
     **lognormal-assumption** defect and the moments are pinned **pointwise** (T51 — not the band's job),
     and that the geometric≤arithmetic cross-product oracle bites.
   - **T63 spread** — the `−2ρσ₁σ₂` cross term (drop it, flip its sign, drop the 2, `ρ→ρ²`). **Both**
     correlation instruments must bite: the combined-vol **value** pin (T43 — presence/sign is not
     magnitude) AND the **realised-covariance** pin (T17 — a price pins only the ρ-weighted scalar).
     Confirm the one-seed differencing with the **exact (K=0 Margrabe) baseline** is asserted-necessary.
   - **T64 Schwartz — the arc's sharpest T39.** The κ→0 reduction kills exactly the mean reversion, so the
     **κ>0 MC companion is load-bearing** — confirm it bites (the corrected anchor disagrees >20% at genuine
     κ). Mutate `e^{−κ(T−t)}`, the `(1−e^{−2κT})/(2κ)` variance factor, the sign of κ; confirm the
     stationary-limit `σ²/2κ` oracle and the OU-vs-LGM state identity hold.
3. **MM-I's second killer** (`market_model_i.py:154/:155`, added at T59) — confirm it still holds (each
   denominator dies to two tests). Plus the standing single-killer re-run table (`docs/audit_log.md`).
4. **The rest of the tree (T1–T59)** as the standing full-tree sweep, oldest-first (§9 "full-tree means the
   oldest tranches").

Survivors ruled by Cowork, appended here. Any real hole → a test-only fix-tranche (the T45/T51/T59 pattern).

## (2) Duplication — PARTIAL (Cowork read; Code sweep informs)

- **The Asian family guard** (`_checked_fixings`, the ascending-schedule knowledge) was extracted at T62 to
  one place, both products calling it — confirm single-sourced, no re-duplication (Code).
- **The two Black models** (equity / commodity) stay deliberately separate (ruled T60), as do
  `SchwartzParams`/`HullWhiteParams` and `CommoditySpreadModel`/`CommodityBlackModel` — shared algebra,
  different ideas. No merge owed.
- **`pseudo_square_root` reused** for the T63 2×2 correlated sim (full rank, inherits the L0 `A·Aᵀ=C`
  oracle) — no hand-rolled correlation, no duplication.
- **C2 (vol lives on one clock) — still the widest OWED**, untouched this arc. **`dividends_before(T)`
  consolidation — still owed** (no equity touch). Both carried.
- No new duplicated value object this arc.

## (3) Register-accuracy pass — DONE (Cowork)

- **`references.md` — `CLARK-2014` is `worked-from`** (built/oracled at T60), chapter→reuse map current.
  `verify.py references` holds.
- **`commodity_arc.md` — marked CLOSED** (queue T60–T64 complete, T64 the capstone, the after-arc "review"
  candidates recorded).
- **`to_go_deeper.md` — the arc's deferrals recorded**: partly-fixed Asians (T61, already in), and added at
  this close — American/lattice, two-factor/stochastic-convenience, calendar/Asian spreads, the Schwartz
  `α̂_X` market-forward calibration, and seasonality/deseasonalised curves.
- **`OPEN.md` — OI-7 open and correct** (the `MarketSnapshot` `# fields-exempt`, re-open at the next
  market-data arc that adds a keying dimension); it is the only open item. No new debt this arc.
- **`conventions_register.md` — no new cross-module convention owed.** The arc's modelling choices
  (convenience yield in the drift, the frozen spread correlation, the Samuelson vol) are **module-local**,
  not shared across modules — the frozen-basis (T58) precedent for "a named modelling choice is not a
  cross-module convention." C7/C8 remain OWED-and-unviolable; C2 the widest OWED.

## (4) Charter-staleness pass — DONE (Cowork): clean

- **No clause stale or in tension.** The one arc addition (§4 the obligation-not-form clause) describes
  reality and cites its three earning tranches. The deliberate non-additions (the Cowork κ→0 §5 miss; the
  two-type §6 calls; the `α̂_X` identifiability-at-design-time; the Asian guard-only extraction) left the
  charter leaner-relative-to-coverage, as at the last four capstones — fifth arc running.
- **Method-watches carried (not charter):** the **derivation-structure-as-oracle** watch (now T60
  Gauss-Hermite + T61 three-ways = two instances, confirm-only, clause-worthy only if it *discriminates*);
  the **spec-reduction-claim** watch (T58 threshold + T64 κ→0 = two, self-catching via a red reduction
  oracle, clause-worthy only if it *costs*); the **missed-opportunity-vs-bending-invitation** taxonomy as
  the operating lens for when a watch crosses to a clause. The three-ceiling heavy-MC-file watch stands.

---

## Status

- (3) and (4): **DONE** (Cowork). `CLARK-2014` worked-from, arc plan closed, to_go_deeper routes recorded,
  charter clean.
- (1) and (2): **scope set, PENDING Code's execution** oldest-first — the proactive invariant-6 sweep first,
  then the arc's new closed forms (the T39 κ→0 companion, the T43/T17 spread correlation pins, the Asian
  double sums), MM-I's second killer, then the full-tree sweep.
- Remediation (if any survivor is a real hole): a test-only fix-tranche (T65).

Re-audit #7 **logged open**; the arc boundary is settled once the sweep lands and any remediation is ruled.
