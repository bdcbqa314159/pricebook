# The collateralised-discounting arc — project seven (`LICHTERS-2015`)

**Proposed by Cowork, committed by Bernardo 2026-11-18 ("the biggest spine we could do — balance with
`to_go_deeper`").** The seventh vertical: the **CSA / collateralised-discounting layer** the tree has never
had — the post-crisis reality that a collateralised trade discounts at the *collateral rate*, not a single
risk-free curve — and its extension into collateralised credit exposure. The spine is "big" (it spans a new
discounting layer, a new short-rate model, and a collateralised-exposure extension) but **not huge**,
because three of its major pieces are already held and reused, not rebuilt.

## The thesis — a layer UNDER the tree, reusing three arcs
Every price in the library today discounts on **one curve per currency** — the pre-crisis world. Lichters'
subject is that a collateralised trade discounts at the **collateral (OIS) rate**, with separate
forward-projection curves — and that this changes the discounting *under* everything. It is the most
foundational thing we have not built, and it is **additive, not a rewrite**: a new curve-set and a
collateral-aware pricing mode, with today's single-curve as the **degenerate case** (perfect cash-CSA with
OIS = projection recovers the old discounting — a clean reduction anchor). The reuse is deep enough that the
"biggest spine" is affordable:

| Lichters | we hold | the arc reuses |
|---|---|---|
| ch 1–2 pre-crisis discounting, curve building | our single-curve bootstrap (T1/T8) | the bootstrap machinery |
| **ch 3–5 OIS/multi-curve, CSA discounting** | — | the **new core** |
| Part II CVA/DVA (ch 7–9) | the XVA arc (T25–T29) | reused + cross-product oracle, **not rebuilt** |
| **ch 11 LGM short-rate** | Hull-White (T4/T10a) | **LGM = HW reparameterised — exact reduction** |
| ch 13.2 Jarrow–Yildirim | our JY (inflation arc) | reused for inflation RFE if pulled |

Method-fit is strong: the OIS bootstrap is **T8's par-reprice round-trip** on a new curve-set; CSA
discounting carries a **pre-crisis reduction** (OIS=Libor → the single-curve value); the LGM has an **exact
reduction to Hull-White**; collateralised exposure **reduces to the uncollateralised CVA** (threshold → ∞).
Reduction and requirement oracles everywhere, as the method likes.

## Design invariant — single-curve stays FIRST-CLASS, not a degenerate accident (Bernardo, 2026-11-18)
**Single-curve and multi-curve are co-equal, deliberately-selectable pricing modes for the whole arc.** This
is a ruled invariant, because Bernardo wants to run **parallel projects that show the two eras side by
side** — the pre-crisis single-curve world (the `MCINERNEY-ZASTAWNIAK` stochastic-rates / Hull-White setting
we already built) alongside the post-Libor multi-curve world Lichters describes. So:
- **No tranche may make single-curve reachable only as the `OIS = projection` degenerate case.** An engine
  discounts off a *source* that is **either** a single discount curve **or** a `CurveSet`'s OIS curve; both
  are valid inputs a `MarketSnapshot` (§3) can carry, and the choice of era is the choice of what the
  snapshot holds — explicit, not an accident. Multi-curve is a **sibling** of single-curve, never a
  replacement; the existing single-curve path is preserved, not deprecated.
- **The single-curve path is kept exercised so it cannot atrophy.** If every new fixture is multi-curve, the
  single-curve path goes untested and can silently break — the **T33b input-population degeneracy** (a mode
  whose branch is never in the test population). So every arc tranche that touches discounting keeps at
  least one oracle pricing in **genuine single-curve mode** (the existing path, not `OIS=projection`), and
  asserts the two modes **agree where they coincide** and **differ where a real basis exists** — which is
  both the parallel-era guarantee and the degenerate-fixture premise.
- **The reduction oracles prove the eras coincide at the boundary; this invariant guarantees both are
  reachable.** They are different claims: a reduction says "multi-curve → single-curve at OIS=projection";
  the invariant says "single-curve is a mode you can select and build a project on." The arc owes both.

## The spine (phased; each unit ruled at its own spec)

### Phase 1 — the collateralised-discounting layer (Part I, the core new objects)
- **T54 — the OIS / multi-curve bootstrap (ch 3).** Build the **OIS discount curve** and the separate
  **forward-projection curves** from OIS swaps + tenor/basis swaps. **L1** (a curve bootstrap — market
  quotes → curves, §2). **Oracle:** the **T8 known-curve round trip** (bootstrap, reprice the calibrating
  instruments to par 0); the **single-curve reduction** (OIS = projection ⇒ today's single-curve values,
  exact). Reuses T8's bootstrap discipline whole.
- **T55 — CSA discounting (ch 5).** A collateralised trade's PV discounts at the collateral rate; **perfect
  cash-CSA = OIS discounting**. The collateralised swap/linear instrument reprices under OIS. **Oracle:**
  the **pre-crisis reduction** (OIS=Libor ⇒ the old single-curve value); a collateralised swap reprices at
  OIS to par; the self-consistency guard (telescoping) kept secondary to the reduction. Reuses T54's OIS
  curve + the existing linear products.
- **T56 — global / cross-currency collateralisation (ch 4). Provisional — folds or defers at its spec.**
  Collateral posted in a foreign currency; the **cross-currency basis**. **Oracle:** **covered interest
  parity** (the FX-forward identity) and the collateral-choice option ≥ 0 (cheapest-to-deliver is an
  option). If it pulls a materially new object (a cross-currency curve set) it is built; if it reduces to
  T55 + the FX forward with nothing new, the §1 pull gate folds it to `to_go_deeper`. Ruled at its spec.

### Phase 2 — collateralised exposure (reusing the XVA arc)
- **T57 — the Linear Gauss-Markov (LGM) short-rate model (ch 11).** The LGM model for risk-factor
  evolution. **L3.** **Oracle:** the **exact reduction to Hull-White** (LGM's `(H, ζ)` is a
  reparameterisation of HW's `(a, σ)` — exact by construction, the strongest anchor); swaption repricing
  (calibration). Reuses Hull-White whole; the companion exercises the non-degenerate LGM parameterisation
  (T39 — a reduction a HW no-op also satisfies is vacuous).
- **T58 — collateralised exposure / CVA under a CSA (Part II + the collateralised EPE).** Extend the
  T25–T29 exposure/CVA machinery with collateralisation: the **collateralised exposure** (thresholds, MTA,
  the **margin period of risk**), the CSA-aware EPE feeding CVA. **L5.** **Oracle:** the **reduction to the
  uncollateralised CVA** (threshold → ∞, no collateral ⇒ the T25 CVA, exact); **collateralised exposure ≤
  uncollateralised** (collateral reduces exposure — a requirement for all parameters); **cross-product vs
  the T25 CVA integral** (different code each side). Reuses the XVA arc + T55's CSA discounting + T57's LGM.
- **Checkpoint** `CP_collateralised_discounting.md` closes the arc; its close fires **re-audit #6** (§9 —
  also the eleven-tranche cadence lands near here, with MM-I's second killer already queued for it).

## The honest completion ratio (proposed)
Built as the spine: the OIS/multi-curve bootstrap, CSA discounting, (cross-currency, provisional), the LGM
short rate, and collateralised exposure/CVA. Deferred to `to_go_deeper` (oracle-able, unpulled):
**fair-value hedge accounting** (ch 6 — arguably §0 out-of-scope, accounting not pricing), the **FX / equity
/ commodity RFE** models (ch 12, 14), the **Dodgson–Kainth inflation** model (ch 13.3), **seasonality**
(13.4), **initial margin / MVA** simulation (ch 5.6), and the **multi-currency LGM** beyond the single-
currency IR spine (ch 12.2). **`out_of_method` candidate:** **exposure-model backtesting** (the book's
backtesting part + the held `kerkhof.pdf`) — statistical/historical validation anchored by backtest
statistics and real data, with no closed form, cross-check or requirement-identity to pin it; recorded as a
candidate for the §4 statistical-oracle extension, ruled only if interest ever pulls there, never built
under §4 as it stands.

## Decision — the reuse-heavy full spine is committed (Bernardo, 2026-11-18)
Bernardo chose the **biggest spine affordable given the reuse** — the collateralised-discounting layer and
its exposure extension, with CVA/DVA, Hull-White and JY reused rather than rebuilt, and the RFE zoo /
backtesting balanced into `to_go_deeper` / `out_of_method`. **The queue:** T54 (OIS/multi-curve bootstrap) →
T55 (CSA discounting) → T56 (cross-currency, provisional) → T57 (LGM short rate) → T58 (collateralised
exposure/CVA) → `CP_collateralised_discounting.md` (closes the arc, fires re-audit #6). Cowork writes each
per-tranche spec before its build (§10); this is the queue, not the specs. **T54 is specced**
(`docs/tranches/54_ois_multicurve_bootstrap.md`); the rest follow in turn. `LICHTERS-2015` is registered
`in-hand` (flips to `worked-from` at T54).
