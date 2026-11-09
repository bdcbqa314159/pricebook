# Tranche 27 — bilateral CVA + DVA: both parties can default

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.33.0`.
**Sources:** `CREPEY-2026` (the bilateral frame and the identity `FV = CVA − DVA`, Prop 1 eq (18)) +
the library's own T25/T26 pieces. **Pre-registered RED, as always.** Third unit of the XVA spine arc
(`docs/xva_spine_arc.md`); follows T26.

## 0. The product worth having
**The bilateral credit adjustment of an uncollateralized netting set** — the step from "only the
counterparty defaults" (T25/T26, unilateral) to "both parties can default." Two legs:

- **CVA** (our loss if the **counterparty** defaults while we are **in the money**): the positive
  exposure against the counterparty's default — what T25/T26 built.
- **DVA** (the mirror: the value to us of our **own** default while we are **out of the money** — a
  benefit to our shareholders, a cost to our creditor): the **negative** exposure against *our own*
  default.

The bilateral fair-value adjustment is **BCVA = CVA − DVA** (Crépey eq 18, `FV = CA − CL = CVA − DVA`).
This makes the adjustment reflect that a defaulting bank does not pay what it owes — the DVA the
unilateral spine omitted.

## 1. The pieces and their layers
- **L5 `risk/` — the DVA functional, and the rule-of-two generalization of `unilateral_cva`.** CVA and
  DVA **share a shape** (§6): `LGD · ∫ DF · (exposure side) · (surviving-party survival) · dPD_(defaulting
  party)`. T25 was the first consumer; DVA is the **second**, so this is where the shared integral is
  extracted (§6 — generalisation decided at the second real consumer, as `simulate_forward_paths` was
  at T18). The extracted `credit_adjustment(side, defaulting_survival, surviving_survival, discount,
  lgd)` has:
  - **CVA** = `credit_adjustment(EPE, S_counterparty, S_own, …)` — counterparty defaults, we survive;
  - **DVA** = `credit_adjustment(ENE, S_own, S_counterparty, …)` — we default, counterparty survives;
  - **the unilateral `unilateral_cva` of T25 is the special case `surviving_survival ≡ 1`** (nobody
    else can default) — so T25's function becomes a thin caller, and the generalization **adds** the
    first-to-default conditioning the unilateral case dropped.
- **`ENE` is already built** (T26's `ExposureProfile` carries `ENE(t) = mean(V(t)⁻)`), and the T25
  survival curve (L1) is reused — now **two** curves, the counterparty's and our own. No new L1.
- **`BCVA = CVA − DVA`** as the headline, carrying its SE when the exposure is MC (invariant 7). The
  **pathwise estimator (T26, ratified) extends directly and is required here**: each path contributes
  `(∫DF·V⁺·S_own·dPD_C − ∫DF·V⁻·S_C·dPD_own)`, so the SE of the *difference* is the pathwise std — which
  correctly accounts for CVA and DVA being computed on the **same** paths (V⁺ and V⁻ are negatively
  related, so the difference's variance is **not** the sum of the two SEs; a reconstructed SE would get
  this wrong).

## 2. The oracle (RULED)
- **Reduction to the prior tranche (exact, the preferred anchor §4).** Set our own hazard `λ_own → 0`
  ⇒ `S_own ≡ 1` ⇒ DVA → 0 and the CVA leg → **exactly** T25/T26's unilateral CVA; BCVA → unilateral
  CVA. Symmetrically `λ_counterparty → 0` ⇒ CVA → 0, BCVA → −DVA. The unilateral case is the
  `surviving_survival ≡ 1` special case, so this reduction is also the **rule-of-two purity proof**:
  T25's `unilateral_cva` through the extracted integral must stay bit-identical (or, if the refactor
  reorders ops, an exact-arithmetic identity to a few ULP, §4 — report which).
- **The party-swap symmetry — the independent, non-telescoping anchor.** **Our DVA = the
  counterparty's CVA against us**: the same netting set viewed from the other side, with exposure sign
  flipped (`V → −V`, so our ENE is their EPE) and the two parties' curves swapped, must reproduce our
  DVA from a *different* composition of the pieces. A cross-product oracle (different code each side)
  that catches a shared sign or survival-assignment error no within-DVA check can — the T3/T5/T17
  pattern, and the correctness anchor beside the (telescoping) reduction.
- **Requirement oracles (Crépey eq 18 + the finance).** `BCVA = CVA − DVA` exactly (the identity, true
  for all parameters). `CVA ≥ 0`, `DVA ≥ 0`, `BCVA` either sign. Monotonicity: CVA ↑ in `λ_counterparty`
  and `LGD_C`; DVA ↑ in `λ_own` and `LGD_own`. The first-to-default conditioning is **present and
  correctly signed**: assert CVA strictly **decreases** when `λ_own` rises from 0 (our possible earlier
  default removes counterparty-default loss) — the term the unilateral case dropped, made visible.
- **Degenerate-fixture guards (mandatory, §4).** The exposure must be genuinely **two-sided** — both
  EPE and ENE materially non-zero — else DVA ≈ 0 and the whole bilateral character is invisible; assert
  it. The two hazards must **differ** (`λ_own ≠ λ_counterparty`) and the two LGDs differ, else CVA and
  DVA read the same survival and the party-swap and assignment are invisible. **Mutation table**
  (report it): swap `S_own`/`S_counterparty` between CVA and DVA; drop the surviving-party survival
  factor (→ two unlinked unilaterals, the first-to-default term gone); flip the `CVA − DVA` sign; feed
  DVA the EPE instead of ENE. Each must die.

## 3. Process
Branch `tranche/27-bilateral-cva-dva`. `test` (RED: the reduction + the party-swap symmetry + the
mutation table, pre-registered) → `feat` → `docs` → `chore(release): v0.33.0`. **Open the draft PR
early** (T26b's CI note). `source:` `CREPEY-2026`. The new/edited `risk/` modules keep their
`crosscheck:` markers (QuantLib has bilateral CVA/DVA, or `none — <reason>`, §8). `references` gate
stays green — no new source key. **Raise as an `Ask:`** (do not rule): if Crépey's eq-18 identity is
stated for a *marginal* (un-conditioned) CVA/DVA rather than the first-to-default-conditioned ones, so
that `BCVA = CVA − DVA` and the first-to-default factor cannot both hold exactly — then which the
tranche asserts as the identity and which as the modelling refinement is a design call, and mine.

## 4. What this tranche is NOT (scope fence)
No FVA (T28). No collateral/CSA (T29 or deferred). No **wrong-way risk** — exposure/default
independence still holds, so each leg's exposure and survival factorise and the joint survival is the
product `S_own·S_counterparty`; WWR (dependence between exposure and default) is what independence
drops, deferred. No KVA/BSDE/ML (`out_of_method`). The amortising exposure hump stays deferred
(`to_go_deeper`, T26 route 1) — T27 reuses whatever exposure profile it is handed.

---

## 5. Cowork's ruling on T27 — PR #38 (§10 step 4, 2026-11-09)

Accepted — v0.33.0, 935 green, nine gates, eight mutants killed, draft PR opened early.

**The convention — RULED: keep the explicit-factor, market-standard form (built). Crépey's
`(1−γ)⁻¹` normalisation → `to_go_deeper`.** You established there is no exactness conflict — eq 18's
`FV = CVA − DVA` and the first-to-default conditioning coexist in both conventions, so our requirement
oracle holds either way — and the choice is a *level*: the first-to-default term is 2.679% of the
unilateral CVA (built), Crépey's `(1−γ)⁻¹` a further 4.707%. The elementary spine stays **standard
risk-neutral**, for three reasons:

1. **Consistency.** T25/T26 are standard and do not normalise; the explicit factor is their coherent
   extension. Grafting `(1−γ)⁻¹` onto an otherwise risk-neutral framework mixes measures.
2. **Scope.** The `(1−γ)⁻¹` normalisation belongs to Crépey's **cost-of-capital / fininsurance**
   apparatus — the advanced frontier this arc deferred (`out_of_method`). Importing the normalisation
   without the framework is half a measure. Crépey is our *definitional frame* (the identities as
   requirement oracles, which you confirmed hold in both conventions); he is not the measure the rest
   of the tree prices in — that is the standard one C-Z and market practice use.
3. **Blast radius.** Adopting it would retroactively move every CVA number T25/T26 produced, for a
   framework we are not building. Not an arc-wide change made for a deferred apparatus.

So the built form stands, and this is honest rather than defaulted: both levels are *measured* and the
alternative is **recorded** (`to_go_deeper`, CREPEY route 2) so the 4.707% is not lost — it is pulled
when/if the cost-of-capital framework ever is.

**The bit-identical purity via `surviving=None`-skips-the-multiply — ratified, and the design instinct
is right.** A unit survival curve would have reordered nothing but *performed* an extra multiply,
leaving only the ULP argument; skipping the op keeps bit-identity. §4's "take the stronger proof where
available", with availability made by design. Good.

**Noted, not a finding: the `-layout` reflow misread eq 18's number to the *reader of our own spec*
(you), and the dual rendering corrected it.** The §5 "our docs are not an authority" hazard, one level
further out — the spec's citation was right, and the discipline that protects against a bad source
also protects a reader second-guessing a good one. Worth a journal line, not a rule (the rule exists).

**Next:** T28 (FVA — the funding valuation adjustment, Crépey's semilinear fixed point `FVA =
γ(P−CVA−FVA)⁺`, eq 19) is mine to spec (§10 step 1), on disk before you build.
