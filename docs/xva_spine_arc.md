# The XVA spine arc — the elementary valuation-adjustment vertical

**Decided by Cowork with Bernardo, 2026-11-08.** Project three, after the Jäckel arc. Bernardo brought
`CREPEY-2026` (*XVA Analysis*) and picked the **elementary XVA spine** over the book's own frontier,
knowing the source is a stretch for our method. The arc is scoped around that honesty — the same shape
as BERHAN (ch 1–5, smile deferred) and Jäckel (9 of 14 chapters, trees/ML deferred): build the
**oracle-able static/one-period core**, ledger the rest as `to_go_deeper`, close with a checkpoint.

This vertical is the one the archived predecessor **faked** — the phantom `risk/exposure.py`,
`risk/xva.py`, `risk/wwr.py` rows that re-audit #1 found in the conventions register and T24 is
pruning. We replace fiction with code oracled against a held source.

## The thesis, and why two sources
XVA is a correction applied *on top of* a risk-free price: value the netting set, then adjust for the
counterparty defaulting (CVA), yourself defaulting (DVA), the cost of funding (FVA), the cost of
capital (KVA). The elementary, tractable core is **unilateral CVA = LGD · ∫₀ᵀ DF(t)·EPE(t)·dPD(t)**
under independence of exposure and default, and its siblings.

The method-fit problem (§4, and the reason for the honest scope): **Crépey does not print these
elementary closed forms.** His CVA/FVA/KVA are abstract balance-sheet expectations and semilinear
fixed-point equations — `CVA = E[(1−J)P⁺]`, `FVA = γ(P−CVA−FVA)⁺`, `FV = CVA−DVA`,
`FTP = ΔCVA+ΔFVA+ΔKVA` — and his numerics are neural-net regression (ch IV–VII), validated by twin
Monte Carlo, not a closed form. So the arc **splits the two sources along §4's own seam**:

- **`CREPEY-2026` supplies the identities → requirement oracles.** Relations true for all parameters
  by the finance: `CVA=0` when there is no default or no loss; `FV = CVA − DVA`; the FTP
  decomposition; the FVA fixed point. These *survive a transcription error* (§4) and are what the
  spine is anchored on at the structural level.
- **`CAPINSKI-ZASTAWNIAK` supplies the concrete closed forms → pointwise transcription.** The survival
  curve `S(t)=exp(−∫λ)` (ch 2), the defaultable bond and its calibration (ch 3), the CDS par spread
  (ch 4), Merton structural default probability (ch 1). These are transcribed pointwise in two
  renderings, per §4.

Neither source replaces the other — exactly §4's "requirement oracle + pointwise transcription."

## The layered picture
- **L1 `market/`** — the **survival / default-probability curve**, a new market-data primitive beside
  the discount curve: `S(t)=exp(−∫₀ᵗ λ)`, built from a hazard rate (flat, then piecewise-constant),
  calibratable to CDS/defaultable-bond quotes (C-Z ch 2–4). Pulled by CVA, not built speculatively.
- **L5 `risk/`** — the **expected-exposure profile** `EPE(t)=E[V(t)⁺]`, `ENE(t)=E[V(t)⁻]`, and the
  **valuation-adjustment functionals** (CVA, DVA, FVA) over an exposure profile + a survival curve.
  L5 already exists (Greeks, T22); XVA is its second, larger consumer.
- No new L2 product type is forced at first: the first unit defines the CVA *functional* on an
  exposure profile, with a closed-form analytic exposure as the anchoring input; the engine-sourced
  exposure (through `price()`) arrives at the MC-exposure tranche.

## The arc — the tranche sequence (the queue; Cowork writes each spec before its build)
- **T25 — unilateral CVA + the survival curve.** The first product worth having: unilateral CVA of an
  uncollateralized single-instrument netting set, under exposure/default independence, with a
  **closed-form expected-exposure anchor** (a Gaussian/Bachelier exposure, EPE in closed form) against
  a hazard-rate survival curve. Pure closed-form oracle, no MC — honoring §4's "closed form > MC."
  Spec: `docs/tranches/25_unilateral_cva.md`.
- **T26 — the Monte-Carlo exposure engine.** EPE/ENE by simulation over a real instrument through
  `price()` (reusing the Jäckel MC machinery), **cross-checked against T25's closed-form EE** on the
  analytic case (the reduction + cross-product oracle), then the CVA of an uncollateralized swap. This
  is where "L5 on the engine + a `Priceable` protocol" (§2) materializes.
- **T27 — bilateral CVA + DVA.** Both parties default; oracled by Crépey's `FV = CVA − DVA` identity
  and the party-swap symmetry (DVA of A = CVA seen by B), plus first-to-default survival.
- **T28 — FVA.** The funding adjustment as Crépey's semilinear fixed point `FVA = γ(P−CVA−FVA)⁺`
  (eq 19); oracled by fixed-point convergence and the reduction `FVA=0` at zero funding spread.
- **(conditional) T29 — collateral / CSA (VM).** Variation margin and the collateralized exposure, if
  interest pulls it before the checkpoint; else deferred.
- **Checkpoint `CP_xva_spine.md`** closes the arc, as `CP_jaeckel_arc` closed the last.

## The completion ratio — honest
At arc end, of Crépey's twelve chapters we will have built the **static/one-period elementary spine**:
the survival/credit curve, expected exposure, and unilateral + bilateral CVA/DVA and FVA — oracled
against `CAPINSKI-ZASTAWNIAK`'s closed forms and `CREPEY-2026`'s identities. Deferred wholesale to
`docs/to_go_deeper.md` under `CREPEY-2026`: the neural-net/regression chapters (IV–VII), KVA and the
anticipated-BSDE machinery (II, VII), the risk-measure / equilibrium / reverse-stress-test chapters
(VIII–X), and HVA (XI). The cost-of-capital KVA — the book's actual thesis — is explicitly *not* in
the spine; it is the richest deferred route, the "part two."

## The honesty caveat (as for BERHAN and Jäckel)
Crépey's results are figures (and neural-net convergence diagnostics), not tables. No number is read
off a plot (§4): every spine number comes from a C-Z closed form, a Crépey identity, or the library's
own verified pieces, each with its oracle.
