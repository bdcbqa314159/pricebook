# Tranche 25 — unilateral CVA + the survival curve: the XVA spine opens

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.30.0`.
**Sources:** `CREPEY-2026` (the CVA definition and its identities — the requirement oracles) +
`CAPINSKI-ZASTAWNIAK` (the survival curve closed form — pointwise transcription). Both `in-hand`;
this tranche makes them `worked-from`. **Pre-registered RED, as always.** First of the XVA spine arc
(`docs/xva_spine_arc.md`); follows the re-audit remediation (T23 → T24). **Build T25 only after T24
lands.**

## 0. The product worth having
**Unilateral CVA of an uncollateralized, single-instrument netting set**, under independence of
exposure and default — the elementary, closed-form core of XVA:

> CVA = LGD · ∫₀ᵀ DF(t) · EPE(t) · dPD(t),  dPD(t) = −dS(t) = λ(t)S(t)dt

This is the irreducible first unit: it is the thing every other adjustment (DVA, FVA, the MC exposure
engine) will reduce to or build on. It replaces the archived predecessor's **faked** `risk/xva.py`
(the phantom register rows T24 is pruning) with code oracled against a held source.

Kept deliberately closed-form for the first unit — **no Monte Carlo** — honoring §4's "closed form >
external cross-check > …". The exposure is given analytically (a Gaussian/Bachelier mark, EPE in
closed form); the MC exposure engine over a real `price()` is T26, cross-checked *against* this
tranche's closed form.

## 1. The pieces and their layers
- **L1 `market/` — the survival / default-probability curve** (new market-data primitive, beside the
  discount curve). From a hazard rate: `S(t) = exp(−∫₀ᵗ λ(u)du)`; flat `λ` ⇒ `e^{−λt}`,
  piecewise-constant `λ` ⇒ `exp(−Σλᵢτᵢ)`. The default density is `dPD(t) = λ(t)S(t)dt`. Source:
  `CAPINSKI-ZASTAWNIAK` ch 2 (hazard function, default time, risk-neutral probability). **Pulled by
  CVA** — not built speculatively (§1 product-pull). Calibration to CDS/defaultable-bond quotes
  (C-Z ch 3–4) is *not* in this tranche; a constructed hazard curve suffices for the first consumer.
- **L5 `risk/` — the expected-exposure profile and the unilateral CVA functional.** `EPE(t)=E[V(t)⁺]`
  for an analytic exposure `V(t) ~ N(m(t), s(t)²)`, closed form `EPE(t)=m(t)·Φ(m(t)/s(t)) +
  s(t)·φ(m(t)/s(t))` (the Bachelier/call-on-exposure identity). Then the CVA functional above over an
  exposure profile + a survival curve + a discount curve. No new L2 product is forced: the analytic
  exposure is a **known-profile test fixture**; the CVA functional is the product. L5 already exists
  (Greeks, T22) — XVA is its second consumer.

## 2. The oracle (RULED)
The two sources split along §4's seam: **Crépey's identities are requirement oracles** (true for all
parameters by the finance, so they survive a transcription error); **C-Z's survival and the Bachelier
EPE are pointwise transcriptions** (independent, two renderings). Neither replaces the other.

- **Pointwise transcription, closed form (§4 top-of-order).**
  - `S(t)=exp(−∫λ)` pointwise; the density relation `dPD = λS dt` **derived independently** in the
    test, never read back from the curve code.
  - the EPE identity `E[(a+bZ)⁺] = a·Φ(a/b) + b·φ(a/b)` transcribed two independent ways and reconciled
    — the analytic form *and* a direct numerical integral `∫(a+bz)⁺φ(z)dz` (the test computes its own
    reference). Read the Φ/φ composition in `-raw` and `-layout` if taken from a source; it is standard
    enough to derive, which is stronger (§4 — a derived reference beats a transcribed one).
  - the CVA integral against an **independent quadrature** of the closed-form integrand at a fine grid;
    the test builds its own `DF·EPE·dPD` and integrates, never calling the CVA code for its reference.
- **Requirement oracles (Crépey Prop 1 eq (17), Ch I §3 p.37, + the finance).**
  - **Two separate zeros:** CVA = 0 when `LGD = 0` (no loss) *and* when `λ ≡ 0` (`S ≡ 1`, no default).
  - CVA ≥ 0 always; **monotone increasing** in `λ` (level shift up) and in `LGD`.
  - **Sandwich:** CVA ≤ LGD · (the always-positive-exposure value) = `LGD·∫DF·E[V]⁺-bound·dPD`.
  - **The independence factorization is the definition the code implements:** Crépey's
    `CVA = E[(1−J₁)P⁺]` (eq 17) reduces, under exposure⊥default, to `LGD·∫EPE·dPD`. Pin the one-period
    reduction (a single default date `T`) `CVA = LGD·DF(T)·EPE(T)·PD` **bit-identically** against a
    hand computation — a procedure's step made reachable (§4: extract the buried step into named
    surface).
- **Reduction + companion (§4).** Reduction: an **always-positive** exposure (`m ≫ s`) makes
  `EPE(t)=E[V(t)]` = the forward value, so CVA reduces to `LGD·∫DF·V_fwd·dPD` — exact, cross-checked a
  different way. **Companion exercising the non-degenerate case:** two-sided exposure with `m/s = O(1)`
  so both the `Φ` and `φ` terms bite — a reduction a no-op also satisfies is vacuous.
- **Degenerate-fixture guard (mandatory, post-F-1.2/T23) — assert non-degeneracy as a premise:**
  - hazard **non-flat** in at least one fixture (piecewise-constant `λ`, distinct pieces), else
    `S=e^{−λt}` and a wrong cumulation (`Σλτ` vs a mis-indexed sum) survives. Assert `max|λᵢ−λⱼ| >
    bound`.
  - exposure variance **non-zero** and `|m/s| = O(1)`, else `EPE=forward` (the always-positive
    degeneracy) and the `Φ/φ` split is invisible. Assert `s(t) > bound` and `m/s` moderate.
  - discount curve **non-flat**, else a mutant dropping `DF(t)` from the integrand survives.
  - **mutation is the proof of power** (RED by mutation, report the table as T23 did): drop `LGD`;
    replace `dPD` by `S` (level, not density); replace `EPE` by the forward; flat-cumulation `λ` error;
    drop `DF`. Each must be **killed**; a survivor is a hole.
- **Independence (§4).** The test computes its own reference throughout; nothing reads a value back
  from the code under test or from the same rendering the code was transcribed from.

## 3. Process
Branch `tranche/25-unilateral-cva`. `test` (RED: the closed-form CVA reference + the mutation table,
pre-registered) → `feat` → `docs` → `chore(release): v0.30.0`. `source:` lines name **both**
`CREPEY-2026` and `CAPINSKI-ZASTAWNIAK`. The new `market/` survival-curve module wants a
`crosscheck:` target — QuantLib has `DefaultProbabilityTermStructure` / survival curves
(`crosscheck: quantlib DefaultProbabilityTermStructure — deferred to the QuantLib tier`); the L5 CVA
functional likewise (`crosscheck: quantlib CVA/BasketLoss — deferred`, or `none — <reason>` if judged
out of the tier's scope, §8). **Open as a DRAFT PR** (§10 step 3). Raise any source discrepancy or
identity that does not hold as an `Ask:`, do not rule it.

## 4. What this tranche is NOT (scope fence)
No MC (T26). No bilateral CVA/DVA (T27). No FVA (T28). No collateral/CSA (T29 or deferred). No CDS or
defaultable-bond calibration — a constructed hazard curve is enough for the first consumer; the
survival↔quote calibration is pulled when a tranche needs it. KVA, BSDEs, neural-net regression, and
the risk-measure chapters are deferred wholesale to `to_go_deeper` at the arc checkpoint.

**Re-audit carry (C10's trigger, ruled on PR #33 §9.2).** The CVA integral's lower limit is the
historical-flow boundary (`t > valuation_date`). If this or a later XVA tranche genuinely needs that
test, **pull** the `is_historical(pay, valuation_date)` atom the re-audit identified (C10's trigger in
`conventions_register.md`) rather than inlining a sixth copy of the comparison — doing so upgrades C10
from `per-module` to `guarded — structurally` by product-pull. T25's exposure is an analytic fixture,
so it likely does *not* touch the boundary; the natural consumer is T26 (MC exposure over the real
engines). Do not extract the atom speculatively (§6) — only when a consumer needs it.
