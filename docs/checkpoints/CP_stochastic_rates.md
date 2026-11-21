# Checkpoint — the single-curve short-rate arc (project nine, `MCINERNEY-ZASTAWNIAK`)

**Closed 2026-11-21 at `v0.74.0`, 3069 tests, eleven gates green.** The ninth vertical, and the first
**revisit** arc — a source already `worked-from` since T10 (the Hull-White cap machinery), mined for the
chapters T10 left. The target Bernardo set: a solid **single-curve** short-rate spine, the deliberate
pre-crisis counterpoint to the collateralised/multi-curve arc (project seven) — these models generate **one
curve** from the short rate or the forward curve, so they *exercise* the single-curve mode of the design
invariant rather than asserting it. Built in **seven** tranches (the ~6 Bernardo chose, +1 when the ch-8
smile split along its real seam).

## What was built
- **T66 — Vasicek** (+ Merton as its κ→0 reduction). The canonical mean-reverting Gaussian short rate; the
  affine `B(t,T)` **is** `hw_beta`, so Vasicek is the constant-θ sibling of the held Hull-White. `v0.68.0`.
- **T67 — G2++** (two-factor Hull-White). Two correlated Gaussian factors; the one-factor reduction a
  cross-check against already-verified `bond_option_variance`. `v0.69.0`.
- **T68 — Bermudan swaptions in Hull-White**, on an exact `r`-lattice (not the LMM's MC+regression). Brought
  the HW Jamshidian European swaption. `v0.70.0`.
- **T69 — Gaussian HJM** — the forward-rate framework; the drift condition a *requirement* oracle, the HW
  reduction exact. `v0.71.0`.
- **T70 — multi-factor HJM**, generalised into one `GaussianHjmModel` holding a vol tuple (independent
  factors — no inter-factor correlation, the framework's actual content). `v0.72.0`.
- **T71 — Normal/Bachelier + displaced-diffusion** smile caplets (reparameterisations of Black); the
  Bachelier positive part promoted to **L0** so the L4 caplet and the L5 CVA EPE share one production formula.
  `v0.73.0`.
- **T72 — CEV**, the capstone — Cox's (1996) non-central chi-squared caplet (via `BROCKHAUS-2016` §6.1; the
  `SCHRODER-1989` arrangement is quarantined in `references.md` as recollection, no held source names it), a
  new L0 ncx2 primitive.
  `v0.74.0`.

## What the arc reused — the single-curve machinery as its backbone
Every ch-3 model is **affine**, so `hw_beta` (the `B(t,T)`), `zero_bond_option` and `bond_option_variance`
(T10) carry the bonds and caplets; **`simulate_lgm_states`** (the OU simulator, T57) is the MC companion for
every short rate; `linalg`'s `pseudo_square_root` (T63) correlates G2++'s factors; the T18 `exercise_boundary`
backs the Bermudan; `black_price`/`black_implied_vol` carry the smile models; and the T25 CVA Bachelier EPE
**is** the Bachelier caplet (promoted to L0 and shared). The models **nest** — Merton ⊂ Vasicek ⊂ HW, G2++ →
1F HW, Gaussian HJM → HW, CEV → Black(β=1)/Bachelier(β=0) — so the reductions are the oracle spine, and most
anchored against **already-verified in-tree code** (the arc's signature).

## Method advances — two clauses, both in the first two tranches (the convergence story)
- **§4 — the third reduction-verification sibling (T66):** a reduction exact in ℝ yet **numerically
  unreachable** (a `(1−e^{−ατ})/α` removable singularity) earns a **convergence rate in a stated regime,
  never a band** — because banding at a numerically convenient point passes while the limit is unverified,
  the over-application of T61's band-it clause.
- **§4 — the regime is measured PER MODEL (T67):** the sound regime depends on how many times the formula
  divides by the vanishing parameter, so a prior tranche's boundary is not this one's (G2++ divides by α³,
  three orders tighter than Vasicek). T72 sharpened it again (the regime can be bounded by *our
  implementation's* reach, not the formula's — the Poisson mixture cannot evaluate where scipy can).

**And then nothing — T68 through T72 grew the charter not at all.** Every finding landed inside existing
text. That is the arc's real headline: the method has **converged** on this class of work. The early arcs
grew §4 most tranches; this one grew it twice, both in the opening pair, and then only *applied* a charter
that had become sufficient — including absorbing a figure-only source and a misprinted one without a new rule.

## The CEV source-correction — a first for the project
T72's spec assumed MZ §8.3 prints Schroder's CEV formula. **It prints none** (an SDE, two simulated
histograms, a pivot to displaced-diffusion) — a figure-only result, `to_go_deeper` by §1, not buildable from
MZ. Rather than drop or improvise, the build searched every held PDF and sourced the formula from
**`BROCKHAUS-2016` §6.1** (Cox 1996, already `worked-from`) — and found **that** printing **misprinted**,
identically in both renderings. Three **requirement oracles** condemned it (K→0 → −K, K→∞ → the forward,
β=0.5 undefined), and the correct form was **ruled by Monte Carlo** (600k paths, absorption) committed as the
anchor that shares no line with the code. The lens (not a clause — a synthesis of existing ones): a
*reconstructed* formula is anchored by requirement oracles (structure) + closed-form reductions (endpoints)
+ MC (interior magnitude), admissible only because the MC is **not** the sole anchor (MC-only is
out-of-method, §1). Both source gaps are in `references.md` (the MZ figure-only note, the Brockhaus
defect-log entry) so neither is re-attempted.

## Cowork's spec misses — named, because the split absorbed every one
This arc the specs were repeatedly **analogy-driven** — specced from the prior tranche and the familiar form
rather than from *this* source's notation/structure/printed-values and *this* tree's layering: T66's `κ(θ−r)`
parameterisation (source uses `θ−αr`), T68's `BermudanSwaption` (LMM-index-keyed), T70's correlation oracle
(the factors are independent), T71's `β` name / L4→L5 reuse / ULP-that-isn't / under-ask-vs-printed-marks,
T72's missing-then-misprinted source and the β=0 "identity" that absorption breaks. **Code caught every one
at build, cheaply, nothing shipped** — the two-surface split working exactly as designed. The consolidated
method-watch (no clause): *at step 1, derive the oracle from this source and this tree's actual state —
notation, printed values, structure, layer — never from the analogous prior tranche.*

## Honest completion ratio — the deferred ledger
Built as the spine: Vasicek/Merton, G2++, the HW Bermudan, Gaussian + multi-factor HJM, and the three ch-8
smile models (Bachelier, displaced-diffusion, CEV). Deferred to `to_go_deeper` (oracle-able, unpulled):
the **cross-model Bermudan validation** and the **boundary-split quadrature** (T68); **stochastic-volatility
smile** (ch 8.5, SABR-adjacent, no closed form — the arc's `out_of_method` candidate); **CEV as a short-rate
process** (absorbing-boundary behaviour), **drifted CEV**, and **CEV/HJM/G2++ surface calibration** (the T11
identifiability discipline); the **ch-7 exotic IR products** (CMS, range accruals, ratchet floaters, trigger
swaps). **Open debt:** `OPEN.md` **OI-7** alone (the `MarketSnapshot` fields-exempt, carried). **Convention
C16 fired** (the four caplet engines' shared strip walk) → re-audit #8's duplication pass.

## Governance this arc
- **Cowork stayed one spec ahead** the whole arc (T67–T72 written before each merge).
- **The journal held live** at every step-4 (T66–T72), current to v0.74.0.
- **The draft-PR mechanism's own near-miss and fix:** the ready flag was set once with a red Windows runner
  (the wait loop gated on *not-pending*, not *passing*); caught, reverted, fixed structurally, and reported —
  the mechanism's failure mode surfaced and closed by the surface using it.

## Close
The arc is complete on its builds. The close triggers **re-audit #8** (§9 — arc-close trigger, ~eight
tranches since #7 at T64/T65). Scope and findings in `docs/audit/reaudit_08_findings.md`: the proactive
invariant-6 sweep (its **fourth bite** landed at T72, so it is load-bearing), the arc's reduction/HJM/CEV
anchors, the CEV reconstructed-formula anchor set, and **C16's strip-walk duplication as the duplication
pass**. After it, a fresh **arc boundary** — Bernardo's interest + a Cowork spec.
