# CLAUDE.md — pricebook charter

The law for this repo. When anything you would do conflicts with this file, this file wins. It is
deliberately lean: the detail lives in `docs/`, this is the enforceable and ruled subset. It is
also a teaching text — legible beats clever, everywhere.

---

## 0. What this is

A **pricing analytics library** where **every number is verified against the book or paper that
defines it**. Functional core, imperative shell: pricing is a pure function
`price(product, model) → result`; everything stateful is a thin shell around it.

**Analytics only.** In scope: products, models, calibration, engines, risk, the market-data value
types, and the **representation** of a product and a pricing request (the durable public contract).
Out of scope, built in *other* projects that consume the representation: GUI, database, broker/data
feeds. The library is pluggable, not plugged in. (Downstream ideas are parked in
`docs/future_projects.md`.)

---

## 1. The working loop — product-pull, driven by interest

The unit of work is **a product worth having**:

```
name a product  →  plan what it needs (method, model, core pieces)
                →  check the stock for each piece
                →  purify it (adapt to our conventions) or build it fresh
                →  ORACLE it against the held source
                →  reflect it into the library
```

Interest picks the target; the discipline is the constant; the books and papers are the oracle.
Top-down: the product drags in exactly the core it needs and nothing it does not. The library grows
by interest; `0.x` holds through the build. **What `1.0.0` will mean is deliberately left open — it
is too early to fix a finish line** — except that it will not be a module count (the migration bar
is retired). The question returns when it is ripe, from a ruling, not a guess now.

**What enters the spine, and where the rest goes.** A capability is *built* — enters the spine — only
if it passes three gates, all already in this charter: the **oracle gate** (§4 — a named green oracle
exists, in the order closed form > cross-check > self-consistency > trusted mark), the **pull gate**
(§1/§6 — a product worth having needs it *now*; nothing speculative), and the **interest gate**
(interest picks it). The oracle gate is the hard one, and it *is* the debt line: **what cannot be
oracled is not "debt to go deeper" — it is out of method.** So deferred work splits in two, by the
gate it fails:

- **`docs/to_go_deeper.md`** — a route that *would* pass the oracle gate (oracle-able with our
  discipline, perhaps after acquiring a source) but is not pulled yet. Genuine come-back-same-method
  debt.
- **`docs/out_of_method.md`** — a capability a held source offers that *cannot* pass the oracle gate
  as the method stands: a result anchored only by a plot or a stochastic twin-MC run, with no closed
  form, cross-check, or requirement-identity to pin it (Crépey's neural-net XVA regressions). Not
  repayable as-is — repaying it means a deliberate **§4 extension** (a statistical/stochastic-oracle
  tier that accepts convergence-based validation). Recorded as a **candidate** for that extension,
  ruled on only if interest ever pulls that way; never built under §4 as it stands.

*(Ruled with Bernardo, 2026-11-08, after three arcs made the spine/deferred cut by judgement —
BERHAN ch 1–5, Jäckel 9-of-14, XVA's elementary spine. The cut is a test now, not taste. The gates
are not new; the two-bin split and the out-of-method candidate flag are.)*

---

## 2. The spine (layers — dependencies point DOWN only)

```
L6  LIFECYCLE / SHELL        stateful; remembers, never computes
L5  RISK & CAPITAL           on the engine + a Priceable protocol
L4  ENGINES                  the stateless heart: price(product, model) → result
L3  MODELS + CALIBRATION     carry the MarketSnapshot they were calibrated to
L2  PRODUCTS                 pure data — describe, never price themselves
L1  MARKET DATA              immutable MarketSnapshot · curves · fixings · vols
L0  FOUNDATION               time, money, value types — FINANCE-FREE
     _stock/ (side)          read-only reference; never imported by the tree
```

A lower layer never imports a higher one (`verify.py acyclic`). Layer membership is **semantic,
not just acyclic** (`verify.py layers`): L0 is finance-free (no strikes, vols, payoffs, call/put,
discounting *in code* — a docstring explaining finance context is fine). Products are frozen data;
they do not price themselves — pricing lives in L4.

**Calibration splits by what it produces and what it inverts** — the one place the box above
misleads, so it is spelled out. A **curve bootstrap** (market quotes → a curve, using only L0+L1 —
T2's break-even inversion, T8's nominal par-swap bootstrap) produces *market data* and is **L1**
(it lives in `market/`). A **model-free parameter calibration** (model parameters fitted
analytically, no engine) is **L3**. A calibrator that **inverts an engine** — fitting parameters by
repricing through `price()`, as JY-to-a-floor-strip does — is a consumer of the engine and sits at
**L4** (`engine/`). `acyclic` enforces all three; the split is by what the calibration *produces*
(a curve → L1, model params → L3) and what it *reads* (the engine → L4), not by taste. *(The earlier
text called the break-even bootstrap L3; it is in `market/` at L1, and a charter stating a false
practice is the error §8 and §9 also had to correct.)*

---

## 3. The stateless engine contract (L4)

`engine.price(product, model) → PricingResult | PricingFailure`. The **model carries the
`MarketSnapshot` it was calibrated to** (`model.market`); the engine reaches market data only
through it, never as a second argument, so a model/market mismatch cannot be expressed.

1. **Referential transparency** — identical inputs ⇒ identical output.
2. **No ambient state** — no clock reads; "today" is `model.market.valuation_date`.
3. **No mutation** — product, model and snapshot are frozen.
4. **Failure is a value** — return `PricingFailure`, never raise-and-hope or a silent `NaN`.
5. **Config is explicit, and deferred to its first consumer** — a numerical-config value arrives
   with the first engine that reads it; never threaded speculatively (the dropped `numerics`).
6. **Valuation-date-aware (invariant 6)** — a flow on or before `valuation_date` is historical:
   excluded from the mark, **PV 0** (the shell's business), never discounted with non-positive `t`.
   `PricingFailure` is for inputs that *cannot* be priced, never for a settled trade.
7. **An estimate carries its error** — a `PricingResult` from a Monte-Carlo pricer carries its
   **standard error** (an optional field, `None` for an exact pricer), because for an estimate the SE
   *is* part of the result: a price is `100 ± 0.3`, not `100`. This is constitutive of an MC result's
   identity (§6), not speculative — two MC pricers want it (the caplet and the swaption), rule-of-two.
   And it is a **correctness** matter, not only efficiency: an oracle that re-simulates to recover a
   discarded SE gets the SE of *different* paths than the price, so the band no longer describes that
   price. The field rides along where it is meaningful and is ignored where it is not; the contract
   shape (`PricingResult | PricingFailure`) is unchanged. *(T17 finding 5; ruled by Cowork and
   **accepted by Bernardo** 2026-11-05, implemented at v0.20.1 — `OPEN.md` OI-2 closed. The caplet
   pricer needed no change: it returns a `MonteCarloEstimate`, which carried its error already.)*
   **`MonteCarloEstimate` and `PricingResult` are two layers, not two shapes for one idea.**
   `MonteCarloEstimate` (mean, standard error, n_samples) is the L0 **statistical primitive** the
   estimator produces; `PricingResult` is the §3 **engine return**. An L4 engine that prices via MC
   holds an estimate internally and **projects** it into a `PricingResult` (pv = mean × discount ×
   notional, SE likewise); a non-engine MC *helper* (`mc_caplet_price` — not `@register`ed, no Caplet
   product) correctly returns the raw primitive. A caplet gains a `PricingResult`-returning engine
   only when a Caplet product pulls one (product-pull), not before. *(T18 ruling: a third MC pricer —
   the Bermudan engine — returns a `PricingResult` with SE, so there is one engine-result shape, not
   three.)*

---

## 4. Oracles — the heart of the method

Nothing lands without a **named, green oracle** proving it against a known value. Preference order:
**closed form > external cross-check > self-consistency > trusted mark.** "Runs and looks right" is
not an oracle. Ruled principles:

- **Pre-register the oracle.** The failing reference-value test (RED) is committed *before* the
  code that satisfies it. Implement-then-check invites bending the code to a number.
- **Independence.** The test computes its own reference (e.g. `t = days/365`), never reads a value
  back from the code under test — **nor from the same source rendering the code was transcribed
  from.** A `pdftotext -layout` reflow merges a formula's parentheses, so a second transcription of
  it repeats the error: read a load-bearing formula in **`-raw` and `-layout` and reconcile**, and
  an "independent" transcription that shares the implementation's rendering is not independent.
  *(The predecessor's 5.14% bug passed because its oracle restated the builder's own composition;
  v0.4.0's mis-nested 2.5.11 passed a 108-point sweep because sweep and code were two copies of one
  merged rendering.)* **Where the source *derives* a formula in steps, an earlier step is an
  independent check on the final one — stronger than two renderings of one line, and available when
  the renderings fail.** A `-layout` reflow that mangles the final equation beyond reconciliation
  does not touch the pre-substitution form a page earlier; the two must agree, and a misprint would
  have to corrupt *both* derivation steps identically to survive. *(T12(b): `-layout` broke eq
  3.1.11 into columns, but p.25's pre-substitution integral — over `y` before `z=(y−µy)/σy` —
  reproduced it to 2.2e-16; both oracled.)* **Its more valuable mode is a discrepancy, not a
  confirmation:** when two steps of one source disagree, the source is internally inconsistent and
  the reader must *rule* which form is right (by dimension, by the standard convention, by what the
  downstream equations use) and pin that choice with an oracle — the discrepancy is found by the
  finance, not deferred to the reader. *(T13: eq 4.1.3 printed `Fn/(1+Fn)` while `Di` printed
  `τFn/(1+τFn)` three lines on — ruled `1+τF` (it is `Pn(Tᵢ₋₁)/Pn(Tᵢ)`, dimensionally sound, and the
  form 4.1.6 uses) and pinned.)* **The dual risk: a fresh transcription can introduce its OWN error,
  not just repeat the code's — and the disagreement PATTERN says which side is wrong.** A disagreement
  confined to the indices/parameters that exercise one special term implicates that term; a uniform
  disagreement across all of them implicates the implementation. So on a red transcription oracle, read
  the pattern before debugging the code — the instinct to suspect the code first is sometimes exactly
  backwards. *(T16: a reference for eq 12.8 used the off-diagonal ρ=0.80 for ρ_ii too; only forward 0
  — the one index whose drift sum never reaches the diagonal — agreed, and that asymmetry located the
  bug in the reference, not the model.)* **Two different sources are an independent check only if they
  render the SAME formula — confirm they agree before treating one as the other's transcription.** Two
  books addressing one quantity can be different-order *approximations*, not two renderings of one
  exact object; then each is its own transcription target and a measured nonzero gap between them is a
  companion to assert (small *and* non-zero), never the cross-source independence it resembles. Check
  for a structural signature that separates them. *(T17: the swap-rate vol is `Z·C·Zᵀ` in both
  McInerney 5.40 (weights frozen) and Jäckel 12.21 (weights differentiated) — 0.45% apart, and
  `ΣZ_frozen = 1` exactly while `ΣZ_exact ≈ 0.996`, proving different objects; the hoped-for
  cross-source independence did not exist.)* **For a large TABLE of constants, independence comes from
  two parsers, not two eye-reads — and the real guard is a requirement the table must satisfy, not the
  transcription.** The dual-rendering rule is for a formula (a handful of symbols a person reconciles);
  a 350-number table read twice by the same eyes is not two independent reads (fatigue correlates
  between passes), so parse both renderings *mechanically* and reconcile — different code paths are the
  independence. But the transcription is never the oracle: anchor on a property one wrong entry breaks.
  *(T20b: Jäckel's Table 8.3 (~350 **direction** integers — the Gray code is eq 8.22, a separate
  thing) was machine-reconciled across both renderings —
  32 rows, zero disagreements, every exponent `32−l` confirmed — and the oracle is **property A**, that
  any aligned `2^d`-segment of the Sobol sequence has exactly one point per sub-hypercube, which a
  single mistyped direction integer destroys.)*
- **Reduction is the preferred anchor, and it needs a companion.** A capability that generalises
  another is anchored by the **exact reduction** to the old case at the degenerate parameter
  (`σr → 0` ⇒ the deterministic limit; `I(t)=I0` ⇒ the inception formula) — exact by construction,
  better than a numerical limit. It **requires a companion oracle exercising the non-degenerate
  general case**: a reduction a no-op implementation also satisfies is vacuous.
- **Self-consistency telescopes.** A par-reprice-to-zero is satisfied by construction when
  construction and pricing share a convention — *even a wrong one*. Keep it as a consistency guard;
  the correctness anchor is a non-telescoping check of the structure itself (a pointwise identity,
  a count, an independent transcription).
- **No number is fitted to a figure.** When a source's results are plots, not tables, reproduce
  its structure, not its pixels; numbers come from closed forms, an external engine, or the stock's
  regression values — never read off a chart.
- **Requirement oracles — test what must hold for ALL parameters, not only numbers from the
  source.** `C(t=Tᵢ₋₁)=0` (a correction to a then-known quantity), `V² ≥ 0`, put-call parity: these
  are derived from the finance, so they **survive a transcription error** that a source-value oracle
  shares with the code. Every load-bearing formula carries at least one. And a **cross-product**
  oracle (one product composed from another's verified piece, different code each side) catches a
  shared source-reading error that no within-product check can — it has now done so twice (T3's
  strip reconciliation, T5's `cap−floor = the YoY leg`). *(This whole bullet is v0.4.0's mis-nested
  convexity, caught after landing by parity and by `C(t=Tᵢ₋₁)=0`, not by any number.)* A requirement
  oracle also catches **a misprint in the source itself** — a typo both renderings reproduce
  faithfully, which transcription-independence (`-raw` + `-layout`) is powerless against because both
  are faithful to the wrong print. The only property that survives the source being wrong is one the
  finance must satisfy: *a formula and its stated inverse must actually invert.* **T8's known-curve
  round trip caught two misprinted equations in `BERHAN-2012` §5.1** (a spurious sign in 5.1.1, a
  `j=0` vs `j=1` annuity index in 5.1.2) — the first defect traced to the source being wrong rather
  than to our reading of it.

- **Requirement oracles complement a pointwise transcription; they never replace it.** The bullet
  above is why a property survives a transcription error — but the converse is the trap: a property
  is blind to a *missing or wrong term the property still tolerates*. A freezing coefficient oracled
  only by its properties (vanishes at zero vol, moves with the vols, bounded under a shock) let a
  mutant that **dropped a `/(1+τF)` denominator survive all 404 tests** — the denominator is worth
  ~3%, not an equivalent mutant, and every property still held without it. So **every load-bearing
  formula is transcribed pointwise at least once** (independently, per the rules above), *and* carries
  its requirement oracles; the two catch different failures and neither substitutes for the other.
  *(T12(b): properties-only on the freezing relation; the pointwise transcription plus a
  denominators-differ companion is what killed the mutant.)* The two can meet inside a single
  formula, which is the cleanest demonstration that neither replaces the other: a wrong Box-Muller
  angle (`πv` for `2πv`) satisfies the exact requirement `x²+y² = −2 ln u` — `sin²+cos² = 1` whatever
  the angle — while failing the pointwise transcription. Assert both, **and** assert that the wrong
  variant *does* satisfy the requirement, so the companion is proven necessary rather than assumed.
  *(T15.)* **A set of oracles can be complete about a quantity's PRESENCE and SIGN and still be silent
  about its MAGNITUDE** — which is the same trap wearing a subtler disguise: not a vacuous assertion,
  but a *complete* set of assertions about the wrong aspect. When a quantity enters a formula as a
  **product** (`(fτ)/(1+fτ)·σ`), assert the product's **value** against an independent transcription,
  not merely that each factor is present and correctly signed. And beware the anchor's own blind spot:
  **an oracle made sharp by a quantity being exactly zero is, for that very reason, blind to that
  quantity** — so a closed-form anchor run under the measure where the target's drift vanishes cannot
  see the drift at all. The magnitude oracle that fills the gap is itself a transcription, made
  noise-free by differencing two runs that share a seed so everything but the quantity cancels.
  *(T16: three mutations of the LMM drift eq 12.8 — a dropped `dt` (8×), raw forwards for the ratios
  (5×), a dropped denominator (1.04×) — survived all 550 tests; killed by a two-numeraire one-seed
  identity `log f^A − log f^B = (µ^A−µ^B)·dt`, exact every path, matched to a fresh transcription at
  1e-13.)* **The one-seed differencing identity applies only when the quantity under test enters ONE
  place — the drift — so everything else cancels; a quantity in the diffusion/factorisation does not
  cancel and the instrument does not carry over.** When it doesn't apply, pin the quantity where it
  *is* directly observable instead: a correlation entering the factorisation is pinned by the realised
  covariance of the simulated increments reproducing `σ_iσ_jρ_ij·dt` for every pair — which pins all
  entries individually, stronger than a price pinning one correlation-weighted scalar. *(T17: ρ enters
  `C = σσρ·dt`, so the T16 trick gave 13.9% against a guessed band; the realised-covariance pin gives
  1.66% at 80k paths, halving as √N so a factorisation bias shows as failure-to-improve.)*

- **The shape behind all of the above: an oracle must contain the failure in its domain.** A green
  is evidence only about what the oracle *examined*. Every defect this project has shipped was an
  oracle whose domain excluded the thing that was wrong — checked against the stock instead of the
  source (T4's anchor, T6's instrument), against one rendering instead of two (v0.4.0's nesting), on
  the quote maturities instead of the whole curve where it could differ (T6's A4 premise), or one
  parameter at a time instead of jointly (the identifiability valley). Before trusting a green, ask
  *where the failure would live and whether the oracle looks there.* This is the floor rule stated
  at the top: widen the domain until it includes the failure, or the green is empty. **A bound on a
  price cannot see the algorithm that produced it.** Where the method is a multi-step *procedure* — a
  backward induction, a training loop, an exercise policy — an output bound (even a correct sandwich)
  examines the number, not the steps, so a valid-but-wrong variant of the procedure sails through it.
  Oracle the **steps pointwise**, and **extract a buried step into named surface** so a test can reach
  it (the reachability rule again, now for procedures not just guards). *(T18: four of five first-pass
  survivors were §12.8 procedure steps — the induction direction, the continuation roll, the `I>0`
  guard, the second coordinate `s_{i+1}` vs `s_i` — and no price-level bound caught any; the fifth, a
  dropped numéraire division, hid inside oracle 1's `4·SE` band, the exact T17 banded-resolution trap
  one tranche later, and the fix was to make the one-date reduction bit-identical rather than widen
  the band.)*
- **Degenerate fixtures, and mutation as the proof a test has power.** The commonest form of a domain
  that excludes the failure is a fixture whose key parameter sits where it cannot be seen — `τ=1`
  (the accrual invisible), a flat curve (interpolation invisible), a single period or node (a
  horizon or index invisible), a zero (a factor invisible). It has surfaced four times (T6, T8, T9,
  T10), and **twice a review pass — Cowork's and Code's — read the fixture and missed it.** So where
  an oracle's power rests on a parameter being non-degenerate, **assert the non-degeneracy as a
  premise**, and **prefer a mutation to a reading**: flip a sign, drop a factor, swap a horizon, and
  a surviving mutant is a hole a green suite is hiding (T10's dropped accrual survived all 312 tests
  until a non-unit `τ` was added). **The trap is not only numerical — a degenerate fixture can make
  two readings of a SOURCE agree, defeating a *transcription* oracle:** on annual ACT/365F periods
  `τ = 1` exactly, so `τFn` and `Fn` are the same number and the oracle pinning T13's hazard-1 choice
  between them was powerless until a semi-annual fixture (`τ ≠ 1`) separated them. Fifth appearance,
  and the first on a source-discrepancy oracle rather than a numerical one. **A varying fixture can
  still be degenerate — if it is invariant under the transformation the code performs.** A correlation
  `ρ_ij = exp(−β|i−j|)` is non-uniform yet *translation-invariant*, so `ρ[first+m][first+l] = ρ[m][l]`
  and a mutant dropping the span offset survives; a constant-across-a-slice accrual survives a dropped
  slice offset the same way. So the premise to assert is not "the values differ" but **"the block the
  code reads differs from the block a mutation would read"** — and the faithful source reading usually
  forces it (Jäckel 12.12 decays in the actual times `t_i`, not in index distance, which is the
  non-degenerate form). Sixth appearance, and the first where the degeneracy is *uniformity across a
  slice* rather than a specific value. *(T17: non-uniform accruals and a time-decaying ρ killed both
  mutants.)* **A varying, non-special, non-uniform fixture can STILL be degenerate — if the fixture's
  *distribution* has a symmetry the mutation respects.** Reversing an exposure gap
  `V(t)−V(t−δ) → V(t−δ)−V(t)` — the wrong *sign* of the exposure — survived all 986 tests because the
  fixture's mark was **driftless**, so the gap is symmetric about zero and the reversed gap has the
  *identical distribution*; a mean-based oracle (an EPE is a distributional functional) is structurally
  blind to the direction. The premise is neither "the values differ" nor "the block read differs from
  the block a mutation reads" but **"the fixture breaks the symmetry the mutation exploits"** — here a
  material drift, so `E[gap⁺] ≠ E[(−gap)⁺]` (0% apart driftless, 87% at drift 24/yr, 572% at 60/yr).
  Seventh appearance, and the first where the degeneracy is a *distributional symmetry* rather than a
  value or a slice. *(T29.)* **And a fixture can be degenerate in its *coverage of a guard duplicated
  across code paths*, not in any value** — when one condition appears N times (a `tᵢ ≤ T` dividend-date
  filter written once per dividend convention), a fixture exercising the special case in **one** branch
  leaves the other N−1 copies unguarded, and dropping the guard on an unexercised branch survives.
  Count the **copies of a guard**, not just the values — and the robust fix is usually §6:
  **consolidate the duplicated guard into one place** (one `dividends_before(T)`), so a single fixture
  suffices, rather than multiply fixtures forever. Eighth appearance, and the first where the
  degeneracy is a *coverage gap across duplicated code*, which is a duplication smell wearing a
  fixture's clothes. *(T30: a post-expiry cash dividend covered the discrete branch while the
  proportional branch's identical filter, dropped, survived all 1010 tests.)* Mutation is also the
  **mandatory recovery when pre-registration
  was missed** — the empirical proof of power that RED-first would otherwise have given, never a
  licence to skip RED-first. **A surviving mutant is a hole — unless it is an *equivalent mutant*,
  one that leaves behaviour identical** (flipping a residual sign under a sum of squares, which is
  sign-invariant). The exception must be **proved by showing the outputs bit-identical, never
  assumed because the mutation looks harmless** — the assumption is how a real survivor gets waved
  through as "probably equivalent" (T10(b): a sign-flip survived five-of-six, proved equivalent by a
  bit-identical fit, same κ, σ and iteration count). **For code whose output is an iterative solver's
  fitted result, bit-identity is demanded of the OBJECTIVE, not of the fitted output.** A solver's
  last digits are legitimately path-dependent — permuting the free variables reorders the Jacobian
  columns and moves the result ~1e-15 — so bit-identity of the *fit* is unavailable and its absence
  proves nothing. But the objective the solver minimises is a pure function, and the exception's proof
  relocates there: show that the mutation is a **reparameterization** `P` under which the residual is
  bit-identically invariant — `residuals_mutant(x) == residuals_orig(P(x))` exactly, at sample `x` —
  from which the fitted output is provably identical and its agreement to the solver's tolerance is
  then corroboration, not the proof. This is still a bit-identity check (a theorem, not a readable
  "looks equivalent" argument — which is exactly what the bar exists to refuse); it is only moved to
  the level where it is available. *(T14 §9.3: reversing `_params_from`'s key assignment survived all
  453 tests; it is the reversal `P`, residuals bit-identically invariant under it, so the fitted dict
  is identical and the ~1e-15 fit difference is pure solver path — a proved equivalent mutant.)*
  **A third survivor kind, distinct from both a hole and an equivalent mutant: unreachable code.** A
  mutant can survive because its branch is **statistically unreachable** — a redraw guarding against
  an exact `0.0` from a `[0,1)` source (probability ~2⁻⁵³). That is not an equivalent mutant (the
  behaviours are not identical — the mutated branch would misbehave *if reached*; it would hang),
  it is dead defensive code, and the fix is §6's: **delete it, or relocate the check somewhere it can
  be exercised** (the domain guard moves to `Φ⁻¹`, which can be called with `0.0` and refuse). An
  untestable branch in L0 is surface that should not exist. *(T15: a redraw loop survived all 468
  tests; removed, the guard moved to where a test can reach it.)* **The criterion is
  reachability-from-a-test, not improbability** — a 2⁻⁵³ branch and an outright dead branch are the
  same defect, and relocating the guard to where a test *can* call it (`Φ⁻¹(0.0)` refusing) is what
  distinguishes a real check from dead code.
- **An oracle must be measurable above its own numerical noise — prefer a behavioural invariant to a
  noise-dominated number.** A finite-difference Jacobian condition number for the six-parameter JY
  fit swung three orders of magnitude with the step size (2.5e7 → 2.1e10) and differed again under
  central differences, because the smallest singular value sits below the FD noise floor: a
  threshold on it asserts noise and flakes across OSes. Where a numeric diagnostic is noise-
  dominated, the oracle is a **behavioural** invariant instead — a **multi-start parameter spread**
  (does the fit land on one point?), or a stable **ratio** (a singular-value ratio) — both
  step-independent and identical on both OSes. T11: identifiability is proven by the spread, never
  by a condition number. **Multi-start agreement is necessary, not sufficient** — all starts can
  land in the same *wrong* basin, so where a synthetic truth exists, put it among the starts (an
  exact optimum no other start can beat) and assert **distance from truth** beside spread: spread
  shows non-uniqueness, distance shows wrongness. (T11: `ρnI`'s spread read 22% without the truth in
  the start set and 469% with it — recovered at the *wrong sign* at rms 8e-15.) **A NULL needs
  averaging as much as an effect does — an oracle whose expected answer is "no difference" is the one
  a single sample most easily fakes.** The measurable-above-noise rule is usually read as "detect an
  effect above the noise"; its mirror is the neutrality check, where one seed manufactures a
  discovery. Average it over seeds. *(T21: the Brownian bridge must do *nothing* under pseudo-random
  draws — one seed said it was 6× better, pure luck; over twelve seeds it averaged 1.31, neutral as
  the algebra requires. And the same tranche: "Sobol+bridge beats pseudo-random" is not assertable as
  a fitted exponent — one seed's slope swung −0.64 to −1.06 with the range endpoint and on the short
  range beat the bridge by luck — so it is an RMS over the range (stable to two decimals), the
  exponent kept only for the two deterministic constructions. Third tranche running where a headline
  had to be restated into a form measurable above its own noise.)*
- **A variance reduction must be proved UNBIASED before it is credited with reducing anything — and
  the commonest way to break that is a control variate centred on an APPROXIMATION.** A control
  variate `X̄ − β(Ȳ − E[Y])` is unbiased only because `E[Y]` is the *exact* mean; centre it on an
  approximate price and the approximation's error becomes permanent — a bias no path count removes, and
  the unbiasedness oracle (the reduced estimator must still hit the closed form) is what catches it. So
  **an approximation is not a control, however well it correlates.** The free, exactly-centred controls
  live in the measure: under a numéraire measure every asset/numéraire ratio is a martingale, so
  `E[A/B_N] = A(0)/B_N(0)` *by construction, no model result involved*. Pick the control there.
  *(T19: the Rebonato swaption (a §5.11 approximation) was specced as the Bermudan's control and would
  have injected its ~0.5% freezing error permanently; replaced by the numéraire-relative annuity —
  exact mean, ρ = 0.74, a 54% variance reduction matching `√(1−ρ²)` to 1e-12, zero bias. And a
  separate trap: a price-sandwich that is **structural** (foresight bias + policy suboptimality, its
  width invariant as the SE falls — T18's Bermudan held at ~15% over 64× paths) is NOT sampling error,
  so variance reduction sharpens each price's SE but cannot narrow it; the invariance of the bracket
  under reduction is itself the unbiasedness check.)*
- **A tolerance that lumps two error sources together cannot tell them apart — and a band wide enough
  to absorb Monte-Carlo noise is wide enough to absorb a real defect.** Band an MC-vs-approximation
  check as `k·SE + the approximation's own MEASURED error` (the SE measured from the *same* paths the
  pricer consumes, the approximation error measured where sampling noise is negligible — a huge path
  count), never one flat percentage standing in for both. And **assert the band's resolution**: state,
  from measurement, the smallest defect a green actually rules out at the committed path count, and add
  a dedicated higher-path test at the count where the defect size you care about provably falls
  outside. A green is evidence only about what the band can resolve. *(T17: a flat 3% band mixed a
  0.5% freezing error with ~2% MC error and resolved only a ~20% vol error at 6k paths — a 10% error
  sat inside it; rebanded `4·SE + 1.5%`, with a 12k-path power test where 10% falls outside.)*
- **A reference is code too — it needs its own accuracy argument in the regime where it is used.**
  §4's independence rule says the test computes its own reference; this is the sharp corollary: that
  reference is an implementation with its own error, and in an extreme regime (a deep tail, a near-unit
  probability) the reference can be the *weaker* side while the code under test is right. So argue the
  reference's accuracy where you use it, and where the home-grown one is weak reach for an independent
  high-precision reference (`scipy.special.ndtri`/`ndtr`), kept off the production path (tranche 11's
  `distributions.py`-off-scipy arrangement exists precisely so the oracle may use scipy while the
  library does not). And the trap that hides this for tranches: **an ABSOLUTE-tolerance oracle is blind
  to RELATIVE error where the true value is near zero** — "tails → 0/1" passes however bad the relative
  error gets, which is why `norm_cdf`'s lower-tail cancellation (`(1+erf)/2` as `erf → −1`: 6.1e-06
  relative at `x=−7.03` versus `erfc(−x/√2)/2`'s 7.4e-15) survived green for fifteen tranches. In a tail,
  assert **relative** accuracy against a reference that has it. *(T20a: a root-find on `norm_cdf` was the
  weaker side three times — disagreeing in the tail, failing to resolve a root at `p=1−1e-12`, missing at
  `p=0.999` where the code was bit-exact; primary cross-check moved to `scipy.special.ndtri`.)*
  **And a reference's regime error can hide as a convergence *plateau* rather than a value mismatch —
  so the convergence RATE is the detector a flat band misses.** Give each home-grown reference an
  explicit **validity regime** and assert the references agree only where both hold. *(T25: an EPE
  quadrature truncated above the distribution's mass for large `m/s` and returned ≈0 where the true
  EPE was ≈100 (Bachelier EPE → `m` as `m/s → ∞`); the CVA error then **plateaued at 3e-04 instead
  of converging**, and only the plateau — no single value — revealed it. Fourth "a reference is code
  too", and the first caught by the rate, not the value: a flat percentage band would have absorbed
  it as quadrature noise.)*
- **A pure refactor is proved by an exact-arithmetic identity, not always by bit-identity.** Bit-
  identity (the equivalent-mutant proof above, and the Black and HW extractions) is available only
  when the refactor *preserves the order* of the floating-point operations. When the refactor
  legitimately reorders them — as factoring one expression into a shape two consumers share must —
  the last digits move, and **demanding bit-identity would forbid the very reordering the design
  needs.** The purity proof then is the **exact-arithmetic algebraic identity** (the two expressions
  are provably the same quantity in ℝ) plus **agreement bounded to a few ULP** with the reordering
  named as the cause; the pre-existing oracle set staying green at its own tolerances is necessary
  but is *not* the proof, because a loose tolerance can hide a real change — the identity is. Claiming
  "bit-identical" when the ops were reordered is false; reporting the ULP gap with its cause is the
  honest substitute. *(T12(a): `ψ·(Pn(Tᵢ₋₁)·PrRatio·e^C − Pn(Tᵢ))` → `ψ·Pn(Tᵢ)·(m−1)` is one
  identity — `Pn(Tᵢ)·m = Pn(Tᵢ₋₁)·PrRatio·e^C` — reordering one division, so the JY leg moved rel
  1.6e-15, ~7 ULP, one division round-trip. Keeping bit-identity would have meant keeping JY's
  grouping, which is exactly what makes a second model impossible.)*

These principles bind **Cowork's review as much as Code's build**: a formula "verified term-by-term"
against the same rendering the code used is not verified.

---

## 5. The stock — content, never authority

`_stock/` holds the whole previous tree, gitignored, **mined for CONTENT (what is true —
conventions, formulas, edge cases, reference values), never for STRUCTURE**, and never imported. A
stock file's organisation carries no authority here.

**The stock is content and regression; the held source is authority** — for *every* reader,
including whoever writes the oracle plan. Oracle anchors and formulas are read from the source, not
inferred from the stock's code, which may be a weaker or different object. *(Precedent: the T4
anchor was mistakenly taken as `ρ → 0` from the stock's first-order `yoy_convexity_adjustment`; the
source's eq 2.5.11 makes it `σr → 0`. Caught at the handoff by pre-registration.)* Each purified
piece earns its own oracle; an imported-but-unoracled convention is worse than an absent one.

**Our own `docs/` are not an authority either — the same rule, one level up.** A formula parked in a
spec, a `to_go_deeper` anchor, or a ruling is *content*, not the source: at build time it is
re-transcribed from the held source in two renderings (§4), never copied forward from our doc. A
number can be wrong in our doc exactly as it can be wrong in the stock, and a wrong anchor copied
forward launders the error into code. *(2026-11-08: the §1 ruling recorded the static KVA as
`h·ES(L◦)`; Crépey eq (25) is `(h/(1+h))·ES(L◦)` — a `pdftotext -layout` reflow split the fraction
across two lines, the exact §4 single-rendering hazard, caught before it reached `to_go_deeper` by
the dual rendering and by the printed requirement bound `0 ≤ KVA ≤ ES(L◦)`, which holds
unconditionally only for the fraction.)*

---

## 6. Simplicity — solve the real problem

- **Nothing speculative.** No hooks, config knobs or extension points for hypothetical futures.
- **No abstraction without two present consumers sharing a SHAPE, not a word.** A real leg and a
  year-on-year leg share the word "leg" and not the payoff — so they stay separate types, no base
  class. Rule-of-two triggers on a shared shape.
- **Compelled bundling is vocabulary, not abstraction.** Grouping cohesive fields to meet the limit
  (`Term`, `HullWhiteParams`, `RealLeg`) is required, not speculative — kept **concrete and frozen**,
  no polymorphism. Whether it generalises is decided at the second real consumer.
- **A carried-but-unread field is allowed only when constitutive of the instrument's identity /
  required for faithful representation** (and then it is exercised by the round-trip) — never
  "the pricer might read it later." The test is constitutive-of-identity, not does-the-pricer-read
  it. *(An indexation lag distinguishing two otherwise-identical linkers: carried. A publication
  lag the index name already implies: not.)*

---

## 7. Signatures, fields, conventions, references

- **Max 5 arguments per function and 5 fields per dataclass** (`verify.py fields`, ruff `PLR0913`).
  Over the limit is **un-bundled vocabulary**, never a reason to raise it: group into a frozen
  value object (`Money`, `Term`, `HullWhiteParams`, …). **Never suppress** (`# noqa` / `# type:
  ignore`) — fix the cause. The limit and its `PLR0913` gate bind **`src/`**; **test helpers use
  judgement** — a reference fixture's raw knobs need not be bundled, since a value object can hide the
  very inputs a test exists to show, and CI does not run `PLR0913` over `tests/`. An unwieldy test
  helper is still a smell; prefer an existing value object when the arguments genuinely *are* one.
  *(T26b: a pre-existing 6-arg reference helper in `tests/L3/test_jarrow_yildirim.py` is left as-is —
  a rule question Code raised, ruled src-only, not churned.)*
- **Cross-module conventions** (shared day counts, the time measure, discounting dimension) live in
  `docs/conventions_register.md` with a three-state Guard column — **guarded** (a test compares the
  modules), **per-module** (each tested, nothing asserts they agree), **owed** (no oracle). A shared
  convention with no structural guarantee needs a cross-module oracle; until it has one it is
  recorded, never folklore.
- **References.** Every provenance `source:` names a key in `docs/references.md`, and may cite only
  an `in-hand`/`worked-from` key. A work merely *referenced-by* a held source is not citable until
  obtained — no reference by inference. **Gated by `verify.py references` (tranche 26b):** a cited
  key must resolve to a citable status, a key cited by any module must be `worked-from` (the
  register's own vocabulary — *held and built/oracled against*), and a `worked-from` key must have a
  citer. The gate was floated here as *"proposed, not built, until the first drift makes it earn its
  place"* and **five** drifts earned it, none found by careful reading: two sources left `in-hand`
  after T25 built against them, `JAECKEL-2002` self-contradictory (header `in-hand`, body
  `worked-from`, 18 citers), `MCINERNEY-ZASTAWNIAK` `in-hand` with 10 citers and a body promising
  `worked-from` *"until T10 lands"* — sixteen tranches earlier — and `ACKLAM-2000` `in-hand` with its
  coefficients transcribed in `distributions.py`. Three were corrected by hand at T26's handoff and
  **the gate's first run caught the other two**, which is the argument for gating a discipline rather
  than restating it. It keys on the header status and the citations, never the prose body: a gate
  that parsed narrative could not be falsified.

---

## 8. Provenance, debt, cross-checks

- **Provenance header on every `src/` module:** `stock:` · `source:` · `oracle:` · `slice:`
  (`verify.py provenance`).
- **No debt is silenced silently.** Every suppression, shim, skip or load-bearing TODO goes in
  `OPEN.md` with a rationale and a re-open trigger, or it does not exist. CI invariant: suppressions
  − ledger entries = 0. Debt may never buy a green oracle. **A ledger entry's severity is characterised
  by sweeping the defect's domain, not from one measurement** — a single data point inherits its own
  domain, so "6.1e-06 relative, degrades in the tail" written from one sample understated OI-3's
  `norm_cdf` defect by five orders of magnitude (the old form returned *exactly 0.0* below `x ≈ −8.3`),
  and it sat understated for two tranches. The §4 domain rule applies to a debt's characterisation as
  much as to an oracle's. *(T20c closing OI-3.)*
- **Deferred external validation is tagged in the provenance header, and its presence is gated.** A
  `crosscheck: <library> <thing> — deferred …` line sits with `stock:`/`source:`/`oracle:`/`slice:`
  in the header (no `#`), so the optional QuantLib tier (later, dev-only, skippable — never in the
  reproducibility contract or a merge gate) is complete by construction, not from memory. A module
  **wants** one when it implements a numerical finance convention or model formula a known library
  independently implements — the model-dependent pricers, and the market-data constructions (curves,
  the fixing-series resolution); value types, dispatch and pure-data products do not.
  **Gated by presence, with explicit exemption** (the `fields-exempt` / `slow-test-exempt` pattern):
  every module under `market/`, `models/` and `engine/` carries either a real `crosscheck:` target
  or `crosscheck: none — <reason>`, enforced by the `provenance` gate, which already walks them — so
  a new module with neither fails CI and nothing drifts silently. *(OI-1, ruled then upgraded
  2026-11-01: form-only was ruled first, and T7 drifted the same hour — a new `market/` module with
  a real QuantLib target and no marker, which form-only could never have caught. Presence-with-
  exemption is the half that bites. The original "zero instances" reading was a grep for
  `# crosscheck:` — the `#` form the tree never used.)*

---

## 9. Gates, branches, flow

- **`verify.py`**: `acyclic` · `layers` · `fields` · `debt` · `version` · `provenance` ·
  `references` · `slow` · `tests` (floored). CI runs ubuntu + windows, Python 3.12. Tolerance-based oracles are a hard rule
  (transcendental results differ 1–2 ULP across runtimes); the byte-identical statelessness check
  is same-process only. **A timing gate's reading is only valid when measured in isolation** — running
  `slow` concurrently with the full suite or a type-checker reports contention, not run time (T19: a
  backgrounded gate under load read 690s against a true 1.75s, a 400× artefact). And its headroom is
  not bankable: re-time a file after adding oracles rather than assume the margin (T19: six oracles
  added to the Bermudan file without re-timing produced a real Windows `slow` failure). **The floor
  gate's own discovery must be total** — the `tests` floor counts only the directories it enumerates,
  so a test directory it does not discover runs *unfloored* and escapes the floor silently; discovery
  enumerates every test dir, not a fixed `tests/L*` set (layer dirs first, so the lowest layer still
  fails fast). *(T26b: a new `tests/tooling/` ran green under plain pytest but outside the floor, which
  read 902 against a floor of 911 while the suite was green — §4's "contain the failure in its domain"
  applied to the gate that enforces every other gate's floor, and visible only because that tranche
  added the first test there; a tranche adding none would have left the directory unfloored forever.)*
- **Periodic whole-tree re-audit — every 11 tranches, and at each arc close.** The gates above run on
  every push and catch *structural* drift (layering, fields, debt balance, provenance), so the
  re-audit targets only what they cannot: **(1) oracle power** — a full-tree mutation sweep, because
  mutation is per-tranche and never re-run, so an oracle strong at T4 can be silently weakened by a
  later refactor (the project's signature defect, applied to history). **Full-tree means the oldest
  tranches, not the arc just built** — re-audit #1's one hole (F-1.2, T3's linker accrual, dropped
  `ψ` invisible because every fixture was `τ=1`) lived in the *oldest* instrument and was found only
  when the sweep was extended past the recent arc it was first scoped to: the signature defect
  committed against the audit itself. The text already said "full-tree"; the first run under-applied
  it, so this is the precedent, not a new rule. **The sweep also flags any load-bearing mutation killed
  by a single test** — a *single-point-of-failure oracle*, where a later **test** deletion silently
  reopens the hole with the suite still green (the signature defect one level down, in the test suite's
  own domain). These are the re-audit's highest-value re-run targets, and an oracle guarding a
  *previously-shipped* hole especially wants a second independent killer. The periodic re-audit is the
  systemic backstop — a deleted killer resurfaces at the next sweep — so recording them is the floor;
  hardening the thinnest is the ceiling. *(Re-audit #2: F-1.2's dropped-`ψ` dies only to T23's
  non-unit-`ψ` test, T26's numéraire divisor only to Leg B; both recorded, and F-1.2 — an already-made
  mistake — given a second independent killer.)* **(2) cross-tranche consistency
  and duplication** — "owed"/"per-module" conventions that never got their guard, duplicated value
  objects, a rule-of-two abstraction overdue at its second consumer; **(3) register accuracy** —
  `to_go_deeper`, `OPEN.md`, `references.md`, the arc plans still describing reality; **(4) charter
  staleness** — a clause now stale or in tension (it has happened). **(1) and (3) are not separate
  passes — they inform each other.** Honestly re-deriving a register row *is* an oracle-power probe:
  to establish what a convention row actually claims you mutate the convention and watch who notices.
  Re-audit #1's second hole — a 41% `capfloor` mispricing the suite could not catch — was found in the
  *register* pass, not the mutation pass, while re-deriving the invariant-6 row. So run the register
  and oracle-power passes together. *(T24: flipping the historical-flow boundary `>`→`>=` while
  re-deriving C10 gave three survivors — two proved equivalent, an expired Hull-White caplet being
  exactly 0, one the real `capfloor` hole where an already-paid period's payoff is determined.)* Run
  with **independence** (§4's spirit): the reviewer is not the surface that built the code — the `analysis/` read-only-copy +
  `AUDIT_FINDINGS.md` pattern pointed at current `main`. Findings are ruled by Cowork and become
  tranches or `to_go_deeper`/`OPEN` entries; the cadence is tracked in `docs/audit_log.md` (last
  re-audit: version + tranche) so the trigger is mechanical, not remembered. First run at the Jäckel
  arc close. *(Cadence 11 set by Bernardo, 2026-11-07; Bernardo owns §9 and may veto.)*
- **One branch per tranche** (`tranche/<NN>-<slug>`), commit sequence `test (RED) → feat (GREEN) →
  docs → chore(release)`. One version bump + one `CHANGELOG.md` entry per tranche; CI asserts
  `__version__` matches the top entry.
- **The merge method is not legislated here.** The draft said *"land by squash-and-merge, the only
  GitHub merge that leaves `main` both linear and Verified"* — but **every PR to date (#1–#5) landed
  as a merge commit**, so the clause described a practice the repo does not follow, and a charter
  stating a false practice is worse than one that is silent. The signed-and-linear argument is a
  migration-era concern Bernardo has explicitly set aside as not worth the attention, and nothing
  about it is broken: the RED→GREEN sequence is preserved either way, on `main` under a merge commit
  and on `refs/pull/<N>/head` under a squash. If it ever matters, it becomes law then, from a
  ruling — not from a doc describing an aspiration.
- **Review before merge.** Nothing merges until the pre-merge pass is in.

---

## 10. Code ↔ Cowork

**Cowork decides. Code verifies and builds.** Ruled by Bernardo, 2026-11-01, replacing the earlier
"Code leads the build" split — two surfaces each proposing a design produced competing judgement on
the same question, which is overhead, not review.

**Cowork owns the decisions**: what the next unit is, the target, the pieces, the layer, the
vocabulary, and above all **the oracle choice**. It has the external reach (sources, documents).
**Code owns execution**: the code, the tests, git, and **verification by execution** — Cowork cannot
run the suite and never marks anything passing.

The boundary that matters, because it is the one thing that is *not* a decision: **Code reports what
execution shows, including when a Cowork claim does not hold.** A number that fails, a gate that
rejects, a markers-exist claim with zero instances — those are findings, and reporting them is
Code's job, not a competing opinion. Code then **asks Cowork to rule**, and does not rule itself.
*(Precedent: three claims in the T1–T6 checkpoint did not survive execution. Correcting the record
was right; deciding what to do about §8's empty crosscheck mechanism was not Code's call, so it
became `OPEN.md` OI-1 for Cowork to rule.)*

So Code does **not** propose the tranche, rule a design question, or recommend the next unit.

**The next unit is decided in the Cowork conversation and read from `docs/` — never asked of Bernardo
in the build channel.** Cowork keeps the tranche queue explicit (an *arc plan* in `docs/` names the
sequence, e.g. `jaeckel_completion_arc.md`'s T19→T22), so after a merge the next unit is *already
decided*: Code reads the queue and proceeds on the normal handoff rhythm. **Code does not end a report
with "what's next?" or "the next unit is Bernardo's call"** — that pushes a design decision onto the
one person the two-surface split exists to spare it. When the queue empties at an **arc boundary**, the
next direction is settled *here* (Bernardo's interest + Cowork's spec) and written to the queue, not
raised in the build channel. Bernardo's only inputs to a tranche are the two "cowork is done" handoffs
and the merge; he is never asked to make the same decision in both conversations. *(Ruled by Cowork,
2026-11-06, to remove double decision-making; Bernardo owns §10 and may veto.)* **"Read the queue"
settles the direction, not the spec — step 1 is unchanged: Cowork still writes the per-tranche spec to
`docs/` before the build.** The arc plan's one-line queue entry names the unit; it is not the spec. If
the merge lands before Cowork has written the next spec, Code waits for it (or, if it proceeds against
the arc-plan design to avoid stalling, raises *every* judgement call as an `Ask:` rather than settling
one — the T20a fallback, which worked but is not the intended flow). *(T20a built with no spec written
— Cowork's miss; the handoff should carry the spec.)* **And a spec's cited anchor is verified to
exist before handoff** — read-before-asserting (§5) applies to our own tree, not only to sources: a
spec may name a prior tranche's capability as confidently as it names a number, and be as wrong.
*(T22 named "Black's analytic Greeks (T9)" as its anchor; T9 built the vol surface, no Greeks existed
— a one-line grep before the ruling would have caught it. The build-raises-it loop did catch it, so
this is a cheap guard, not a new gate.)*

The prompt channel is `.cowork/thread.md` (append-only, one `Ask:` per entry, pointers not pastes);
durable state goes to `docs/`, never only the thread.

**The tranche sequence, and the one rule that keeps it from breaking.** Bernardo is the only one who
merges, and **he merges when Code says to, never when Cowork finishes.**

| # | who | what |
|---|---|---|
| 1 | **Cowork** | decide the unit: target, pieces, layer, and the oracle. Writes it to `docs/`. |
| 2 | **Bernardo** | "cowork is done" |
| 3 | **Code** | build RED → GREEN against that decision, stamp, open the PR **as a DRAFT**. Raise findings as `Ask:`s — never a counter-proposal. |
| 4 | **Cowork** | review the built PR; rule any finding Code raised. |
| 5 | **Bernardo** | "cowork is done" |
| 6 | **Code** | commit whatever Cowork wrote to disk, re-run the gates, confirm CI, **then mark the PR ready — and that readiness IS the "merge now" signal**. |
| 7 | **Bernardo** | merge. |

Step 6 is why the merge cannot be step 5. **Cowork writes files but does not commit them** — Code
owns git — so a merge triggered by "Cowork finished" lands a PR without Cowork's own artifacts.
*(Precedent: PR #7 merged three minutes before Cowork's checkpoint entry, stranding a ratified
charter edit and a checkpoint record on a merged branch and costing PR #8 to recover. PR #6 was the
same shape, caught only because Code said "wait" in time.)*

**Cowork's artifacts are committed under a `docs:` message that names them, staged by path — never
`git add -A`, and never swept into a `test(...)` or `feat(...)` commit.** A charter or ruling edit
riding inside a test commit is invisible in the log, which is how Code once *enforced a §10 ruling it
had committed blind and never read*, violating it twice in the same session. So Code reads every file
Cowork wrote before committing it, and stages it by name. *(T19's results doc and the 2026-11-06 §10
queue ruling were both misfiled by `git add -A` into unrelated commits; T20a's handoff staged four
paths by name and fixed it.)*

**The PR opens as a draft, and that is what makes the order self-enforcing.** A line in a message is
a weak brake — **PRs #7, #14, #15 and #16** merged ahead of review because "merge" is the obvious
thing to do with a green PR (#6 was the same shape, caught only because Code said "wait" in time). GitHub will not merge a draft, so opening as a draft (step 3) and marking ready only
at step 6 turns "merge when Code says, never when Cowork finishes" from a request into a mechanism
the tooling enforces: the ready flag is a thing only Code sets, and it is the same event as "merge
now." One extra flag for Code, nothing for anyone else. *(Ruled by Cowork, 2026-11-03, PR #17 the
first draft; adopted because it implements §10's existing step-6-before-step-7 law rather than
changing who does what — Bernardo owns §10 and may veto.)*

Steps 1 and 4 are both real passes, not one split in two: step 1 sets the oracle before a line is
written (it is where T6's identifiability oracle was decided), while step 4 reviews what was
actually built, is where a checkpoint gets written, and is where **the tranche's journal beat is
appended** — FP-1's capture discipline for the Substack series (`_substack/pipeline.md` §3), tied to
this step precisely because it is the one that always happens, after the journal once fell twenty
tranches behind. (The journal is gitignored; this is a plain append, not a committed artifact.)
Either may be skipped by Bernardo's call; the step-6-before-step-7 order may not.
