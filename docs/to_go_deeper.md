# To go deeper — per-document routes not taken

What each worked source has **more to give** that we chose not to build, stated plainly with a hint
of the route back in. This is not `OPEN.md` (in-flight debt with a re-open trigger) and not
`references.md` (sources held). It is the honest back-cover of each document: where we stopped, why,
and what it would take to go further — so a later return, or a Substack "part two", starts from a
map rather than a cold read.

**Rules.** One section per worked source (by its `references.md` key). Each route names: the gap, why
we stopped there, what you'd build, and the source it needs — marking whether that source is already
`in-hand`/`worked-from`, `referenced-by` (named by a held source, not obtained — §7's no-citation-by-
inference still applies), or new. A route is a *hint*, not a spec; it becomes a tranche only when
interest picks it (§1) and Cowork writes the real plan (§10).

**This file is only for routes that would pass §4's oracle gate** (oracle-able with our discipline,
perhaps after an acquisition). A capability a source offers that **cannot** be oracled as the method
stands — a result anchored only by a plot or a stochastic twin-MC run, no closed form, cross-check, or
requirement-identity — is *not* debt and does **not** go here; it goes to `docs/out_of_method.md` as a
candidate for a future §4 statistical-oracle extension (charter §1). Draw the line per *capability*,
not per chapter: a closed-form identity inside a numerical chapter is a route here; the chapter's
numerical method is out of method.

---

## `BERHAN-2012` — *Pricing Inflation Derivatives* (ch 1–5 built, T2–T14)

Built: all four ch-1 instruments, the three models (JY, MM-I, MM-II) behind one seam, and the three
ch-5 calibrations — pricing analytics, verified, seasoned. What the document still holds:

1. **Risk & sensitivities (§1.3).** We priced; the *inflation* prices still do not move. **Corrected
   at tranche 24** (re-audit #1 F-3.2): L5 is **no longer empty** — tranche 22 put
   `risk/sensitivities.py` there (Black/forward delta, gamma, vega by three methods, anchored on the
   analytic Greeks). What this route describes is still genuinely unbuilt, and it is a *different*
   thing: a risk layer over the **inflation** engines, reached by bumping a `MarketSnapshot` and
   repricing through the registry, plus the `Priceable` protocol §2 of the charter anticipates (which
   T22 deliberately did not build — one consumer shape would make it a single-implementation
   abstraction, §6). **Route:** bump-and-reprice over the registered pricers first, analytic/AD
   later; the protocol arrives with the second kind of priceable. **Source:** none new. Still the
   most self-contained next step.

2. **HJM dynamics & simulation (§2.1–2.4).** We take the curves as given and never reconstruct them
   from drift conditions; there is no Monte Carlo anywhere. **Route:** a simulation engine for the
   nominal/real forwards under the JY/HJM dynamics — the thing that unlocks path-dependent and exotic
   inflation payoffs *and* gives MM-I/MM-II a **distributional** error bound. **Now in progress from
   the other side:** the Jäckel arc (project two) builds exactly this MC capability; once it has a
   forward-rate process, this route closes. **Source:** `JARROW-YILDIRIM-2003` (the drift conditions,
   `referenced-by` — must obtain); `MCINERNEY-ZASTAWNIAK` ch 4 (`in-hand`) for HJM/Gaussian; the MC
   craft from `JAECKEL-2002` (`in-hand`).

3. **The smile — the weakness all three models share.** Every model here is flat in strike; the
   implied-vol surface (T9) *represents* the market skew but nothing fits it, so OTM is underpriced
   and ITM overpriced (reproduced for both JY and MM-II). **Route:** a stochastic-volatility inflation
   model. **Source:** `BRIGO-MERCURIO-2005-SV` and `BRIGO-MERCURIO-2009-SABR`, or `KENYON-2008`
   ("Inflation is normal") — all `referenced-by: BERHAN-2012`, none in hand. The richest "part two"
   and the one the thesis itself keeps pointing at.

4. **A distributional bound for MM-I's freezing (not just a sensitivity).** T12(b) bounded the
   freezing error as a sensitivity (71% of σI per 200bp forward move); a probability-weighted bound
   needs the forwards simulated. **Route:** the LMM simulation (Jäckel ch 12), applied to the freezing
   recursion. **Source:** `MCINERNEY-ZASTAWNIAK` ch 5–6 (`in-hand`), `JAECKEL-2002` ch 12 (`in-hand`).

5. **The "practitioners approach" to the skew (§5.4.2, named, not taken).** BERHAN names implying a
   parameter set per Expiry/Strike pair as the alternative to the ATM-only restriction. **Route:**
   fit a parameter set per (expiry, strike) cell rather than one per expiry; a calibration variant,
   not a new model. **Source:** `BERHAN-2012` §5.4.2 (`worked-from`) — in hand.

6. **External cross-check (the QuantLib tier).** **23** `crosscheck:` markers now name real targets
   (of 45 provenance headers carrying the marker; the other 22 are explicit `crosscheck: none — …`
   exemptions, which §8's presence gate requires). **Corrected at tranche 24** (re-audit #1 F-3.2):
   the line said "six", which was true at tranche 6 and has not been true since. The tier itself is
   still unbuilt. **Route:** stand up the optional, dev-only QuantLib tier against those 23 sites
   (never a merge gate — §8). **Source:** QuantLib (a library, not a paper).

**The one honest caveat that governs any return (FP-1):** BERHAN's empirical results are *plots, not
tables* (Figs 1.1, 1.2, 5.1–5.10). Nothing above licenses reading a number off a chart — a deeper
build reproduces the *structure* (a sign, a monotonicity, a reduction), with every number attributed
to the library and its oracle.

---

## `JAECKEL-2002` — *Monte Carlo Methods in Finance* (the completion arc: T15–T22)

Built as the working spine: generation (7, 8, 9), processes (2, 3, 4), variance reduction (10 core),
Greeks (11), the BGM/J application (12). See `docs/jaeckel_completion_arc.md`. What the book still
holds — chapters skipped wholesale, then the debt *inside* the chapters we closed:

**Cross-cutting, recorded by re-audit #1 (F-2.1) — a rule-of-two second consumer:** the driftless
forward terminal `F·exp(−½V² + V·z)` is inlined in **two** places, `models/black_mc.py` and
`risk/sensitivities.py`'s `_terminal`. Identical one-liners today, so nothing is wrong; the risk is
the *convention* drifting between them (a sign, a half). §6's rule of two has fired, so **extract a
shared helper at the next touch of either file** — not a tranche of its own, and not worth a change
that no present defect motivates. (Tranche 24 also measured the adjacent C10 case and found five
inline copies of the invariant-6 comparison; that one is registered with its own upgrade trigger.)

**Acquisitions the arc hit (primitive data the book leaves on its CD):**
- **Sobol past 32 dimensions (ch 8.3).** Our table (Jäckel 8.3) covers d≤32; the primitive
  polynomials and direction-number initialisers for higher dimensions are on the book's CD, which we
  do not hold — third CD instance after Acklam's C++ and the inverse-CDF. d≤32 suffices for every
  consumer where Sobol pays (low-dimension payoffs); a high-dimension path does not benefit at
  feasible N anyway (the advantage washes out by d≈24), so this is an acquisition only if a genuine
  high-dimension QMC consumer ever appears. **Route:** obtain the extended direction numbers, then
  transcribe as the d≤32 table was (machine-reconciled, property-A oracle). **Update (T21):** the
  Bermudan's own LMM path (6 forwards × 8 steps = 48 dims > 32) is now a concrete would-be consumer —
  but it is **not pulled**, because the Bermudan's binding constraint is its *structural* sandwich
  (foresight bias + policy suboptimality, T18/T19), which Sobol cannot narrow; a >32 acquisition would
  buy a sharper per-price SE inside an unchanged bracket. So the bridge is demonstrated on a
  four-forward caplet path (where Sobol pays), and the >32 acquisition stays conditional on a
  high-dimension path whose *sampling* — not a structural bracket — is the binding constraint.

**Whole chapters deferred:**

1. **Salvaging a linear correlation matrix (ch 6).** Market correlation matrices are often not
   positive-semidefinite; ch 6 projects them to the nearest PSD matrix (hypersphere / spectral /
   angular decompositions). **Route:** an L1/L0 utility taking a non-PSD matrix to its nearest valid
   one, oracled by the eigenvalue floor and the Frobenius-nearest property. **Source:** `JAECKEL-2002`
   ch 6 (`worked-from`) — in hand. A genuinely useful utility with no consumer yet.
2. **Non-recombining trees (ch 13).** Jäckel's independent cross-check for the Bermudan (T18 oracled
   by a tree-free sandwich instead). **Route:** the ch-13 tree as the independent benchmark on a
   short-dated Bermudan; also the tight dual upper bound (Andersen–Broadie) that would replace T18's
   15.75% loose bracket. **Source:** `JAECKEL-2002` ch 13 (`worked-from`); Andersen–Broadie is
   `referenced-by`, not in hand. Cross-refs the BGM/J arc's own `to_go_deeper`.
3. **Copulæ and multi-asset dependence (ch 5, §9.4).** Co-dependence beyond a linear correlation;
   simulating multivariate copula draws. **Route:** a copula layer when a multi-asset product pulls
   one. **Source:** `JAECKEL-2002` ch 5 / §9.4 (`worked-from`). No multi-asset product yet.
4. **Engineering notes (ch 14).** CPU usage, overflow/underflow, parallelism, embedded path creation.
   **Route:** performance work when a product is too slow in practice — not a pricing capability.

**Debt inside the chapters we closed:**

- **ch 2:** the Feynman-Kac theorem and the Moore-Penrose pseudo-inverse — theory we did not need as
  code (the LMM used a pseudo-square-root, not the pseudo-inverse).
- **ch 3:** the exotic processes — Bessel, constant-elasticity-of-variance, displaced diffusion — none
  pulled by a product (GBM and the lognormal LMM were enough).
- **ch 4:** the Milstein scheme and the strong/weak-convergence theory (we built the predictor-
  corrector log-Euler; Milstein is a higher-order route if a product needs it).
- **ch 9:** simulating copula draws (§9.4) — ties to ch 5's deferral.
- **ch 10:** the variance-reduction techniques beyond the T19/T21 core — stratified sampling,
  importance sampling, moment matching, Latin hypercube, variate recycling, spectral path
  construction. Each a route pulled by a product whose accuracy the core did not reach.

**The honesty caveat (as for BERHAN):** Jäckel's results are convergence diagrams, not tables (§14.4).
Any deeper build reproduces the *structure* — a rate, a bound, a variance-reduction factor — with
every number attributed to the library and its oracle, never read off a plot.

---

## `CREPEY-2026` — *XVA Analysis* (the XVA spine arc: T25–, in progress)

Routes recorded as the arc runs; the full set lands at the checkpoint `CP_xva_spine.md`. So far:

1. **The amortising exposure profile over a swap's life — the realistic "hump".** T26 pins Leg B's
   martingale anchor exactly by observing the swap only over `(0, T_first]`, where the claim is fixed,
   deferring the full-life EPE profile across coupon payments (the canonical CVA picture). **Route:**
   the deflated-value-plus-accumulated-cashflows martingale
   `E[(V(t) + reinvested paid cashflows)/B_N(t)] = V(0)/B_N(0)`, exact across payments, which restores
   the hump without weakening the anchor to an approximation. **Source:** `CREPEY-2026` (`worked-from`)
   + standard exposure mechanics. Oracle-able and exact, not pulled — pulled when a product wants the
   full profile.
