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

## `CREPEY-2026` — *XVA Analysis* (the XVA spine arc: T25–T29, **CLOSED** v0.35.0)

Routes recorded as the arc ran; the full set is in the checkpoint `CP_xva_spine.md`. The recorded routes:
*(header corrected at re-audit #4 — the arc closed at re-audit #2 (T29, v0.35.0) but this line still read
"in progress", stale through two re-audits; a register-accuracy miss recorded in `reaudit_04_findings.md`.)*

1. **The amortising exposure profile over a swap's life — the realistic "hump".** T26 pins Leg B's
   martingale anchor exactly by observing the swap only over `(0, T_first]`, where the claim is fixed,
   deferring the full-life EPE profile across coupon payments (the canonical CVA picture). **Route:**
   the deflated-value-plus-accumulated-cashflows martingale
   `E[(V(t) + reinvested paid cashflows)/B_N(t)] = V(0)/B_N(0)`, exact across payments, which restores
   the hump without weakening the anchor to an approximation. **Source:** `CREPEY-2026` (`worked-from`)
   + standard exposure mechanics. Oracle-able and exact, not pulled — pulled when a product wants the
   full profile.

2. **Crépey's `(1−γ)⁻¹` survival-measure normalisation of the XVA adjustments.** T27 ruled the
   elementary spine stays **standard risk-neutral** (explicit first-to-default factor, no
   normalisation), because `(1−γ)⁻¹` belongs to Crépey's cost-of-capital / fininsurance measure — the
   deferred frontier — and importing it alone would mix measures. **Measured worth: a further 4.707%
   of CVA** on the T27 fixture (beside the 2.679% first-to-default term the spine *does* carry).
   **Route:** apply the bank-survival-measure normalisation across CVA/DVA/FVA — a one-divide-by-`S_own`
   change, but arc-wide (it moves every CVA number) — *only* as part of building the cost-of-capital
   framework it belongs to (most of which is `out_of_method`: KVA, the anticipated BSDEs). **Source:**
   `CREPEY-2026` Prop 1 eq (7), the bank survival measure (`worked-from`). Oracle-able (a closed-form
   factor); deferred by convention, not by method.

---

## `BROCKHAUS-2016` — *Equity Derivatives and Hybrids* (the equity arc: T30–T36b, **CLOSED** v0.44.0)

Arc closed 2026-11-13 at the capstone; the full deferred set is in `CP_equity_hybrids.md`'s honest
completion ratio — the exotic zoo (barriers/cliquets/asians/autocallables ch 3), implied-vol dynamics
(ch 7), correlation & copulas / multi-asset (ch 8–9), FX & quanto (ch 15), local-stochastic &
jump-diffusion (ch 6.3–6.4), portfolio credit (ch 12.4), Heston calibration + Carr-Madan FFT,
portfolio/collateralised equity CVA, and structural/asset-barrier WWR (ch 14.4). The specific
oracle-able routes recorded as the arc ran:

0. **Correlated-default defaultable equity for the STOCK itself (Brockhaus 13.1.2).** T36b built the
   13.1.2 apparatus (`Φ₂`, the `a_T` solve, the correlated Gaussian hazard) for the *counterparty's*
   default in wrong-way risk; the correlated defaultable-*equity* vanilla — the stock's own
   jump-to-default skew, `ρ` between the default and the diffusion — was never pulled (T35 built only
   the independent-default 13.1.1). **Route:** now cheap, since `Φ₂` exists at L0 and the apparatus is
   in the tree; a defaultable-equity model gains the `ρ` and reuses the bivariate-normal four-term call.
   Oracle: `ρ→0` reduces to T35's independent default, `h→0` to T30. Pulled when a skew-from-credit
   product wants it.

1. **Consolidate the `tᵢ ≤ T` dividend-date filter into one `dividends_before(T)` (code hygiene, not a
   capability).** T30 writes the pre-expiry dividend filter once per convention (continuous/proportional/
   discrete), so the guard is duplicated three ways and a fixture covered only one copy (the §4
   coverage-gap finding; closed with per-branch fixtures). **Route:** extract the shared filter so the
   guard is single-point and a single fixture suffices — rule-of-two is well past its trigger (§6).
   **Do it at the next touch of the dividend/forward code** (likely T32's local-vol or a later equity
   tranche), not a tranche of its own. Oracle-able trivially (the extraction is a pure refactor, §4).

---

## Tooling — the τ=1 accrual-fixture guard (declined at T37, 2026-11-14)

A CI guard that would flag an accrual-bearing test fixture whose accruals are all unit (`τ=1`), to catch
a *future* annual-dates fixture at commit rather than at a re-audit. **Assessed and declined at T37**, not
deferred for lack of interest: both prototypes were built and tested. A **textual** scan (annual date
arithmetic) false-flags `test_inflation_capfloor.py`, whose accrual is killed by two tests — it cannot
see load-bearing-ness. A **behavioural** instrument of `CouponSchedule.periods()` is clean on discovery
but, as a gate, flags 7 of 14 modules with zero real holes (four — `index_ratio`, `jy_calibration`,
`jy_full_calibration`, `seasoned` — have no accrual multiplication in their subject code at all). The
missing half — *does the subject code use the accrual?* — is a dynamic property only mutation testing
answers, which is the periodic re-audit's job (§9, now carrying the gateability boundary). **Route to
revive:** a static type/effect marker on accrual-consuming code that a gate could read without running
it (none exists today). Until then the periodic sweep + §4's premise rule are the backstop — they caught
all three `τ=1` instances (F-1.2, F-3.1, F-3.2).


---

## `OKANE-2008` — *Modelling Single-name and Multi-name Credit Derivatives* (the credit arc: T38–T44b, **CLOSED** v0.52.0)

Routes recorded as the arc ran; the full deferred set (bonds & asset swaps ch 4, CDS indices + index
options ch 10–11, CMDS ch 9.4, implied/compound correlation ch 19, the copula-skew zoo ch 21, CPPI/
CPDO/CDO²/tranchelets ch 22, the base-correlation surface / bespoke mapping / tranche hedging ch 20.6–20.9,
the ρ_B interpolation convention ch 20.4, and the ch 23 dynamic models as a possible `out_of_method`
candidate) is the completion ratio in the checkpoint `CP_okane_credit.md`. The recorded routes:

1. **The non-knockout default swaption (ch 9.3).** T40 built the **knockout** swaption (the standard
   contract). The non-knockout variant is, by O'Kane's own decomposition, **knockout + front-end
   protection** — and the front-end protection is a short protection leg over `[0, tE]`, which
   `credit_legs.protection_leg` already computes. **Route:** a new `NonKnockoutDefaultSwaption` product
   (not a flag on the knockout one, §6) whose engine adds the front-end protection term; oracle by the
   decomposition (`non-knockout = knockout + front-end`, different code each side) and the `R→1` /
   `h→0` reductions to the knockout. Cheap; unpulled (nothing needs it), so deferred by the §1 pull gate.

---

## `GATHERAL-2006` — *The Volatility Surface* (the vol-surface arc: T46–T49, **CLOSED** v0.55.0)

Opened at T46 (SVI), closed at T49 (SVJ); the completion ratio is in `CP_gatheral_vol.md`. The routes:

1. **Gatheral–Jacquier (2014)'s closed-form no-butterfly conditions — an ACQUIRE-SOURCE route, and
   the one place T46's certificate has a stated gap.** The 2006 text carries **no** butterfly
   condition for SVI: the ch 3 digression gives the calendar condition and says butterfly is
   *"relatively easy to eliminate"*, and that is all. So T46 derives the risk-neutral density from
   Breeden–Litzenberger instead (`models/svi.py::risk_neutral_density` — `p = g·n(d₋)/√w` with
   `g = (1 − kw'/2w)² − (w'²/4)(1/w + 1/4) + w''/2`, obtained from `e^k p = c'' − c'`) and certifies
   admissibility in two pieces: the **quoted span** numerically, at 401 points, and the **two tails**
   analytically, because `g(±∞) = ¼ − (b(1±ρ))²/16 ≥ 0` is *exactly* Lee's `b(1±ρ) ≤ 2` (verified to
   8 decimals on both wings of three slices). **What that leaves uncovered is the intermediate
   region** — beyond the far edge of the quotes, before the asymptote dominates. A finite scan cannot
   close it and no held source gives the closed form that would.
   **Route:** obtain Gatheral, J. & Jacquier, A., *Arbitrage-free SVI volatility surfaces*,
   Quantitative Finance 14(1), 2014 — **not held**, and a work merely referenced is not citable (§7),
   so this is an acquire-source note rather than a deferred build. With it, the scan is replaced by
   the paper's parameter-space conditions and the certificate becomes global. Until then the gap is
   named in `_butterfly_breach`'s docstring rather than papered over.

2. **Ch 7's quantitative volatility-surface asymptotics — the dropped T47 (§1 pull gate, 2026-11-16).**
   The arc plan queued a T47 for ch 7's asymptotics; T46 took the one part the arc consumes — **Lee's wing
   bound `β*, α* ∈ [0,2]`**, which pins eq 3.20's radical and is the no-arbitrage wing condition. The
   **quantitative remainder** — the `β* ↔ q*` moment inversion (eqs 7.11–7.12), the **Benaim–Friz** limit
   refinement (eqs 7.13–7.14), `g(q*)`, and the Drăgulescu–Yakovenko/Heston tail-genericity argument
   (p.99) — has **no consumer** in the committed spine: T48's vol-swap convexity is a Heston computation
   (ch 11) reusing T31+T33, T49's Bates is a characteristic-function extension, and the Jensen bound is
   model-independent; none reads the quantitative asymptotics. Oracle-able with our discipline (closed-form
   limits cross-checked against Heston/SABR tails), so a genuine come-back-same-method route — **not**
   built, per the pull gate. **Pulled when a product needs the tail quantitatively** (a far-wing
   extrapolation of the SVI surface beyond the quoted strikes; a replication whose tail truncation is
   material), and then built *where it is consumed*, not standalone.

3. **Options on variance `(V_T − K)⁺` — deferred at T48, and the oracle lesson is the point.** The cheap
   half is free: the call's Laplace transform *in strike* falls straight out of the transform T48 builds,
   `Ĉ(s) = (F(s) − 1 + s·E[W])/s²`. The cost is recovering `E[(V_T−K)⁺]` by a numerical **inverse** Laplace
   transform — a materially new object (T48's own quadrature is a *forward* transform of a positive decaying
   integrand; the inversion is Abate–Whitt-class and returned negative prices on a first attempt). So by the
   §1 pull gate it is **not** T48b force-fit — nothing in the spine pulls it (T49/Bates does not) — it is a
   come-back-same-method route, pulled when a product needs variance options. **The mandatory lesson for
   that build (measured at T48):** the two oracles the T48 spec proposed are *structurally blind* — the
   zero-strike reduction `C(K)→E[V]` is carried by the **exact** `s·E[W]/s²` term and passes on a *broken
   inverter* (0.1% at K=1e-4 while negative at interior strikes: the T39 blind spot — the limit is exactly
   where the inversion's contribution vanishes), and put-call parity is satisfied **by construction** (P
   derived from C). A complete oracle set about the limit and the relation, silent about the interior (the
   T43 shape at the set level). So a variance-option build **requires an absolute-level anchor at interior
   strikes** (its own MC), not these two — most of a tranche, which is why it is deferred, not folded.

4. **A non-cancelling inversion (Carr–Madan / COS) — the limit tranche 50 could not fix.** T50 made
   the Fourier truncation adaptive and repaired OI-4, but it also measured a floor the truncation
   cannot reach past: `P1 = ½ + I/π` is a difference of two `O(½)` quantities, so it carries an
   absolute noise floor of ~2e-13 **whatever the grid**, and a price inherits `forward × 2e-13`.
   Below that the sign of a deep out-of-the-money price is arithmetic noise, which is why
   `checked_call_price` returns zero there rather than pretending to a number.
   **Route:** this is a property of the *parameterisation*, not of the quadrature — an inversion that
   integrates the option value directly (Carr–Madan's damped transform, or a COS cosine expansion)
   never forms `½ + I/π` and so never cancels that way. Oracle-able with our discipline (the existing
   high-precision reference, plus agreement with the present pricer wherever the present pricer is
   above its floor). **Unpulled:** nothing needs deep-OTM short-dated prices below `1e-10` of spot
   today, and the §1 pull gate says a capability arrives with the product that needs it. Pulled if a
   product ever prices in that regime — a far-wing tail hedge, or a replication whose weights blow up
   out there.

## `LICHTERS-2015` — *Modern Derivatives Pricing and Credit Exposure Analysis* (the collateralised-discounting arc: T54–T58, **CLOSED** v0.60.0)

Routes recorded as the arc runs; the full deferred set is the completion ratio in
`docs/modern_discounting_arc.md`. The recorded routes:

1. **Cross-currency / foreign-collateral collateralisation (ch 4) — DEFERRED, needs an FX layer first
   (ruled 2026-11-18, the T56 step-1 grep).** A trade collateralised in a *foreign* currency discounts via
   the FX (collateral in currency B discounts a currency-A trade), so it is inherently multi-currency. The
   tree has **no FX machinery at all** — `Currency` is a tag on `Money`, there is no FX spot/forward, no
   cross-currency basis swap, no multi-currency curve interaction. So this is not a tranche: it needs an FX
   layer (spot, forward, the cross-currency basis bootstrap) built first, which the single-currency CSA
   spine does not pull (§1). **Oracle-able** with our discipline — **covered interest parity** (the FX-forward
   identity) and the cross-currency basis — so a genuine come-back-same-method route, not out of method.
   **Pulled when** an FX layer exists (its own arc or prerequisite), or interest wants cross-currency pricing;
   then built *where consumed*, on top of FX. The `CurveSet` keying already anticipates it (`CurveKey` carries
   a currency), so the collateral-currency-dependent discount curve slots in when FX arrives.
2. **Collateral-choice option / cheapest-to-deliver, thresholds / asymmetric CSA, negative-rate floors,
   initial margin / MVA (ch 5.2–5.6).** The CSA optionality and margin detail beyond the perfect-cash-CSA
   discounting of T55. Oracle-able (the collateral-choice option ≥ 0; the threshold reduction to T55 at
   threshold → ∞); unpulled — T55 is the perfect-CSA spine, these are its refinements.

## Tooling — gate `F401` once the registry modules carry `__all__`

Re-audit #6 (F-6.3) found Pyflakes had never been selected: `ruff.toml` carried only `PLR0913` and
`SLF001`. Running it surfaced two dead locals — one of which made tranche 58's capstone **simulate a
whole LGM path set per call and discard it** — and ten genuinely unused imports. All twelve were
removed at tranche 59 and **`F841` was gated**, because an unused local is statically decidable and has
no legitimate exception, which is exactly §9's test for what may become a gate.

**`F401` was deliberately not gated, and the reason is the prerequisite.** The engine registry's
`@register` imports are unused *by design* — importing them for the side effect is the whole point — so
a wholesale `F401` select needs `per-file-ignores`, which is the exemption-list shape §9 declines ("a
gate needing a 7-of-14 exemption list cannot fail"). Ten such imports exist today.

**Pulled when** those modules carry explicit `__all__` entries naming what they re-export, at which
point Pyflakes is satisfied without an ignore and the gate is honest rather than exempted. Until then
an unused import can accumulate unseen — the cost is small and visible (`ruff check --select F401` on
demand), which is why this is a route and not a tranche.

## Commodities (project eight, `CLARK-2014`)

- **Partly-fixed (seasoned) Asian options** — an average-rate option on which some fixings have
  already settled. `CLARK-2014` scopes itself identically ("no historical fixings have taken place
  yet"), and T61 refuses a partly-fixed schedule rather than price it: a commodity `MarketSnapshot`
  carries no realised prints, so treating a seasoned Asian as unfixed is a plausible *wrong* number
  (not a refusal §3.4 roughness). **Needs a commodity fixing series first** (an L1 market-data
  primitive, a real piece — not a flag), at which point the realised geometric/arithmetic partial
  average is a known constant folded into the adjusted forward. Oracle-able (reduces to T61 at zero
  settled fixings, to a deterministic payoff at all settled); unpulled — no product needs seasoned
  Asians yet. *(Surfaced at T61, 2026-11-19.)*

### More commodity routes (recorded at the arc close, re-audit #7)

- **American commodity options — Barone-Adesi-Whaley + a binomial lattice** (`CLARK-2014` ch 2.6).
  Oracle-able, but its natural anchor is a lattice the tree would build only for it; not pulled by the
  closed-form spine. The lattice would also serve American options under Schwartz.
- **Two-factor / stochastic-convenience-yield models** (ch 2.10.2–2.10.3, Gibson–Schwartz). The one-factor
  Schwartz (T64) is the spine; the second factor is pulled only if a product needs it. Reduces to T64 at
  zero second-factor vol.
- **Calendar and Asian spread options** (ch 2.9.3–2.9.4). The Asian spread composes T61/T62 with T63's
  spread machinery — a real route, unpulled. Oracle-able (reduces to the vanilla spread at a single fixing).
- **Schwartz market-forward calibration (`α̂_X`)** (ch 2.10.1, surfaced at T64). T64 carries the
  risk-neutral composite `α̂_X = α_S − σ²/2κ − λ/κ` because no futures set separates its parts (T11
  identifiability); *calibrating* `α̂_X` to a market forward curve is the fit nothing has pulled yet.
  Oracle-able (a one-equation inversion with the T8 round-trip pattern).
- **Seasonality / deseasonalised forward curves** (ch 5.2.1). A seasonal deterministic shift on the forward
  — oracle-able as a deterministic overlay, but the market-specific seasonal *estimation* (ch 3–9) is
  quotes-and-plots, closer to `out_of_method`. Recorded; unpulled.

## Stochastic rates (project nine, `MCINERNEY-ZASTAWNIAK`)

- **Cross-model Bermudan validation — the HW lattice Bermudan vs the T18 LMM-MC Bermudan.** Was T68's
  oracle 3 (specced as the headline), **not delivered and correctly so**: matching the two needs HW
  parameters that reproduce an LMM European swaption, but the LMM swaption is a **Monte-Carlo price** (or a
  Rebonato *approximation*), so the calibration would inherit sampling noise — or an approximation's bias —
  into the **premise** the whole comparison rests on. That is **T19's rule** (an approximation must not be
  the thing you centre on) applied to a *calibration* rather than a control variate — a sharp catch by Code.
  It is a real unit of work with its own oracle question: *quantify the match's own error, and show the
  structural foresight+suboptimality bracket swallows it* before the Bermudan comparison means anything.
  T68 is **sound without it** (the one-exercise reduction is exact/bit-identical — the last exercise step
  *is* the European closed form — plus Bermudan≥European and the backward-induction steps oracled pointwise),
  so this is a confirming cross-check, unpulled. *(Surfaced T68, 2026-11-21.)*
- **Sharper Bermudan quadrature — split the roll integral at the exercise boundary.** The backward-induction
  rolls integrate a kinked `max(exercise, continuation)`, and Gauss–Hermite loses its convergence *rate* at a
  kink (~2e-3 spread across 48/96/144 nodes — T68 honestly asserts a *bounded spread*, not convergence). The
  fix is to split each integral at the exercise boundary so each piece is smooth; oracle-able (the pieces
  recover the Gauss–Hermite rate), unpulled — T68's grid is chosen (401 points, 1.2e-4) so this term is an
  order and a half below the dominant one, so it is not load-bearing yet. *(Surfaced T68, 2026-11-21.)*

### The ch-8 smile routes and the ch-7 exotics (recorded at the arc close, re-audit #8)

*Recorded by Code at re-audit #8, 2026-11-21.* Re-audit #8's register pass (Cowork's) reported these as
already here and they were not — the file's last touch was T68, so it carried the two Bermudan routes above
and nothing from T71/T72. The routes themselves were ruled (in T72's spec, in `CP_stochastic_rates.md`'s
completion ratio, and in `references.md`'s `MCINERNEY-ZASTAWNIAK` chapter table, whose ch-8.5 row *names*
`to_go_deeper` without the route being in it — a pointer to an empty bin). Transcribed here from those
rulings, which is where §1 says a deferred-but-oracle-able route lives. See finding **F-8.1**.

- **Stochastic-volatility smile (ch 8.5) — the arc's `out_of_method` CANDIDATE, not a `to_go_deeper`
  route.** SABR-adjacent, and the section offers no closed form, so it cannot pass §4's oracle gate as the
  method stands. By §1's two-bin test that makes it a candidate for a deliberate §4 extension (a
  statistical/stochastic-oracle tier), **not** come-back-same-method debt. Recorded here only as a pointer to
  that distinction, because every other artifact that mentions it has called it `to_go_deeper` — which is the
  wrong bin, and the reason the bins exist. *(Ruled a `to_go_deeper` route in the T72 spec and the
  checkpoint; re-binned here as a candidate, flagged for Cowork since §1's bin assignment is a ruling.)*
- **CEV as a short-rate process** (rather than a forward-rate smile). The absorbing-boundary behaviour at
  `r = 0` for `β < 1` is its own subject — a short rate that absorbs at zero and stays there is a different
  modelling claim from a forward that does. Oracle-able (the same Cox machinery plus the absorption
  probability already built), unpulled: nothing needs it. *(Surfaced T72.)*
- **Drifted CEV** — an equity or FX CEV option, i.e. `BROCKHAUS-2016` eq (6.10)'s space-time change rather
  than T72's driftless forward-measure case. **Pulling it must start at the source defect**: eq (6.10) as
  printed carries a spurious `/T` against the space-time change printed on the same page, recorded in
  `references.md`'s Brockhaus defect log and deliberately unpinned because a caplet's forward is already a
  martingale. Whoever pulls this transcribes the *derivation step*, not eq (6.10). *(Surfaced T72.)*
- **CEV / HJM / G2++ surface calibration** — fitting `(σ, β)` (or the HJM vol function, or G2++'s five
  parameters) to a market skew rather than taking them as given. The T11 identifiability discipline is the
  whole content: `σ` and `β` trade off along a valley, so the oracle is a **multi-start parameter spread**
  plus distance-from-a-synthetic-truth, never a condition number. Oracle-able, unpulled. *(Surfaced T72.)*
- **The ch-7 exotic interest-rate products** — CMS, range accruals, ratchet floaters, trigger swaps. The
  chapter the arc did not touch; each is a product-pull away and all are oracle-able against the single-curve
  spine the arc built. *(Recorded at the arc close.)*
- **The gamma/corridor variance-swap MGF cross-checks** (`KWOK-ZHENG` §4.3.1/§4.3.2). T75 built the MGF
  route as a *second, independent method* for the Heston fair variance and used it to give T74's vanilla
  variance swap the cross-method anchor it lacked. The same MGF forms exist for the **gamma** and **corridor**
  swaps and would be a bonus independent anchor for T74's other two products — available now that the
  machinery exists, but **not load-bearing** (T74's gamma/corridor are already anchored by transcription,
  reduction and requirement). Pull only if a later tranche wants the §4.3.1/§4.3.2 MGF forms for their own
  sake. *(Ruled `to_go_deeper` by Cowork, 0401, closing T75 Ask 3.)*
