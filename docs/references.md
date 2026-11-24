# References register

The sources this library is built and verified against. **This is a governance artifact, not a
reading list.**

## The rule (the whole point of this file)

**A work enters this register only when the actual source is IN HAND and the claim we use has been
read in it — never by inference from another work that cites it.** The predecessor accreted
references by inference — a provenance `source:` line naming a paper nobody had opened, a "this is
probably Hull–White" in a comment — and an unread citation is worse than none, because it *looks*
like grounding. So:

- Every provenance `source:` line in the code names a **key** in this register.
- A key may be cited as a `source:` **only** if its status is `in-hand` or `worked-from`.
- A work that one of our sources merely *mentions* is recorded as `referenced-by:<key>` and
  **may not be a `source:`** until we obtain it. Its citation is transcribed from the source that
  names it, not from memory — so even an un-obtained reference is grounded in a document we hold.
- When an `in-hand` claim is checked against the source, that is noted; a figure that cannot be
  read to tolerance is marked as such (no code is fitted to it).

**`verify.py references` enforces the three rules above** (built at tranche 26b, after five drifts
earned it — see CLAUDE.md §7): every cited key resolves to a citable status; a key cited by any
`src/` module is `worked-from`; a `worked-from` key has at least one citer. So a status here cannot
silently diverge from what the tree actually cites. The gate reads these **header lines only** — keep
the prose body consistent by hand; it is not parsed, and a gate that parsed it could not be
falsified.

Status vocabulary: `worked-from` (held and built/oracled against) · `in-hand` (held and read) ·
`referenced-by:<key>` (named by a held source, NOT obtained — not citable) · `sought` (intend to
obtain).

---

## Held sources

### `BERHAN-2012` — `worked-from`
Damir Tewolde Berhan. *Pricing Inflation Derivatives: A survey of short-rate and market models.*
MSc thesis, KTH Royal Institute of Technology, 2012. TRITA-MAT-E 2012:13.
- **Locator:** `_stock/NewDocuments/pricing-inflation-derivatives.pdf` (60 pp).
- **Used for:** the inflation vertical — benchmark formulas (§1.2), Jarrow–Yildirim (ch 2), market
  models (ch 3–4), calibration (ch 5). Extraction + per-tranche oracle plan in
  `docs/papers/inflation-derivatives-berhan-2012.md`.
- **Oracle caveat:** empirical results are plots (Figs 1.1, 1.2, 5.1–5.10), not tables. Strong
  oracles are its closed-form identities; empirical values come from QuantLib and the stock's
  inflation tests, not from these figures.

---

### `JAECKEL-2002` — `worked-from`

Jäckel, P., *Monte Carlo Methods in Finance*. Wiley, 2002. 235 pp. Held locally at
`_stock/NewDocuments/Monte Carlo methods in finance (Jaeckel P.) ...pdf` (gitignored — copyrighted;
only this record is tracked).

**Project two (2026-11-04, Bernardo), after BERHAN closed.** A *methods* source, not a model/product
source: the authority is for the **algorithm** (RNG, Sobol direction numbers, the Brownian bridge,
variance-reduction identities, pathwise/likelihood-ratio Greeks, BGM/J under MC), not for a reference
**price** — Jäckel's own results are convergence diagrams, not tables (§14.4, *"A single number or a
convergence diagram?"*). So a tranche transcribes the **method** from Jäckel under §4, and oracles the
method's **output against this library's own closed forms** (Black, JY, MM-I, MM-II, the HW bond
option) by convergence-within-standard-error and the convergence *rate* — the closed-form anchor at
the top of §4's order. Chapters: 2/9 (generation, variates), 8 (low-discrepancy), 10 (variance
reduction, path construction), 11 (Greeks), 12 (BGM/J), 14 (diagnostics).

- **Worked by product-pull, not by chapter** — the chapter is where a method is read from, not a build
  order (§1); routes not taken are recorded in `docs/to_go_deeper.md` when the arc closes.
- **Stock base catalogued** in `docs/stock/mc_base_inventory.md` (an already-once-migrated MC stack —
  mined for content and its migration-hardening edge cases, never for structure, never imported).

**Status `worked-from`** (T15–T19: the MC spine, the LMM, the swaptions, variance reduction).

**A standing pattern — Jäckel states results and leaves load-bearing formulas unprinted.** Twice now a
spec said "transcribe from Jäckel" a formula the book does not contain: the inverse-CDF (§9.3 recommends
Acklam/Moro, prints neither — T15) and the control-variate variance factor `(1−ρ²)` (§10.3 — T19).
*Corrected at commit:* the draft said §10.3 "states the result, prints no derivation". It states
**neither**. All §10.3 says of the optimum is that eq 10.9's `β*` is the choice *"which minimises the
variance of `v̂_CV`"*; the resulting variance is never written, and `ρ` never appears in the section.
That distinction is the point of recording the pattern at all — "stated but not derived" would make
the factor transcribable from a printed line, and it is not. The substitute is to **derive it and assert it as a theorem to
machine precision** (T19: derived from eq 10.9, matched at 2e-17) — which is stronger than
transcription, being immune to a source misprint. Check whether a formula is actually printed before a
spec says "transcribe it."

**Source defects found (recorded, as `BERHAN-2012` §5.1's are):**
- **§12.8 eqs 12.38 vs 12.39 — an internal sign inconsistency.** Both print `+ p_i3`, but 12.39 pins
  the final slice to `(0,0,K)` and §12.7 requires "exercise iff the residual swap is in the money";
  the printed sign exercises a deep-OTM payer at the last date. Ruled by the finance (12.39 carries
  the exact financial content; 12.38's sign is the arbitrary convention), pinned by an oracle the
  printed variant fails (T18 §8). Survived publication because `p_i3` is fitted at every slice but the
  last, so the optimiser absorbs the sign — invisible wherever trained.
- **§12.8 eq 12.41 — the prescribed BFGS cannot train the stated objective.** The objective is an
  indicator-selected average, piecewise-constant in `p`, so a gradient optimiser terminates at
  iteration 0 (its FD gradient is a step-size artefact). Built with Nelder-Mead, which captures ~65%
  of the attainable gain; the deviation is oracled (T18 §9). A limitation of the *published method*,
  not of our reading.
- **§10.9 eq 10.76 — an appendix inconsistency (weaker: reaches no implemented formula).** 10.76
  prints `W_{t_k} = W_{t_j} + √(t_k−t_i)·z`, contradicting 10.77 three lines on, which forces
  `W_{t_i}`; with `W_{t_j}` the first increment is double-counted. Both renderings print it
  identically, so transcription-independence is powerless — the internal inconsistency is what catches
  it. Recorded for completeness and because §4 treats a derivation as a check on its result, so a
  broken check had to be repaired first (T21 §12.3); but it is **non-load-bearing** — the implemented
  bridge uses 10.33/10.34, which are consistent and independently re-derived.

---

### `MCINERNEY-ZASTAWNIAK` — `worked-from`

**Known gap, recorded so it is not re-attempted (tranche 72):** **§8.3 prints no CEV closed form.**
The section gives the SDE (8.5), the absorbing-boundary discussion, two *simulated histograms*
(Figures 8.1/8.2) and then pivots to the displaced-diffusion approximation as the practical answer.
There is no Schroder/Cox formula, no chi-squared argument triple and no worked example anywhere in
the book; its only other mention of the non-central chi-squared is a one-line taxonomy remark about
Cox-Ingersoll-Ross in ch 3. §4 forbids fitting a number to a figure, so the CEV caplet was sourced
from `BROCKHAUS-2016` §6.1 instead. What this text *does* still anchor is the absorbed fraction,
which it states in prose ("over 10%" at `F(0)=1%`, "typically low" at 5%) — reproduced structurally
by `cev_absorption_probability`.

McInerney, D. and Zastawniak, T., *Stochastic Interest Rates*. Mastering Mathematical Finance,
Cambridge University Press. 170 pp. Held locally at
`_stock/NewDocuments/Stochastic Interest Rates (Daragh McInerney, Tomasz Zastawniak).pdf`
(gitignored — the repo is public and the source is copyrighted; only this record is tracked).

**Acquired 2026-11-02 by Bernardo to unblock §5.3.1**, which BERHAN-2012 performs with a Hull-White
cap formula it declines to publish ("beyond the scope of this thesis"). It supplies exactly the two
pieces that were missing, and more:

| § | eq | what |
|---|---|---|
| 2.7 | — | the caplet payoff as `1 + τᵢK` puts on a zero-coupon bond, strike `(1+τᵢK)⁻¹` |
| 2.9 | **spread options** — Margrabe (exact), Kirk (approx) | **built T63** — both 1-D on a combined vol, `black_price` reused, no bivariate normal. One function serves both (Kirk's weight is 1 at `K=0`) |
| 3.5 | 3.16/3.17 | zero-coupon bond call `BC` and put `BP` under a deterministic log-volatility |
| 3.5 | 3.18 | `d±`, with `d₋ = d₊ − √ν(0,S)` |
| 3.5 | 3.19 | `ν(0,S) = D(S,T)²∫₀ˢ σ(u)²e^{−2α(S−u)}du` — the variance of `ln B(S,T)` |
| 3.5 | 3.20 | its constant-σ closed form, `σ²/(2α³)·(1−e^{−α(T−S)})²·(1−e^{−2αS})` |
| 3.5 | 3.21 | **the Hull-White caplet** `Cpl_i(0) = (1+τᵢK)·BP(0;Tᵢ₋₁,Tᵢ,(1+τᵢK)⁻¹)` |
| 3.5 | 3.26 | **the calibration scheme itself** — solve `Cpl_i(0) = Cpl^Black_i(0;σᵢ)` for the model's
implied caplet vol, then fit parameters to the market's spot vols. This is precisely the procedure
§5.3.1 describes and declines to specify, so it unblocks T10(b) as well as T10(a). |

**Internally consistent where it matters, and checked.** (3.19) reduces to (3.20) exactly when
`D(S,T) = (1−e^{−α(T−S)})/α` — which is our own `hw_beta` — so the book's `D` and the library's `β`
are the same object, verified rather than assumed.

**One rendering defect, resolved: `d₋`'s radical is dropped in BOTH `-raw` and `-layout`.** Each
prints `d₋ = d₊ − ν(0,S)` while `ν` is explicitly defined as a *variance*. This is the §5.1 shape
again — a formula the renderings reproduce identically and wrongly — so transcription independence
cannot settle it. Two things do: reading it as `d₋ = d₊ − ν` makes the **call price negative**
(−0.0049 at `B(0,S)=0.97, B(0,T)=0.94, K=0.985, ν=0.0036`), which no option can be; and
`d₋ = d₊ − √ν` reproduces our independently-oracled `black_price` to the last bit, because
**a Hull-White bond option *is* Black on the forward bond price `B(0,T)/B(0,S)` with total variance
`ν`, discounted by `B(0,S)`**. Note that put-call parity does **not** discriminate here — it is an
algebraic identity in `d±` whatever their relation, so it is a check that cannot fail on this
question.

**Status `worked-from`** since T10: the Hull-White cap/floor machinery was built against it, and it
is now cited by **ten** modules (Hull-White, both market models, the swap-rate vol, the swaption and
Bermudan engines, the cap calibration and the products). The earlier text said `in-hand` *"until T10
lands"* and was never updated when it did — one of the five drifts that earned the `references` gate
(tranche 26b).

**Also covers, for later units:** §1.7 yield-curve construction (T8's territory), ch 4 HJM and
Gaussian models, and **ch 5–6 the LIBOR and swap market models with their calibration** — which is
BERHAN-2012's ch 3 "Market Model I", so one acquisition also removes the source question from that
unit. **Ruled good-enough for MM-I (Bernardo, 2026-11-03):** BERHAN ch 3 publishes the MM-I formulas
in full and is the primary source; McInerney ch 5–6 is sufficient LMM background — the LMM is not
this project's main topic, so **no dedicated LMM text is to be acquired.** (Verified: §5.1 LMM, §5.2
Black's caplet, §5.6 Brace–Gatarek–Musiela, §5.9 swap market model, §6 implementation + calibration.)

**Project nine — the single-curve short-rate revisit (Bernardo, 2026-11-20).** The "later units" above,
pulled: a solid **single-curve** short-rate spine (the deliberate pre-crisis counterpoint to the
collateralised/multi-curve arc, project seven — these models generate ONE curve for both discounting and
forwarding, which is the single-curve design invariant exercised by construction). Already `worked-from`
since T10 (the HW cap machinery); this arc mines the chapters T10 left: ch 3 (Vasicek, Merton, two-factor
HW), ch 4 (the HJM framework), ch 8 (the volatility-smile short-rate models). Arc plan
`docs/stochastic_rates_arc.md`.

| ch | what | reuse / status |
|---|---|---|
| 3.3–3.4 | **Merton, Vasicek** — affine Gaussian short rates | new; the short rate is OU, so reuses `simulate_lgm_states`; `hw_beta` IS Vasicek's `B(t,T)`; Merton is Vasicek at κ→0 |
| 3.5 | Hull–White | **held** (T10) — the time-dependent-θ extended Vasicek, reused not rebuilt |
| 3.6 | **Bermudan swaptions in Hull–White** | new; reuses the T18 `exercise_boundary` in a short-rate model |
| 3.7 | **two-factor Hull–White (G2++)** | new; reduces to 1F HW, reuses `linalg` Cholesky (T63) |
| 4.1–4.5 | **the HJM forward-rate framework** (1-factor + drift condition, Gaussian, multi-factor, calibration) | new; **Gaussian HJM reduces to Hull–White** exactly |
| 8.1–8.4 | **volatility-smile short-rate models** — Normal/Bachelier, CEV, displaced-diffusion | new; extend the Black caplet to the smile; closed-form or semi-closed |
| 8.5 | stochastic volatility | **`out_of_method` candidate** — the section offers no closed form, so it cannot pass §4's oracle gate as the method stands; **not** come-back-same-method debt (re-binned at re-audit #8, F-8.1) |
| 5–6 | LMM / SMM | **held** (BERHAN MM-I + the Jäckel arc) — a second source, not rebuilt |
| 7 | exotic IR products (CMS, range accruals, trigger swaps, ratchet floaters) | `to_go_deeper` unless pulled |

---

### `ACKLAM-2000` — `worked-from`

Acklam, P. J., *An algorithm for computing the inverse normal cumulative distribution function*.
Held as `_stock/NewDocuments/An algorithm for computing the inverse normal cumulative distribution
function.htm` — a Wayback capture (2015-08-11) of `home.online.no/~pjacklam/notes/invnorm/`, the
author's own page. Acquired by Bernardo 2026-11-05 to unblock tranche 20.

**Why it was needed at all.** `JAECKEL-2002` §9.3 *recommends* this algorithm and Moro's and prints
neither (`[Ack00]` in its bibliography, the implementation *"included on the accompanying CD"* we do
not have). Tranche 15 hit that wall and deliberately deferred the inversion; this is the acquisition
that closes it. §7 forbids citing a key that is not in hand, so nothing could be transcribed until now.

**Status `worked-from`** since T20a: `foundation/distributions.inverse_norm_cdf` transcribes the
rational-approximation coefficients and refines them on our own `norm_cdf`.

**What the capture actually contains — checked, not assumed, because a capture can be partial:**
- the rational-approximation pseudo-code with **all 21 coefficients** (`a(1..6)`, `b(1..5)`,
  `c(1..6)`, `d(1..4)`) at full printed precision;
- the break-points `p_low = 0.02425`, `p_high = 1 − p_low`, and all three regional branches;
- the stated accuracy: *"relative error ... less than 1.15 × 10⁻⁹ in the entire region"*;
- the **refinement** step in full — one iteration of Halley's rational (third-order) method, which
  Acklam says *"gives full machine precision"*:
  `e ← Φ(x) − p` ; `u ← e·√(2π)·exp(x²/2)` ; `x ← x − u/(1 + x·u/2)`.

*(A first read reported the refinement body as missing from the capture. That was wrong — the text at
that point was the page's table of contents, not the section. The body is present and complete.)*

**The refinement needs `Φ(x)` to full precision, and we own that exactly** — `foundation.norm_cdf`
is `½(1 + erf(x/√2))` on `math.erf`. Acklam used Cody's FORTRAN `erfc` for the same purpose, so the
roles line up: his approximation supplies the starting point, our exact CDF supplies the refinement.
That is also why the algorithm is **not a correctness dependency**: tranche 20 inverts our own
`norm_cdf` by a guarded root-find as the primary, exact path, and Acklam's coefficients stay
*untrusted until they agree with it*.

### `CREPEY-2026` — `worked-from`

Crépey, S., *XVA Analysis: Probabilistic, Risk Measure, and Machine Learning Issues*. Chapman &
Hall/CRC Financial Mathematics Series, CRC Press, first edition 2026. 417 pp. ISBN 978-1-041-01420-1.
Held at `_stock/NewDocuments/XVA Analysis ... (Stéphane Crépey) ...pdf` (gitignored — copyrighted;
only this record is tracked).

**Project three (2026-11-08, Bernardo), after the Jäckel arc.** The XVA vertical — the one the
archived predecessor faked (the phantom `risk/xva.py` rows T24 is pruning) and this tree never built.

**Read honestly, this is a stretch for our method, and the arc is scoped around that.** Crépey is a
research monograph on the **cost-of-capital XVA frontier** (balance-sheet KVA, "fininsurance" measure,
anticipated BSDEs), and its numerics are overwhelmingly **neural-net regression + Monte Carlo** (neural
131×, regression 147×, BSDE 61×, expected-shortfall 61× across 417 pp; "closed form" appears 4×). Its
own validation is *twin Monte Carlo*, not a closed form, and its results are **figures, not tables** —
so §4's "no number off a plot" binds hard.

- **Used for: the XVA frame and its identities, which are requirement oracles — NOT concrete prices.**
  Crépey's CVA/FVA/KVA are abstract balance-sheet expectations and semilinear fixed-point equations:
  `CVA = E[(1−J)P⁺]`-type (Ch I §3), `FVA = γ(P−CVA−FVA)⁺` (the fixed point), `FV = CA−CL = CVA−DVA`,
  `FTP = ΔCVA + ΔFVA + ΔKVA`. These relations hold for all parameters by the finance, so they are
  exactly the §4 **requirement oracles** the spine is anchored on. The concrete closed-form numbers
  (survival curves, defaultable bonds, CDS, expected exposure) come from `CAPINSKI-ZASTAWNIAK` and the
  standard CVA integral, pointwise-transcribed — the two sources split as "requirement oracle +
  pointwise transcription, neither replaces the other" (§4).
- **Deferred wholesale** (→ `to_go_deeper` when the arc closes): the neural-net/regression chapters
  (IV–VII), KVA and the anticipated-BSDE machinery, the risk-measure/equilibrium/reverse-stress-test
  chapters (VIII–X), and HVA (XI). None is closed-form-oracle-able our way; the honest spine is the
  static/one-period elementary XVA, same move as BERHAN ch 1–5 and Jäckel's 9-of-14.

**Status `worked-from`** since T25 (v0.30.0): the CVA frame and eq (17)'s independence reduction are built and oracled in `risk/credit_valuation_adjustment.py`, extended by T26's exposure engine. Concept map + arc in `docs/xva_spine_arc.md`.

---

### `CAPINSKI-ZASTAWNIAK` — `worked-from`

Capiński, M. and Zastawniak, T., *Credit Risk*. Mastering Mathematical Finance, Cambridge University
Press. 202 pp. Held at `_stock/NewDocuments/Capiński, Zastawniak - Credit Risk.pdf` (gitignored —
copyrighted; only this record is tracked). Same series as `MCINERNEY-ZASTAWNIAK`.

**The closed-form credit authority the XVA spine pulls** — paired with `CREPEY-2026` (which assumes
this material as background, writing for readers who already know it). An elementary-to-intermediate
text with proofs, so its formulas are genuine pointwise-transcription anchors:

| ch | what | XVA spine use |
|---|---|---|
| 1 | Structural (Merton, barrier) models | closed-form default probability `Φ(−d₂)`-type |
| 2 | Hazard function, default time, risk-neutral probability | the **survival curve** `S(t)=exp(−∫λ)` (L1 credit market-data primitive) |
| 3 | Defaultable bond pricing with hazard + calibration, recovery schemes | closed-form defaultable bond; survival↔price calibration |
| 4 | Credit Default Swap | the **CDS par spread** closed form; survival↔CDS calibration |
| 5–6 | Hazard *process* (enlarged filtration) | the advanced tie-in to Crépey's filtration machinery — deferred |

**Status `worked-from`** since T25 (v0.30.0): ch 2's hazard function and survival probability are built and oracled in `market/survival_curve.py`, and the Bachelier positive part in `risk/credit_valuation_adjustment.py`.

---

### `BROCKHAUS-2016` — `worked-from`

Brockhaus, O. *Equity Derivatives and Hybrids: Markets, Models and Methods.* Palgrave Macmillan
(Applied Quantitative Finance), 2016. 304 pp. Held at `_stock/NewDocuments/Equity Derivatives and
Hybrids. Markets, Models and Methods (Oliver Brockhaus) (z-lib.org).pdf` (gitignored — copyrighted;
only this record is tracked).

**Project four (2026-11-10, Bernardo), after the XVA arc.** The first *equity* vertical — and the book
is almost a map of the tree we already hold, with equity on top: ch 10 (fixed income — Hull-White,
CIR, HJM, Markov Functional) overlaps our rates; ch 12 (credit — reduced-form/structural) our survival
curve; ch 14 (CVA, FVA, wrong-way risk) our just-built XVA spine; ch 17 (Monte Carlo — RNG, Brownian
path construction, variance reduction, Greeks) the Jäckel arc; ch 18 ("Gauss" — Black-Scholes, barrier
formulas, useful integrals) is a **closed-form appendix**, the oracle anchors. So the equity spine
reuses the tree heavily, and the **hybrids** (ch 11 equity-rate, ch 13 defaultable equity, ch 14
equity CVA) are where equity ties the whole library together.

**Method-fit: strong** (unlike Crépey). Equity derivatives are rich in closed forms — Black-Scholes
(ch 18.2), the variance-swap static replication (ch 5.8), Dupire local volatility (ch 6.1), the
affine/Fourier characteristic-function vanilla (ch 16) — each an oracle anchor. The exotic zoo (barriers/
cliquets/asians/autocallables ch 3, implied-vol dynamics ch 7, correlation/copulas ch 8–9, FX ch 15) is
`to_go_deeper`, pulled by interest.

**Used for:** the equity-and-hybrids arc (`docs/equity_hybrids_arc.md`), and — from tranche 72 —
**§6.1's CEV closed form**, which is the only printed CEV option formula in any held source. **Status
`worked-from`**
since T30 (v0.36.0): ch 5.1-5.3's dividend forward is built and oracled in `market/equity.py`, and
the European equity option off that forward in `engine/equity_vanilla.py`. *(The status was flipped
by the `references` gate catching it mid-tranche — cited by four modules while still `in-hand` —
which is the gate built at T26b doing its job on its first real opportunity, in the tranche that
created the drift rather than five tranches later.)*

---

**Source defects found (recorded, as `BERHAN-2012` §5.1's and `JAECKEL-2002` §12.8's are):**
- **§6.1 p.79 — the printed Cox (1996) CEV call formula is wrong in two places, identically in
  `-raw` and `-layout`.** As printed:

      C(K,T) = S0 Q(2w; 2 + 1/(1-rho), 2z) - K [1 - Q(2w; 2 - 1/(1-rho), 2z)]

  (a) Both chi-squared terms take `2w` as the argument and `2z` as the noncentrality; the two must
  **swap** them — the second term is `Q(2z; 1/(1-rho), 2w)`. (b) The first term is missing its
  complement: it is the **survival**, `1 - Q(2w; ...)` in the printed notation. Because both
  renderings are faithful to the same wrong print, transcription independence is powerless (§4);
  what condemns it is three **requirement** oracles, each asserted failing on the printed form in
  `tests/L4/test_cev_caplet.py`: `K -> 0` must give the forward and gives roughly minus the strike;
  `K -> inf` must give zero and gives the forward; and at **`rho = 0.5`** — the value
  `MCINERNEY-ZASTAWNIAK` p.148 calls market practice — the printed second term asks for **zero
  degrees of freedom** and is simply undefined, so the formula as printed cannot be evaluated on the
  whole upper half of the range it is stated for. The corrected form was **ruled by Monte Carlo**
  (600k paths, Euler with absorption, under 1 SE at three strikes), confirmed against the section's
  own derivation of the forward's law, and is the standard Cox/Schroder arrangement — see
  `COX-1996` and `SCHRODER-1989` below. **The project's first source *correction* rather than a
  choice between two renderings**; the three oracles are what make it a correction and not a guess.
- **§6.1 eq (6.10) — a dimensional inconsistency against the derivation two steps above it.** The
  printed drifted-case substitution carries a spurious `/T`: `C(K,T) = e^(-qt) Ctilde(K/e^((r-q)t),
  sigma^2 (e^(2(...)T) - 1)/(2(...)T))`, whose second argument is the accumulated variance divided
  by `T`, while the space-time change printed on the same page gives
  `tau_T = int_0^T sigma^2 e^(2(...)s) ds` with no `/T`. The derivation step wins (§4: an earlier
  step is an independent check on the final one) and the dimensions agree only without it.
  **Not load-bearing here and so not pinned by an oracle:** a caplet's forward is already a
  martingale under its own forward measure, so eq (6.10) never arises — tranche 72 uses the
  driftless eq (6.9) form directly, with the total variance as its clock. Recorded in case a
  *drifted* CEV (an equity or FX CEV option) is ever pulled.

### `OKANE-2008` — `worked-from`

O'Kane, D., *Modelling Single-name and Multi-name Credit Derivatives*, Wiley Finance Series, 2008.
~450 pp. Held at `_stock/NewDocuments/Modelling Single-name and Multi-name Credit Derivatives …
(Dominic OKane).pdf` (gitignored — copyrighted; only this record is tracked).

**The dedicated credit-derivatives authority — project five (Bernardo, 2026-11-14).** The canonical
practitioner text for the CDS and the correlation products. It pulls the credit work the XVA arc left
deferred: the **CDS survival-curve bootstrap** T25 explicitly parked ("inverting CDS quotes for λ is a
curve bootstrap, arriving with the product that needs it"), the CDS premium/protection legs, and the
multi-name Gaussian-copula/tranche machinery. A **second source** for objects we already hold — the
survival curve (its ch 3 vs `CAPINSKI-ZASTAWNIAK` ch 2) and the CDS par spread (its ch 6 vs CZ ch 4) —
so a cross-source independence check is available where the two render the same formula (§4's
same-formula caveat applies: confirm they agree before treating one as the other's transcription).

Candidate map (ruled per-tranche at build; the arc plan is `docs/okane_credit_arc.md`):

| ch | what | reuse / status |
|---|---|---|
| 2 | Libor discount curve | our discount curves (T1/T8) |
| 3 | hazard / Cox / **credit triangle** `S=λ(1−R)` | our `SurvivalCurve` (T25); the triangle is a new closed-form anchor |
| 5–6 | the **CDS**: premium leg `S·RPV01`, protection leg `(1−R)∫Z·λQ dt` | the protection leg **is** the CVA integral shape (T25) — cross-product reuse |
| 7 | **CDS survival-curve bootstrap** | the T8 nominal-bootstrap round-trip pattern, applied to CDS |
| 8–9 | CDS risk (CS01), forward CDS, **CDS swaption** | Black (T5) on the forward spread, RPV01 the annuity |
| 13 | the **one-factor Gaussian copula** / latent-variable model | Φ/`norm_cdf`; conditional independence |
| 15–16 | default baskets, the **LHP tranche** closed form | Φ, Φ₂ (T36b); the Jäckel MC (T15–22) for baskets |
| 18, 20 | the recursive loss distribution, **base correlation** | the market-standard tranche machinery |
| 4, 10–11, 19, 21–23 | bonds/asset swaps, indices + index options, implied/compound corr, copula-skew models, CPPI/CPDO/CDO², dynamic models | `to_go_deeper`; ch 23 (intensity-gamma, affine-jump) a possible `out_of_method` candidate |

**Status `worked-from`** since T38 (v0.45.0): ch 3's credit triangle and ch 6's
premium leg / risky PV01 (eqs 6.2/6.4) are built and oracled in
`engine/credit_default_swap.py`.

### `GATHERAL-2006` — `worked-from`

Gatheral, J., *The Volatility Surface: A Practitioner's Guide*, Wiley Finance Series, 2006. 210 pp.
Held at `_stock/NewDocuments/Gatheral - The Volatility Surface, A Practitioner's Guide copy.pdf`
(gitignored — copyrighted; only this record is tracked).

**The dedicated volatility-surface authority — project six (Bernardo, 2026-11-16).** The canonical
practitioner text on the implied-volatility surface. Much of its model layer the equity arc already holds —
local volatility (Dupire, T32), the Heston model by Fourier inversion (T33), the variance swap by static
replication (T31), SABR (`models/sabr_model.py`), the implied-vol surface (T9/T32) — so this arc pulls what
Gatheral adds *on top*: the **SVI** parameterization of the surface (ch 3, eq 3.20) and its no-arbitrage
conditions, the **surface asymptotics** (ch 7, Lee's moment formula / wing slopes), **volatility
derivatives** beyond the variance swap (ch 11, the volatility swap's convexity, options on variance), and
**stochastic-volatility-plus-jumps / Bates** (ch 5) extending T33. **Worked from at T46:** ch 3's eq 3.20
(the SVI slice, `models/svi.py`) and ch 7 pp. 97–98 (Lee's moment formula, `β*, α* ∈ [0,2]`, which is what
pins eq 3.20's radical — see `models/svi.py`'s header for why neither `pdftotext` rendering shows it).
**What it does NOT carry:** no butterfly-arbitrage condition for SVI — the text gives only the calendar
condition and says butterfly is *"relatively easy to eliminate"*; the rigorous `g(k) ≥ 0` is
Gatheral–Jacquier (2014), **not held** (`docs/to_go_deeper.md`), so T46 derives the density from
Breeden–Litzenberger instead. A **second source** for Heston, local
vol and the variance swap we already hold — so a cross-source independence check is available where it and
`BROCKHAUS-2016` render the same object (§4's same-formula caveat applies).

Candidate map (ruled per-tranche at build; the arc plan is `docs/gatheral_vol_surface_arc.md`):

| ch | what | reuse / status |
|---|---|---|
| 1–2 | local vol, Heston | **already held** (T32, T33) — not rebuilt |
| 3 | the **SVI** surface parameterization (eq 3.20) + no-arbitrage (calendar, butterfly/wings) | new; reuses the implied-vol surface (T9/T32) + total variance (T31) |
| 7 | **volatility surface asymptotics** — Lee's moment formula, the wing slopes ≤ 2 | new closed-form reductions; cross-check vs Heston (T33) and SABR |
| 11 | **volatility derivatives** — the vol swap's convexity, options on variance | extends the variance swap (T31) |
| 5 | **SV + jumps / Bates** — the jump term in the characteristic function | extends the Heston Fourier pricer (T33), reduces to it at zero jump intensity |
| 4, 6, 8–10 | Heston-Nandi, default risk (Merton jump-to-ruin), skew dynamics, barriers, cliquets | `to_go_deeper`; some re-tread (ch 6 vs T35) or MC-exotic (ch 9–10) |

**Status `in-hand`** (held, TOC + the SVI section ch 3 and the Lee asymptotics ch 3/7 read, 2026-11-16);
flips to `worked-from` at the first build (T46, the SVI surface). The full butterfly/no-arbitrage SVI
conditions in their rigorous form are Gatheral–Jacquier (2014), which the 2006 text may not carry — ruled
at T46's build whether the density-non-negativity requirement oracle suffices or that paper must be
obtained (a `to_go_deeper` "acquire source" note if so).

### `LICHTERS-2015` — `worked-from`

Lichters, R., Stamm, R. & Gallagher, D., *Modern Derivatives Pricing and Credit Exposure Analysis:
Theory and Practice of CSA and XVA Pricing, Exposure Simulation and Backtesting*, Palgrave Macmillan,
2015. 491 pp. Held at `_stock/NewDocuments/Modern Derivatives Pricing and Credit Exposure Analysis_ …
.pdf` (gitignored — copyrighted; only this record is tracked).

**Worked from at T54:** §3.5 "Building the forward projection curves" (the swap-stripping relation
`c·A^ON = Σ f^δ(t_{i−1},t_i)·δ_i·df^ON_i`, the projection recursion) and §1's eqs (1.1)/(1.2) for the
forward and its PV identity — the latter being the telescoping test that separates the two eras.

**The dedicated collateralised-pricing authority — project seven (Bernardo, 2026-11-18).** The practitioner
text on **CSA / collateralised discounting** and credit exposure. It pulls the one layer the tree has
never had — the post-crisis reality that a collateralised trade discounts at the **collateral rate**
(OIS), multi-curve (OIS discount + separate forward-projection curves) — and extends the XVA arc's
exposure machinery with collateralisation. A **second source** for objects we hold (CVA/DVA vs
`CREPEY-2026`, Jarrow–Yildirim vs `BERHAN-2012`, the short rate vs our Hull-White), so cross-source checks
are available where they render the same object (§4's same-formula caveat applies).

Candidate map (ruled per-tranche at build; the arc plan is `docs/modern_discounting_arc.md`):

| ch | what | reuse / status |
|---|---|---|
| 1–2 | pre-crisis discounting, curve building | our single-curve bootstrap (T1/T8) — the machinery reused |
| 3 | **OIS / multi-curve bootstrap** | new; reuses T8's par-reprice round-trip |
| 4 | **global / cross-currency collateralisation**, the cross-currency basis | new (covered-interest-parity oracle) |
| 5 | **CSA discounting** — collateral rate, collateral choice, thresholds | new; **reduces to single-curve at OIS=Libor** (the pre-crisis limit) |
| 7–9 | CVA / DVA, single-trade CVA | **held** (the XVA arc T25–T29) — reused, cross-product oracle, not rebuilt |
| 11 | the **Linear Gauss-Markov (LGM)** short-rate model | new; **LGM is a reparameterisation of Hull-White** — exact reduction |
| 13.2 | Jarrow–Yildirim inflation | **held** (inflation arc) |
| 6, 12, 13.3, 14, 5.6 | hedge accounting, FX/equity RFE, Dodgson–Kainth, initial margin | `to_go_deeper` |
| backtesting (+ `kerkhof.pdf`) | exposure-model backtesting | **`out_of_method` candidate** — statistical/historical validation, no closed-form oracle |

**Status `in-hand`** (held, TOC + ch 1–3 read, 2026-11-18); flips to `worked-from` at the first build (T54).

---

### `CLARK-2014` — `worked-from`

Clark, I. J., *Commodity Option Pricing: A Practitioner's Guide*, Wiley Finance, 2014. 343 pp.
ISBN 978-1-119-94451-5. Held at `_stock/NewDocuments/Commodity Option Pricing A Practitioners Guide.pdf`
(gitignored — copyrighted; only this record is tracked).

**The commodities authority — project eight (Bernardo, 2026-11-19).** The practitioner text on
**commodity option pricing**. Its quant spine is Ch 2 ("Commodity Mathematics and Products"), which is
unusually closed-form-dense: the commodity forward (cost-of-carry + convenience yield), Black-76 on
futures, Asian options (geometric exact + arithmetic moment-matched), spread options (Margrabe + Kirk),
and mean-reverting (Schwartz) models. Chapters 3–9 are market-by-market (precious/base metals, energy,
power, agriculturals, alternative) — conventions and structure, largely `to_go_deeper`/`out_of_method`.
A **second source** for Black-76 (vs our `black.py`/`BERHAN-2012`) and the OU short rate (vs our
Hull-White/LGM), so cross-source checks are available where they render the same object (§4's
same-formula caveat applies — a different parameterisation is its own transcription target).

Candidate map (ruled per-tranche at build; the arc plan is `docs/commodity_arc.md`):

| ch | what | reuse / status |
|---|---|---|
| 2.1–2.2 | spot/forward/futures, cost-of-carry, **convenience yield**; **Black-76** | forward new (L1/L2); Black-76 **held** — `models/black.py` IS Black-76-on-a-forward (`black_price`), reused not rebuilt |
| 2.5 | European options on spot/futures | the commodity option product + engine reusing `black_price` |
| 2.7 | **Asian options** — geometric (exact closed form), arithmetic (Turnbull–Wakeman) | new; geometric exact, arithmetic a moment-match with an MC companion (reuses `mc_estimator`) |
| 2.9 | **spread options** — Margrabe (exact), Kirk (approx) | new; both 1-D on a combined vol, so **reuse `black_price`** — no bivariate normal needed |
| 2.10.1 | **mean-reverting (Schwartz one-factor)** | new; OU log-spot — **reduces to the LGM/OU simulator** (`simulate_lgm_states`), GBM limit at κ→0 |
| 2.6 | American (Barone-Adesi-Whaley), lattice | `to_go_deeper` — its natural oracle is a binomial lattice the tree would build only for it |
| 2.10.2–2.10.3 | multi-factor, stochastic convenience-yield models | `to_go_deeper` |
| 3–9 | market-specific: vol surfaces, seasonality, electricity spikes, weather/freight | `to_go_deeper` / **`out_of_method`** (quotes-and-plots, structural/statistical, no closed-form oracle) |

**Status `worked-from`** (held, full TOC read 2026-11-19; **built and oracled against at T60**).

**Worked from at T60:** §2.1.2 (forwards; contango and backwardation), §2.2.3 (the convenience-yield
Black–Scholes PDE 2.19, where `r^f` enters the convection term and not the forcing term — which is
why a convenience yield moves the forward and not the discounting) and §2.2.6 (eqs 2.55/2.56, the
cost-of-carry forward, transcribed in both renderings). The same page prints the **pre-substitution
expectation** `F = S₀·E[exp(σW_T + (r−δ−½σ²)T)]`, an earlier derivation step used as an independent
check on the final form (§4) — evaluated by Gauss-Hermite quadrature rather than transcribed twice.

**Worked from at T61–T63 (ch 2.7, 2.9).** §2.7.3 the Kemna–Vorst geometric average — whose variance the
source prints **three** ways (the `min(tᵢ,tⱼ)` double sum, its sorted single-sum evaluation, and a closed
form in `n` alone), all three required to agree; §2.7.4 Turnbull–Wakeman, whose `M₂` is printed **two** ways
and whose measured error curve is that tranche's deliverable; §2.9.1/2.9.2 Margrabe and Kirk, both collapsing
to 1-D Black-76 on a combined volatility. The recurring value of this source is that it **derives** rather
than quotes, so each result arrives with earlier steps that serve as independent checks — §4's
derivation-step rule, available three tranches running.

---

### `KROESE-TAIMRE-2011` — `in-hand`

Kroese, D. P., Taimre, T. & Botev, Z. I., *Handbook of Monte Carlo Methods*, Wiley Series in Probability
and Statistics, 2011. 772 pp. Held at `_stock/NewDocuments/Handbook of Monte Carlo Methods …pdf` (gitignored
— copyrighted; only this record is tracked).

**The canonical Monte-Carlo reference — the extension to `JAECKEL-2002` (Bernardo, 2026-11-22).** Where Jäckel
is the finance-MC methods source (Sobol, the Brownian bridge, variance-reduction identities, BGM/J), this is
the general, derivation-rich handbook that sits behind it: a reference to **pull a chapter from when the MC
arc is extended**, not an arc of its own. Held, not yet built against; flips to `worked-from` at the first
tranche that cites it. **It is a reference textbook (canonical algorithms with derivations and forced
properties), not a results-and-figures paper — so it clears the oracle gate the Miller anthology does not.**

Chapter map (ruled per-tranche at build; most of our MC foundation already exists and would be *cross-checked*
against it, not rebuilt):

| ch | what | relation to the tree |
|---|---|---|
| 1–2 | uniform & **quasirandom** (Sobol) generation | **held** (`rng`, `sobol`, the Jäckel arc) — a second source, §4 same-object caveat |
| 3–4 | random-variable generation, distributions | **held** (`distributions`, `noncentral_chisquare`, `inverse_norm_cdf`) |
| 5 | random-**process** generation (Markov chains, SDE paths) | extends `mc_path_engine` (path construction) |
| 9 | **variance reduction** — antithetic, control variates, **importance sampling**, **quasi-MC** | extends T19/T21; **importance sampling is NEW** (not in the tree), oracle-able by the unbiasedness requirement + the measured variance ratio (the T19/T21 discipline) |
| 10 | **rare-event simulation** (importance sampling for tails) | new; relevant to deep-OTM / tail pricing — oracle-able by unbiasedness + the rate |
| 11 | **estimation of derivatives** (pathwise / likelihood-ratio Greeks) | extends the Jäckel-ch-11 Greeks; oracle-able against our closed-form Greeks |
| 6–8, 12–13 | MCMC, estimation, randomized optimization, cross-entropy | mostly `out_of_method` / not pricing (statistical, optimisation) |

### `MILLER-EDELMAN-APPLEBY-2007` — `in-hand` (extra, per-chapter)

Miller, J. J. H., Edelman, D. & Appleby, J. A. D. (eds), *Numerical Methods for Finance*, Chapman & Hall/CRC
Financial Mathematics Series, 2007. 312 pp. Held at `_stock/NewDocuments/Numerical methods for finance …pdf`
(gitignored — copyrighted; only this record is tracked).

**An extra reference, kept handy and mined PER CHAPTER when a chapter feeds a topic we already have
(Bernardo, 2026-11-22) — not an arc.** It is a **research anthology** (15 standalone conference papers), so
**most of it fails the oracle gate**: results are empirical, statistical, ML, or validated by the author's
own convergence figures / twin Monte Carlo — which §4 ("no number fitted to a figure") and §1 (twin-MC-only
is `out_of_method`) forbid. The usable minority, flagged so a future tranche can pull one:

| ch | what | feeds | oracle |
|---|---|---|---|
| 12 | **Multilevel Monte Carlo** (Giles) | the **MC arc** (`mc_estimator`, `mc_path_engine`) | **oracle-able** — unbiasedness (telescoping sum hits plain-MC price, the T19 pattern) + the complexity/variance-decay **rate** (T25/T21, measured not plotted); MLMC is standard enough to transcribe against the canonical method |
| 2 | high-dimensional American options | the **Bermudan** work (T18 `exercise_boundary`) | oracle-able via the structural foresight+suboptimality bracket (T18/T19) |
| 11 | performance-dependent options | the spread/basket work | maybe — if it reduces to a multivariate-normal closed form |
| 1, 3, 5–10, 13–15 | risk measures, allocation, HPC, forecasting, VaR self-similarity, Kalman, EDDIE | — | `out_of_method` (empirical / statistical / ML, no pricing closed form) |

**Mining rule:** when a tranche's topic matches a flagged chapter, **read the chapter first** and confirm it
prints a transcribable method with a reachable anchor (the T72 lesson — a chapter can be figure-only), then
cite it as a *second* source for the relevant section; it flips to `worked-from` only when a `src/` module
cites it.

### `KWOK-ZHENG` — `worked-from`

Kwok, Y. K. & Zheng, W., *Pricing Models of Volatility Products and Exotic Variance Derivatives*, Chapman &
Hall/CRC Financial Mathematics Series. 283 pp. Held at `_stock/NewDocuments/Pricing Models of Volatility
Products and Exotic Variance Derivatives …pdf` (gitignored — copyrighted; only this record is tracked).

**The variance/volatility-DERIVATIVES authority — added by Bernardo, 2026-11-22.** Distinct from
`GATHERAL-2006`, which is the vol *surface*; this is the *products on* variance/volatility and their pricing.
A proper textbook (replication identities, closed forms, char-function reductions), so an **arc candidate**,
not just a per-chapter mine — it **extends** the variance/vol swaps and the Heston Fourier machinery the tree
already holds. Held, not yet built against; flips to `worked-from` at the first tranche that cites it.

Chapter map (ruled per-tranche at build; heavy reuse of project-six machinery):

| ch | what | relation to the tree |
|---|---|---|
| 1 | derivatives on realized variance; **replication of variance swaps**; VIX as model-free vol | the **base is held** (T31 variance-swap static replication, T48 vol swap) — a second source + the discrete-vs-continuous correction |
| 2 | model pricing (Variance Gamma, continuous realized variance) | extends the Heston/Lévy char-function work (T33/T49) |
| 3 | **VIX derivatives** — the VIX²/variance-swap-rate relation under jumps, consistent & direct models | new; prices off the variance-swap rate + the char function |
| 4 | **swaps on discrete variance/vol** — Heston & 3/2 model, **gamma & corridor** variance swaps, time-changed Lévy | extends T31/T48; **generalized variance swaps** (gamma/corridor/conditional) are new, oracle-able by reduction to the vanilla variance swap in the degenerate corridor |
| 5 | **options on discrete realized variance** — lognormal, **normal approximation**, partially-exact-bounded | new; a moment-matching approximation with an MC companion — the **Turnbull–Wakeman pattern** of the Asian arc (T62), measured error the deliverable |
| 6 | **timer options** (perpetual) | new; the first-passage-on-accumulated-variance exotic — oracle-able by the perpetual closed form |

**Project ten — the vol-derivatives arc (Bernardo, 2026-11-22; scope: bigger ~6).** Opened at v0.75.0; flips
to `worked-from` at T74. Reuse-heavy throughout: **T48 already built `integrated_variance_transform`** (the
Heston Laplace transform of integrated variance), which is the engine behind VIX, options on realized
variance, and timer options; `fair_variance`/the `1/K²` strip (T31) is the base for the generalized swaps; the
Heston char function (T33) and the T62 moment-approximation+MC-companion discipline carry over. Arc plan
`docs/vol_derivatives_arc.md`; queue T74 (generalized variance swaps) → T75 (options on realized variance) →
T76 (VIX derivatives) → T77 (3/2-model variance swap) → T78 (time-changed Lévy + discrete-vs-continuous
convergence) → T79 (timer options).

**Status `worked-from`** since tranche 74 (v0.76.0): §1.4's Proposition 1.2 (the gamma swap,
eq 1.24) and eq (1.26) (the corridor variance swap) are transcribed and oracled in
`engine/variance_replication.py`; §1.3's two payoffs in `products/generalized_variance_swap.py`.
**One correction to the arc plan, recorded here because it will mislead the next reader otherwise:**
the static replication of the generalized swaps is **§1.4**, not ch 4.3 — ch 4.3 is *Moment
generating function methods*, a model-based route (the Heston MGF) that T75/T76 will use. The T74
spec cited ch 4.3 for the replication; the build read §1.4.

**One sourcing correction (T76, recorded at 0404):** VIX² as the variance-swap rate is **§1.5.2 /
eq (1.31)** — which *is* T31's strip verbatim — **not** §3.1.1, which is Carr–Wu's *jumps* correction
(how VIX² and the variance-swap rate **differ** under jumps, out of scope for pure-diffusion Heston).
The futures/options are **§3.1.2** (eqs 3.6/3.7a), the affine pricing **§3.2.1**. The T76 spec's §3.1.1
pin was the §1.4-vs-ch-4.3 shape again.

**Source defects found (recorded, as `BERHAN-2012` §5.1's, `JAECKEL-2002` §12.8's and `BROCKHAUS-2016`
§6.1's are):**
- **§3.2.2 eq (3.30) — the 3/2-model Laplace transform of integrated variance fails `L(0)=1`;
  DIAGNOSED and RESOLVED by acquiring the source paper (`ZHAO-CUI-FLORESCU-2018`, now held).** As
  printed in the book, `L(0) = 4.32`/`4.19` across renderings and the renderings disagree on `α` —
  it looked irreparable (T77 step-1, 0407). The fix came from the attributed source: eq (3.30) cites
  **Zhao et al. (2018)**, which Bernardo then acquired (SSRN 3043402). The paper's **Proposition 3
  (the canonical Carr–Sun (2007) form), eq (19)**, writes the confluent hypergeometric as
  `M(α, γ, −2/(σ²y_T))` — **parameters α, γ first, argument last** — and satisfies **`L(0) = 1`
  EXACTLY** (measured 1.000000000000, and exact by structure rather than numerically close: α's
  grouping makes `α(0) = 0`, whence `M(0,γ,z) = 1`, the power `(·)^0 = 1` and `Γ(γ)/Γ(γ) = 1`, so the
  requirement oracle is an equality and not a band — which is also what *pins* α's grouping, the
  renderings having disagreed about it). The two routes then agree to **8.1e-10**, which is the
  separate number: ladder against transform, not `L(0)` against 1. The book's `₁F₁(−2/σ²U; α, γ)` put the **argument first**; read in
  that order it diverges (→∞). So the defect is the **book's argument-order rendering of `M`**, not
  the mathematics — the §5 "acquire the source and reconcile" move working exactly. T77 transcribes
  eq (19) from the held paper as the **oracle** cross-check and prices on the **reciprocal-CIR
  Poisson-ladder** (route c) in production, the two agreeing (ruled 0409). *(The 3/2 SDE eq (2.75d)
  and the reciprocal map eq (4.35) were sound and held throughout — only the book's eq (3.30)
  rendering was defective.)*
- **§4.4 Prop 4.8 — the `Kv(∞)` time-changed-Lévy fair strike prints the SECOND MOMENT where the
  VARIANCE is required (the arc's fourth source defect, T78 step-1).** Prop 4.8 prints
  `Kv(0,T;∞) = −(1/T)φ_L⁽²⁾(0)·E[Z_T−Z₀]`, and with the book's own (4.84) `−φ_L⁽²⁾(0) = E[L₁²]` — the
  second moment. But **Prop 4.2 + Prop 4.1** give `Kv(∞) = Q_L(κ_L(1)−κ_L′(0)) = κ_L″(0) = var(L₁)`.
  They differ by `(E[L₁])²` (measured 20% / 5% / 24,245% apart on three VG sets; equal only at
  `E[L₁]=0`). **Ruled `var(L₁)` by requirement identity (T13 derivation-step-discrepancy rule):** a
  variance-swap strike prices *quadratic variation*, which is **drift-free** (`[L,L]` ignores drift), so
  the strike cannot depend on `E[L₁]`; Prop 4.1's proof says `E[L,L]₁ = var(L₁)`, Props 4.9/4.10 use
  `ψ_L` not `φ`, Brownian agrees both ways — so Prop 4.8's `φ⁽²⁾` is the typo. **Production computes
  `Kv(∞) = var(L₁)·E[Z_T]/T`** (T78, ruled 0419); Prop 4.8's printed form is not used. *(Note: the
  "multiplier route" `Q_L(κ(1)−κ′(0))` and the `var(L₁) = κ″(0)` route are an **algebraic identity**,
  not two independent computations — oracle 1's independence comes from the MC companion, not from
  running both; the T17 "confirm two forms are different objects" lesson, 0417(3).)*
- **Table 4.3 CGMY row — the multiplier printed with no `Y`/`Γ` is right only at `Y=0` (= Variance
  Gamma); its generality claim is wrong (minor, T78 step-1).** Derived against the Lévy density: it
  matches to machine precision at `Y=0`, and is off 2.8% at `Y=0.5`, 5.9% at `Y=1.2`. The formula is a
  correct VG multiplier mislabelled as general CGMY; T78 builds the VG case, so no production impact.

### `ZHAO-CUI-FLORESCU-2018` — `worked-from`

Zhao, Z., Cui, Z. & Florescu, I. (2018). *VIX derivatives valuation and estimation based on closed-form
series expansions.* International Journal of Financial Engineering, 5(2), 1850020 (18 pp). Held at
`_stock/NewDocuments/ssrn-3043402.pdf` (the SSRN preprint, 26 pp; gitignored — only this record is tracked).

**Acquired by Bernardo, 2026-11-23, to resolve the `KWOK-ZHENG` eq (3.30) defect** — eq (3.30) attributes
itself to this work, and the book's rendering of the 3/2-model integrated-variance Laplace transform failed
`L(0)=1` (the defect log above). The paper's **Proposition 3** (the canonical **Carr–Sun (2007)** form; the
paper restates it, we cite the held paper, not the un-held Carr–Sun) carries the correct transform,
`M(α, γ, −2/(σ²y_T))` — **confluent hypergeometric, argument last** — which satisfies `L(0)=1` exactly; the
book put the argument first. The paper's method is a **Hermite series expansion** for VIX derivatives under
general stochastic-vol models (Heston, mean-reverting CEV, 3/2); for T77 only **Proposition 3** (the 3/2
transform) is load-bearing, used as the **oracle** transcription cross-check of the reciprocal-CIR route.
*(The Hermite-series VIX-pricing method itself is a `to_go_deeper` candidate, not pulled.)*

**Status `worked-from`** since tranche 77 (v0.79.0): **Proposition 3 / eq (19)** — the canonical
Carr–Sun (2007) Laplace transform of the 3/2 model's realized variance — is transcribed as T77's
**oracle** (`tests/L4/test_three_halves.py`), and `KWOK-ZHENG` §4.2.2 eq (4.35)'s reciprocal-CIR map
carries production. **Acquired for a specific reason and it paid for itself immediately:** the book's
eq (3.30) is a mis-rendering of this paper's eq (19) (argument order), and reading the paper is what
turned an apparently irreparable source defect into a resolved one. Cited for eq (19) only; the
paper's own Hermite-expansion method is not built against.

**Note on the attribution chain.** Eq (19) is credited to **Carr and Sun (2007)**, which is *not*
held — so it is this paper, the held one, that is cited (§7: no reference by inference). The
`Carr–Sun` name appears in the provenance as the result's origin, never as a `source:` key.

### `OVERHAUS-2002` — `in-hand` (a second equity source; the weakest of the shelf)

Overhaus, M., Ferraris, A., Knudsen, A., Milward, R., Nguyen-Ngoc, L. & Schindlmayr, G. (the Deutsche Bank
Global Quantitative Research team), *Equity Derivatives: Theory and Applications*, Wiley, 2002. 238 pp. Held
at `_stock/NewDocuments/Equity Derivatives Theory and Applications …pdf` (gitignored — copyrighted; only this
record is tracked).

**Added by Bernardo, 2026-11-22 — a SECOND equity source, and recorded honestly as the thinnest of the four
recent additions.** It overlaps heavily with `BROCKHAUS-2016` (project four, the equity-hybrids arc) and the
vol arc, it is older (2002), and roughly a third of it (chs 6–8) is dated pricing-over-the-Internet software,
which is `out_of_method` and out of scope (§0). Its genuinely-useful content is narrow:

| ch | what | relation to the tree |
|---|---|---|
| 1–2 | mathematical foundation; hedging in incomplete markets | theory; a cross-check, little to build |
| 3 | **Lévy processes + the Heston Jump-Diffusion model** | a **second source** for Heston (T33) and SVJ/Bates (T49) — cross-check where it renders the same object (§4 same-formula caveat), and a more general Lévy framing |
| 4 | a stochastic-volatility model | extends Heston (T33) |
| 5 | **convertible bonds** + convertible-bond asset swaps | **new to the tree** — the one genuinely-new pull, but convertibles are **tree/PDE-priced** (American conversion), so harder to oracle cleanly than a closed form; a `to_go_deeper`/careful-arc question, not a reduction-rich spine |
| 6–8 | pricing/hedging analytics "over the Internet" | `out_of_method` / out of §0 scope (software delivery, not a numerical method) |
| 9 | a portfolio & hedging simulation engine (discrete hedging, hedging in Heston) | a simulation study — figures, mostly `out_of_method` |

**Use:** primarily a **cross-check** source for the jump-diffusion / Lévy stochastic-vol work we already hold
(a different team's rendering of Heston/Bates), and the convertible-bond reference **if** a convertibles arc
is ever pulled — with the honest caveat that convertibles want a lattice/PDE the tree would build largely for
them (the T68 lattice is one-factor rates; an equity-credit convertible lattice is a bigger step). On the
shelf, weakest first to pull.

---

## Named-but-not-in-hand — NOT citable as `source:` until obtained

Transcribed verbatim from `BERHAN-2012`'s reference list (p. 49), so they are grounded in a held
source rather than in recollection. Obtain the original before any formula from it becomes
load-bearing — JY and the market models become so at tranches 4–5.

- `JARROW-YILDIRIM-2003` — `referenced-by: BERHAN-2012 [1]`. Jarrow, R. and Yildirim, Y.,
  *Pricing Treasury Inflation Protected Securities and Related Derivatives using an HJM Model*,
  2003, Journal of Financial and Quantitative Analysis 38(2), 409–430. Origin of the HJM/FX-analogy
  model (tranche 4).
- `MERCURIO-2005` — `referenced-by: BERHAN-2012 [2]`. Mercurio, F., *Pricing inflation-indexed
  derivatives*, 2005, Quantitative Finance 5(3), 289–302. The forward-CPI market model.
- `BRIGO-MERCURIO-2007` — `referenced-by: BERHAN-2012 [3]`. *Interest Rate Models — Theory and
  Practice*, 2007, Springer Finance. (Listed as "Damiano M. and Mercurio F."; the author is
  Damiano Brigo — flagged as a transcription inference, not silently corrected.)
- `BELGRADE-BENHAMOU-KOEHLER-2004` — `referenced-by: BERHAN-2012 [4]`. Belgrade, N., Benhamou, E.,
  and Koehler, E., *A Market Model for Inflation*, 2004, ssrn.com/abstract=576081.
- `HJM-1992` — `referenced-by: BERHAN-2012 [5]`. Heath, D., Jarrow, R., Morton, A., *Bond Pricing
  and the Term Structure of Interest Rates: A New Methodology for Contingent Claims Valuation*,
  1992, Econometrica 60(1), 77–105.
- `BRIGO-MERCURIO-2005-SV` — `referenced-by: BERHAN-2012 [6]`. *Pricing inflation-indexed options
  with stochastic volatility*, 2005. (Author listed "Damiano M." — i.e. Damiano Brigo.)
- `BRIGO-MERCURIO-2009-SABR` — `referenced-by: BERHAN-2012 [7]`. *Inflation modelling with SABR
  dynamics*, 2009, Risk June, 106–111. (Author listed "Damiano M." — i.e. Damiano Brigo.)

Transcribed verbatim from `BROCKHAUS-2016`'s reference list (tranche 72), so these too are grounded
in a held source. They are the **attribution chain for the CEV closed form**: tranche 72 cites
`BROCKHAUS-2016` as its `source:` because that is the held text that prints the formula, and these
name where the formula comes from. Obtaining either would turn the corrected arrangement from
"restored, and confirmed by requirement oracles, the section's own derivation and a simulation" into
a direct transcription — which is the only reason to obtain them.

- `COX-1996` — `referenced-by: BROCKHAUS-2016` (p.79, "due to Cox (1996)"). Cox, J. C.,
  *The constant elasticity of variance option pricing model*, 1996, The Journal of Portfolio
  Management, pp. 15–17. Origin of the CEV option formula tranche 72 implements.
- `DELBAEN-SHIRAKAWA-2002` — `referenced-by: BROCKHAUS-2016` (p.79, for the squared-Bessel
  process's non-central chi-squared law). Delbaen, F. and Shirakawa, H., *A note of option pricing
  for constant elasticity of variance model*, 2002, Asia-Pacific Financial Markets 9, pp. 85–99.
- `SCHRODER-1989` — **not named by any held source**; recorded here only because the corrected
  argument/noncentrality arrangement is standardly attributed to Schroder, and naming it is honest
  about where the correction's shape comes from. **Flagged as an attribution from recollection, not
  from a held reference list** — the distinction `BERHAN-2012`'s "Damiano M." entry draws, and the
  reason this one is listed last and separately. It is not citable, and nothing rests on it: the
  corrected form is pinned by the three requirement oracles, the derivation and the simulation.
- `KENYON-2008` — `referenced-by: BERHAN-2012 [8]`. Kenyon, C., *Inflation is normal*, 2008,
  Risk 21(7).
