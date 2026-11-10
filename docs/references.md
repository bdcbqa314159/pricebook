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
| 2.9 | 2.24 | Black's caplet formula and the implied Black (spot) volatility |
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

### `BROCKHAUS-2016` — `in-hand`

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

**Used for:** the equity-and-hybrids arc (`docs/equity_hybrids_arc.md`). **Status `in-hand`, not
`worked-from`** until the first equity tranche lands.

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
- `KENYON-2008` — `referenced-by: BERHAN-2012 [8]`. Kenyon, C., *Inflation is normal*, 2008,
  Risk 21(7).
