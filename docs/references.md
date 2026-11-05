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

*(A future `verify.py references` gate can assert every `source:` resolves to an `in-hand`/
`worked-from` key here — proposed, not built, until the first drift makes it earn its place.)*

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

### `JAECKEL-2002` — `in-hand`

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

**Status `worked-from`** (T15–T18: the MC spine, the LMM, the European and Bermudan swaptions).

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

---

### `MCINERNEY-ZASTAWNIAK` — `in-hand`

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

**Status is `in-hand`, not yet `worked-from`:** the relevant sections are read and the key formulas
transcribed in both renderings, but nothing is built against it until T10 lands.

**Also covers, for later units:** §1.7 yield-curve construction (T8's territory), ch 4 HJM and
Gaussian models, and **ch 5–6 the LIBOR and swap market models with their calibration** — which is
BERHAN-2012's ch 3 "Market Model I", so one acquisition also removes the source question from that
unit. **Ruled good-enough for MM-I (Bernardo, 2026-11-03):** BERHAN ch 3 publishes the MM-I formulas
in full and is the primary source; McInerney ch 5–6 is sufficient LMM background — the LMM is not
this project's main topic, so **no dedicated LMM text is to be acquired.** (Verified: §5.1 LMM, §5.2
Black's caplet, §5.6 Brace–Gatarek–Musiela, §5.9 swap market model, §6 implementation + calibration.)

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
