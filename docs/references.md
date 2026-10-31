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
