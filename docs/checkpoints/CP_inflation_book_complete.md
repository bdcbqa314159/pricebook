# Checkpoint — BERHAN-2012 is built, ch 1 through ch 5

**Date:** 2026-11-04 · **By:** Code (Cowork to review) · **Baseline:** v0.17.0, PR #20
**Scope:** the whole thesis except §1.3's risk material. Written because tranche 14 closes the book.

## 0. Coverage — the book, chapter by chapter

| | built | by |
|---|---|---|
| **ch 1** — linker, ZCIIS, YoY, cap/floor | **all four** | T2–T5 |
| ch 1.3 — inflation & rate risk | **no** — there is no risk layer (L5 is empty) | — |
| **ch 2** — JY: §2.5 YoY, §2.6 cap/floor | **yes** | T4, T5 |
| ch 2.1–2.4 — HJM dynamics, drift conditions | **no** — only `β`; market curves are taken as given | — |
| **ch 3** — Market Model I | **yes** | T12(b) |
| **ch 4** — Market Model II | **yes** | T13 |
| **ch 5.1** — nominal + real curves | **yes** | T2 (real), T8 (nominal) |
| **ch 5.2** — the implied-vol surface | **yes** | T9 |
| **ch 5.3.1** — `(κn, σn)` from ATM caps | **yes** | T10 |
| **ch 5.3.2** — the six-parameter JY fit | **yes**, with its failure reproduced | T11 |
| **ch 5.4** — MM-II's calibration | **yes** | T14 |

**v0.17.0 · 453 tests · eight gates green on ubuntu + windows · ruff + pyright clean · zero
suppressions.** Three models (JY, MM-I, MM-II) behind one capability protocol; three calibrations.

Everything prices **seasoned** (T7), not only at inception.

## 1. Four honest negatives — the part worth citing

The arc's most valuable output is not the prices. It is four results about the source's own models,
three of which the thesis does not state:

1. **JY cannot smile** (T11). Gaussian, so its implied YoY vol is identical at every strike. The
   thesis says this.
2. **JY's six-parameter fit is structurally underdetermined — at ATM, before any skew** (T11). The
   index correlations enter `V²` only as products with `σI`, and `ρnI` is absent from the convexity
   entirely, so three of six are unidentifiable from YoY floors **at any strike**. `ρnI` recovers at
   the **wrong sign** (−0.597 against +0.15) at rms 8e-15. *The thesis blames only the skew.*
3. **MM-I's freezing relation cannot be iterated** (T12b). `σI` goes negative in two steps at
   realistic rate vols (+0.012 → +0.0056 → −0.0005), so *"we can still assume the volatilities σI,i
   are all constant"* does not survive a realistic parameter set.
4. **MM-II's terminal index vol is ambiguous, and a single-start fit can land on the wrong root**
   (T14). Its caplet price is non-monotone in that vol, so two values give one price; which one a
   fit finds is **start-dependent** and **both report a residual of ~1e-17**.

And one positive the thesis asserts and this library can now explain: **MM-II is the most promising
model — not on price but on identifiability.** The three models' prices differ by 7.7e-09 to 3.7e-05
(T13); their calibrations differ by three unidentifiable global parameters versus one locally
ambiguous point.

## 2. Source defects found — six, in a published MSc thesis

| § | defect |
|---|---|
| 2.5.11 | mis-nested bracket **in our transcription** of it (v0.4.0), from a `-layout` reflow — ours, not the source's |
| 5.1.1 | a spurious leading minus: every par rate would be negative |
| 5.1.2 | the annuity sums `j=0..i−1` where inverting 5.1.1 requires `j=1..i−1` |
| 4.1.3 | denominator `1 + Fn` contradicts `Di`'s `1 + τᵢFn`, in **both** renderings |
| 4.2.2 / pre-4.2.4 | `Vi(t)` defined **twice**, for two different quantities |
| §5.3.1 | states the nominal-vol calibration is *"beyond the scope of this thesis"* — so the model it fits is never published (closed by acquiring MCINERNEY-ZASTAWNIAK) |

MCINERNEY-ZASTAWNIAK contributed one of its own: `d₋`'s radical dropped in both renderings, caught
because the literal reading prices a **call below zero**.

## 3. What the method actually caught — the discipline's own record

Every defect was found by an oracle whose domain was widened, never by review. Charter §4 grew from
each, and now carries: the oracle-must-contain-the-failure rule; transcription independence and
dual-rendering; **derivation-step corroboration**; **properties never replace a pointwise
transcription**; **degenerate fixtures** (now five instances, including one that defeated a
*transcription* oracle); **mutation as proof of power** with the equivalent-mutant exception;
**measurable-above-noise** (no FD condition numbers); multi-start **necessary but not sufficient**;
and **refactor purity by exact-arithmetic identity** rather than bit-identity.

The recurring defect in *my own* work is the degenerate fixture — a parameter sitting where it
cannot be seen (`τ=1` four times, a flat curve, a single period). Mutation found what review did not,
twice conclusively.

## 4. Honest scope of what is built

- **No risk layer** (§1.3): no sensitivities, no scenario engine. L5 is empty.
- **No HJM dynamics** (ch 2.1–2.4): curves are taken as given, never reconstructed from drift
  conditions. No simulation anywhere in the library.
- **No smile model.** All three models are flat in strike; the implied-vol surface (T9) represents
  the market's skew but nothing fits it.
- **MM-I has no distributional error bound.** Its freezing error is bounded as a *sensitivity*
  (71% of `σI` per 200bp); a probability-weighted bound needs an LMM simulation, not built.
- **No external cross-check tier.** QuantLib is deferred; six `crosscheck:` markers name the sites.
- **`σn` is scalar in MM-II** while its index vols are a term structure (T14 §9.2).

## 5. Next — Bernardo's call
The book is closed. What remains outside it: **§1.3's risk layer** (the only unbuilt chapter), a
**smile model** (SABR-type, which the thesis explicitly declines), the **QuantLib cross-check tier**,
or a **new source** — the Capiński–Zastawniak credit text is already in the stock.
