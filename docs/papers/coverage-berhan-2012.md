# Coverage against BERHAN-2012 — what is built, what is not

**As of v0.6.0** (tranches T1–T6, PR #8 merged). Printed page numbers from the thesis's own
contents. Verified against the PDF in `-raw` and `-layout`, not from memory.

**Headline: all four instruments, one of three models, one of four calibration sections — and
everything prices at inception only.**

| ch | section | pp | status |
|---|---|---|---|
| 1 | 1.1 Inflation basics | 1–2 | — background, nothing to build |
| 1 | **1.2.1 Inflation-linked bond** | 2–3 | **BUILT** (T3) — 1.2.4, 1.2.5 |
| 1 | **1.2.2 Zero-coupon inflation swap** | 3–6 | **BUILT** (T2) — 1.2.13, 1.2.14, 1.2.16 |
| 1 | **1.2.3 Year-on-year inflation swap** | 6–7 | **BUILT** (T4) — 1.2.17, 1.2.18, 1.2.21 |
| 1 | **1.2.4 Inflation cap/floor** | 7–8 | **BUILT** (T5) — 1.2.22, 1.2.23 |
| 1 | 1.3 Inflation and interest-rate risk | 8–10 | **not built** — there is no risk layer in the tree at all (L5 is empty) |
| 2 | 2.1–2.2 Definitions, model specification (HJM) | 11–14 | **not built** — no HJM forward-curve dynamics, no simulation |
| 2 | 2.3 ZCB term structure, JY drift conditions | 14–16 | **not built** — market curves are taken as given, never reconstructed from the model's drift conditions |
| 2 | 2.4 Hull-White specification | 16–18 | **partial** — `β(t,T)` only (2.4.3, 2.4.16). No `θ(t)` fitting, no short-rate dynamics, no bond reconstruction |
| 2 | **2.5 YoY under JY** | 18–20 | **BUILT** (T4) — 2.5.9, 2.5.11, 2.5.12, 2.5.13 |
| 2 | **2.6 Cap/floor under JY** | 20–22 | **BUILT** (T5) — 2.6.1–2.6.5 |
| 3 | **Market Model I** (LMM on nominal + real forward rates) | 23–28 | **nothing** |
| 4 | **Market Model II** (forward inflation indices — Mercurio; Belgrade–Benhamou–Koehler) | 29–32 | **nothing** |
| 5 | 5.1 Nominal and real curves | 33–34 | **half** — the real-from-break-even half is built (1.2.16, `market/inflation_curve.py`); the **nominal bootstrap from par swap rates** (5.1.1, 5.1.2) is not — nominal curves arrive built |
| 5 | 5.2 Simplified Jarrow–Yildirim | 35–37 | **the model yes, the surface no** — see below |
| 5 | 5.3.1 `κn, σn` to ATM nominal cap vols | 38 | **not built** — needs a Hull-White nominal cap engine |
| 5 | **5.3.2 Fit to YoY floor quotes** | 39–43 | **BUILT, NARROWER** (T6) — see below |
| 5 | 5.4 Market Model II calibration | 43–44 | **nothing** |
| 6 | Conclusions, extensions | 45–46 | — |

## §5.2 is already ours, and we did not notice

The thesis's "Simplified Jarrow–Yildirim" posits `dI/I = [rn(t) − rr(t)]dt + σI dW` with
deterministic short rates, then states: *"by Proposition (2.3.1), this model is equivalent to the
Jarrow-Yildirim model with σn = σr = 0."*

That is **exactly T5's Black-reduction oracle**
(`test_with_both_rate_vols_zero_the_caplet_IS_plain_BLACK_on_the_forward_ratio`). We implemented a
chapter-5 model as a degenerate case of a chapter-2 one, which is what the reduction pattern is
for — and the thesis proving the equivalence is independent confirmation the reduction is the right
anchor, not merely a convenient limit.

What is **not** built from §5.2 is the *inversion*: bootstrapping floorlet prices from quoted floor
prices and implying `σI` per (expiry, strike) to construct the volatility surface (Figures 5.3, 5.4).
That is an implied-vol solve the library does not have.

## §5.3.2 — we fit two of the thesis's six parameters, deliberately

The thesis fits **`{κr, σr, σI, ρnr, ρIr, ρIn}`** — six parameters — to YoY quotes, having fitted
`κn` and `σn` to nominal ATM caps in §5.3.1. T6 fits **`(σr, σI)`** and holds the rest fixed.

The subset is deliberate, and the thesis's own result is why: the six-parameter fit **does not
work.** *"We know that we cannot fit to the whole surface"*; removing the two OTM strikes *"still
results in high relative errors"*; excluding the long end *"still results in a poor fit"*; and
*"the presence of a strike skew makes calibration unfeasible for non ATM contracts."* Its conclusion
is to restrict to near-ATM contracts over a truncated expiry range.

T6's A2b is the diagnostic the thesis lacked a name for: six parameters against a thin,
near-collinear strip is ill-conditioned, which is what a poor fit with plausible per-parameter
sensitivities looks like.

**T6's A1 therefore proves our fit recovers parameters from a strip our own model generated — a
round trip. It is not evidence that JY fits market quotes, and the thesis states that it does not.**

## The gap that matters most: everything prices at `t = 0`

Not a chapter of the thesis, so it does not appear in the table, but it is the largest practical
limit. Every engine prices at inception: the ZCIIS uses 1.2.14 (`t=0`) rather than 1.2.13 (general
`t`), and the linker and YoY leg likewise. A trade valued mid-life cannot be priced at all. That
needs an index **series** in the snapshot and a lag-resolution convention — `indexation_lag` is
carried on the product and never read.

## Provenance of this document

Every "not built" row was checked against the tree, not recalled. The equation-number inventory per
module comes from `grep -ho '[0-9]\.[0-9]\.[0-9]\+' src/pricebook/**/*.py`. The §5.3.1 and §5.3.2
parameter lists were read in both `-raw` and `-layout` per charter §4, because the first version of
`docs/tranches/06_jy_calibration.md` asserted the mean reversions and correlations came from
historical estimation — they do not, and that error is corrected there with this as its source.
