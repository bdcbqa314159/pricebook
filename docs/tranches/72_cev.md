# T72 — the CEV smile model (the single-curve short-rate arc capstone)

**Cowork spec, 2026-11-21.** Last spine tranche of project nine (`MCINERNEY-ZASTAWNIAK`; ch 8.3). The
constant-elasticity-of-variance model `dF = σ·F^β·dW` — the one ch-8 smile model that is **not** a
reparameterisation of Black: its caplet is Schroder's **non-central chi-squared** closed form, which needs a
primitive the tree does not have. That primitive is the tranche's real new infrastructure, and the two T71
models are its reduction anchors (CEV → Black at β=1, → Bachelier at β=0). Specced with the five T71 lessons
applied up front: role-names not source letters, layers checked, printed source values treated as anchors.

## Target
Price a caplet under the CEV model — `price(Caplet, CevModel) → PricingResult` — producing the CEV
implied-vol **skew**.

## Pieces, layer, reuse
- **The non-central chi-squared CDF** (new, **L0** — a finance-free statistical primitive):
  `noncentral_chisquare_cdf(x; dof, noncentrality)`. The honest production implementation is the
  Poisson-weighted sum of central chi-squared (regularised incomplete gamma) terms, or Benton–Krishnamoorthy.
  **Oracled against `scipy.stats.ncx2.cdf` off the production path** — the T20a `distributions.py`-off-scipy
  arrangement (the library ships no scipy dependency; the oracle may use it). A reference is code too (§4):
  argue the primitive's accuracy in the **tail and high-noncentrality** regimes where the series is slow, and
  assert **relative** accuracy where the CDF is near 0 or 1 (the T20a near-zero/near-one rule).
- **The CEV caplet** (new engine, L4) — Schroder's formula in terms of the two non-central chi-squared CDFs.
  The `β` here is genuinely the **CEV exponent** (which is exactly why T71's shift is `displacement`, not
  `β` — F-71.1). Reuses the new L0 primitive; reuses `black_implied_vol` for the smile.
- **The model** (new, L3) — `CevModel` carrying the `MarketSnapshot` (§3), the vol `σ`, and the exponent `β`.
  Distinct from the T71 models (CEV is a different idea, with β its defining parameter); §6.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **McInerney–Zastawniak ch 8.3 (CEV)** in `-raw` and `-layout` and reconcile (§4). Transcribe Schroder's
caplet formula and the `(a, b, c)` arguments of the two χ² CDFs pointwise. **If ch 8.3 prints a worked example
(as 8.4 did — F-71.4), those printed marks are transcription anchors, assert them** — do not reduce the
external check to "measure and report" where the source hands you a number. Confirm the source's `β`
convention (some texts use `β` for the exponent directly, others for `1−` it — read it, do not assume).

## Oracles (RED-first, §4)
1. **The L0 non-central chi-squared CDF, pointwise against scipy** (off production path), across a grid that
   **includes the tail and high-noncentrality** (where the series converges slowly) — relative accuracy near
   0/1 (T20a), its validity regime stated (a reference is code too, T25). Prefer a mutation (drop a Poisson
   weight, mis-index the gamma series) to a reading.
2. **The CEV caplet — Schroder's formula transcribed pointwise**, plus any **printed source example** (F-71.4).
3. **Reduction: β=1 ⇒ Black exactly** (`black_price`), and **β=0 ⇒ Bachelier exactly** (T71's
   `normal_positive_part`/Bachelier caplet). Two exact reductions to **already-verified code** — the arc's
   signature anchor, now with two anchors bracketing CEV. **Companions required (§4/T39):** genuine
   **intermediate β** (e.g. 0.3, 0.7) exercising the real CEV χ² structure — both reductions are blind to the
   non-central chi-squared machinery, which is everything this tranche adds. **Watch the reduction numerics
   (T66 clause):** β→1 and β→0 may be **removable singularities** in Schroder's `(a,b,c)` (e.g. `F^{2(1−β)}` /
   `(1−β)` forms) — if so, **rate-in-a-regime, measured for this model**, not a band; and β=0/β=1 exactly may
   be bit-identical (a separate code path) — label a tautology as not-an-oracle (F-71.3).
4. **Requirement oracles:** put–call parity; the `σ→0` intrinsic limit; the CDF monotone in `x`; and the
   **β-monotone skew direction** asserted only as a **sign** the finance forces (β<1 gives a downward skew),
   with the skew's shape **measured** (obligation-not-form — but a printed β-example is an anchor, F-71.4).
5. **Cross-model smile continuity:** CEV at β near 1 approaches the Black (flat) smile and near 0 the
   Bachelier smile — a measured continuity check against the T71 models across β.

## Degeneracy premises (§4 / §9)
- The reductions' companions are **non-degenerate in β** (intermediate β, T39) — the whole model is the
  parameter the reductions zero/one.
- **Off-the-money strikes** (a smile is invisible at the money), **non-trivial σ√T**, and a β genuinely in
  (0,1). Prefer a **mutation to a reading**: swap the two χ² argument triples, drop the non-centrality, flip a
  `1−β`, use `β` where `2β` or `2(1−β)` belongs (the classic CEV index bugs the parity/reduction oracles catch).
- The χ² primitive's fixture must reach the **slow-convergence regime** (high noncentrality / deep tail) — a
  low-noncentrality-only sweep makes a truncated series look exact.

## Provenance / gates
- Header `source: MCINERNEY-ZASTAWNIAK §8.3`; `oracle:`; `crosscheck:` (QuantLib / scipy has CEV and ncx2 — a
  real target; the L0 primitive's scipy check is the crosscheck made production-adjacent).
- The L0 primitive needs its own provenance (`stock:`/`source:` — Schroder 1989 or the standard ncx2
  reference the source cites; cite only if `in-hand`, else transcribe from the held text and note the method).
- Version → **v0.74.0**; one floor bump; one `CHANGELOG` entry. Re-time any slow χ²-series test on **both** CI
  runners (§9).

## Arc close
T72 is the last spine tranche. Its merge triggers the **checkpoint** (`CP_stochastic_rates.md`) and
**re-audit #8** (§9 — arc-close trigger; the cadence lands here, ~T72 since re-audit #7 at T64/T65). Both are
Cowork's.

## Out of scope (deferred → to_go_deeper)
- **Stochastic-volatility smile (ch 8.5)** — SABR-adjacent, no closed form; the `out_of_method`/`to_go_deeper`
  candidate the arc has pointed at throughout.
- **CEV as a short-rate process** (not just a forward-rate smile) — `to_go_deeper` (the absorbing-boundary
  behaviour at F=0 for β<1 is its own subject).
- **Full CEV calibration to a smile** — the model carries `(σ, β)`; fitting both to a market skew is
  `to_go_deeper` (the T11 identifiability discipline — σ and β trade off).
