# Tranche 36b — wrong-way risk: equity CVA under equity/default correlation (reduced-form)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.44.0`.
**Source:** `BROCKHAUS-2016` ch 14.3 + ch 14.5 (wrong-way risk, reduced-form), on the ch 13.1.2
apparatus (Gaussian hazard correlated to equity) T35 deferred. **Pre-registered RED.** The second half
of the capstone (T36a built the independent-default seam, `v0.43.0`); its close ends the equity &
hybrids arc's builds and fires `CP_equity_hybrids.md` + re-audit #3 (§9).

## 0. The product worth having
**Equity CVA when the option's exposure and the counterparty's default correlate** — the wrong-way risk
T36a's independent-default CVA deliberately omitted. Brockhaus 14.3: ignoring the correlation between
portfolio value and default is "questionable due to typical negative correlation between CDS spreads and
stocks." The reduced-form route (ch 14.5, on 13.1.2's Gaussian hazard correlated to the equity at `ρ`)
gives a **closed form** `CVA = V̂ − V` via a bivariate-normal CDF, plus the correlated MC that pins it.

## 1. The pieces and their layers
- **L0 `foundation/` — the bivariate-normal CDF `Φ₂(x, y; ρ)`.** A new primitive: `src/` has none and
  deliberately avoids scipy (the `quadrature`/`distributions` split, so an oracle may use scipy off the
  production path while the library does not — tranche 11's arrangement). Built here because 13.1.2's
  four-term call and 14.5's `V̂ − V` need it. It is finance-free (a probability integral), so it is L0.
- **L3 `models/` — the correlated equity-credit model.** The 13.1.2 parameters on top of T35's
  defaultable-equity idea: the equity/hazard correlation `ρ` and the Gaussian-hazard volatility `σ_H`
  (bundle them as vocabulary if the field count presses §7 — they are one object, the hazard's law),
  carrying the `MarketSnapshot` and `SurvivalCurve`. The `a_T` of eq 13.4 is solved from the survival
  probability (a model-free calibration, L3 — no engine inverted).
- **L5 `risk/` — the WWR CVA (closed form + MC).** The closed-form `CVA = V̂ − V` (ch 14.5, the
  bivariate-normal terms with `y, ν, μ` of eq 14.5), and an independent **correlated-default
  Monte-Carlo** (equity + correlated Gaussian hazard, the *exact* indicator, no truncated-Gaussian
  approximation) carrying its SE.

## 2. The oracle (RULED)
- **The new L0 `Φ₂` primitive is oracled in its own right first (§4 "a reference is code too").**
  Transcribe its series/quadrature in both renderings; anchor by the reductions `Φ₂(x,y;0) =
  Φ(x)Φ(y)`, `Φ₂(x,y;ρ) → Φ(min(x,y))` as `ρ→1`, the marginals `Φ₂(x,∞;ρ)=Φ(x)`, and the reflection
  identities; and assert **relative** accuracy in the tail against `scipy.stats.multivariate_normal`
  (off the production path, the T20a high-precision-reference rule), because an absolute-tolerance
  oracle is blind to tail relative error. `Φ₂` lands green before any WWR code consumes it.
- **The `ρ → 0` exact reduction — the preferred anchor.** At zero correlation the WWR closed form must
  collapse **exactly** to T36a's independent-default CVA (`(1−R)(1−p₀)P₀·BSCall`), which is already
  reconciled to our `unilateral_cva`. A no-op (ignoring `ρ`) satisfies it, so **companion — `ρ ≠ 0`:**
  the WWR CVA differs **materially** from the independent CVA, asserted non-degenerate.
- **Wrong-way sign and monotonicity — DERIVED from the model, never from intuition (the T35 lesson,
  built in).** CVA must be **monotone in the equity-default correlation**, and the *direction* is
  asserted only after deriving it from the joint law (or reading it off the MC across a `ρ` grid) —
  not from "wrong-way makes CVA bigger," whose sign depends on the exposure's direction and the
  correlation's reference. T35 shipped a backwards economics test from an intuited direction; the guard
  here is to compute the comparative static and assert the sign the **model** produces, with `ρ=0` as
  the pivot, and to **state the wrong-way convention explicitly** in the test so no later reader
  re-imports the intuition.
- **The independent correlated MC — the non-degenerate correctness anchor, `ρ` pinned where observable
  (§4, T17).** Closed-form `V̂−V` vs an MC simulating equity + correlated Gaussian hazard with the
  *exact* indicator (no truncation), computing `(1−R)·E[1{τ≤T}·V_τ⁺]` within band. Because `ρ` enters
  the **joint law**, a single CVA scalar is a weak test — pin `ρ` **directly** by the realised
  correlation of the simulated equity and hazard increments (the T17 realised-covariance pin), so a
  factorisation bias shows as failure-to-improve as `√N`. Band `k·SE + the truncated-Gaussian
  approximation's own MEASURED error` (measured at a huge path count where sampling noise is
  negligible), never one flat percentage for both; **assert the band's resolution** — the smallest WWR
  effect a green rules out at the committed path count.
- **Requirement oracles (survive a transcription error).** `CVA ≥ 0`; `CVA → 0` as `h → 0` and `R → 1`;
  `CVA` bounded above by the full-loss `(1−R)P₀·BSCall` (the `p₀→0` limit); the `a_T` solve (eq 13.4)
  must satisfy `p(T) = E[e^{−H_T}]` by construction (a requirement-identity on the calibration, the
  "a formula and its stated inverse must actually invert" rule). **Transcribe the `a_T` solve and the
  `V̂−V` terms in both renderings** and use the book's derivation steps (13.4; the four-term assembly of
  14.5) as the independent check on the final line (§4's stronger rung).
- **Mutation.** Each must die: `ρ=0` in the WWR path (→ companion dies); flip `ρ` sign (→ monotonicity
  direction + realised-ρ pin die); drop a term of `V̂−V` (→ MC + reduction die); break `Φ₂` back to
  `Φ·Φ` (→ the `ρ≠0` companion dies). Non-degeneracy premises: `ρ≠0`, `h>0`, `R<1`, survival curve read
  beyond its first segment.

## 3. Process
Branch `tranche/36b-wwr`. `test` (RED: `Φ₂` reductions + tail relative accuracy; the `ρ→0` reduction to
T36a; the derived WWR sign/monotonicity; the correlated MC with the realised-ρ pin; the requirement
identities; the mutation table) → `feat` → `docs` → `chore(release): v0.44.0`. **`Φ₂` lands and is
oracled before any WWR code consumes it** (the primitive first). **Open the draft PR early;
pre-register the requirement identities + the `ρ→0` reduction first.** **Re-time `slow`** — the
correlated MC is new cost; the arc's `slow` margin is tightest at `test_local_volatility` (~7.7s/23%).
`source:` `BROCKHAUS-2016` (ch 14.3, 14.5, 13.1.2); `crosscheck: quantlib` only if a reduced-form
equity-CVA/WWR engine matching this convention is confirmed, else `crosscheck: none — <reason>`.
**Raise an `Ask:`** if, at build, the truncated-Gaussian approximation's error can be pinned **only** by
a twin-MC with no closed-form/requirement anchor — then WWR is `out_of_method` (§1), recorded as a
candidate, and T36a stands as the arc's (complete) capstone. *(The reading says it IS oracle-able: ρ→0
and h→0 reduce exactly, and the approximation error is MC-measurable against the exact indicator — so
this is the §1 guard, not an expectation.)*

## 4. What this tranche is NOT (scope fence)
No **structural / asset-barrier WWR** (ch 14.4, Merton/CreditGrades — a new model family, `to_go_deeper`).
No **portfolio / netting** CVA (single equity-option exposure). No **collateralised** WWR (T29's MPoR on
the correlated exposure — a later pull). No **FFT/Gauss-Hermite** performance work beyond what the single
closed form needs. T36b is wrong-way risk on the one equity-option exposure, reduced-form; it closes the
arc's builds.

---

## 5. Cowork's ruling on T36b — (reserved for §10 step 4, after the build)
