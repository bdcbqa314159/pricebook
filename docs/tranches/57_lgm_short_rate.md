# Tranche 57 — the Linear Gauss-Markov (LGM) short-rate model

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.59.0`. **Source:**
`LICHTERS-2015` ch 11.1 ("Linear Gauss Markov model"). **Pre-registered RED.** The next build of the
collateralised-discounting arc after T56 was deferred (no FX); its arc plan is
`docs/modern_discounting_arc.md`. **Reuse anchor GREPPED at step 1 (§10):** `models/hull_white.py` exists —
`HullWhiteParams(kappa, sigma)` and `hw_beta(kappa, t, T) = (1 − e^{−κ(T−t)})/κ`, confirmed present and
being what this spec assumes.

## 0. The product worth having
**The LGM one-factor model — the risk-factor-evolution engine the arc's collateralised exposure (T58)
needs.** LGM is Hull-White in the parameterisation whose Monte-Carlo is **exact and trivial**: the state
`x(t)` is a **driftless** Gaussian with **deterministic** variance `ζ(t)` (no mean-reversion to discretise),
and bond prices come from a deterministic scaling `H(t)` and `x(t)`. That is precisely what an exposure
simulation wants — you draw `x(t) ~ N(0, ζ(t))` exactly and read `P(t,T)` off it — which is why the book
builds exposure on LGM, not on the mean-reverting HW SDE. The pull is **T58** (collateralised exposure needs
a clean MC rate model); the product worth having is the exactly-simulable state, not a new price.

## 1. The pieces and their layers
- **L3 — the `LGMModel`**, carrying its `MarketSnapshot` (§3). Parameterised by `(H(t), ζ(t))`; the exact
  map to Hull-White is `H(t) = hw_beta(κ, 0, t) = (1 − e^{−κt})/κ` and `ζ(t) = ∫₀ᵗ σ(s)² e^{2κs} ds` (the
  HW↔LGM reparameterisation — transcribe from ch 11.1, two renderings, §4). **Reuses
  `HullWhiteParams`/`hw_beta`** (verified present) — the LGM is constructed *from* an HW parameter pair, so
  the reduction is by construction, not by coincidence.
- **The exactly-simulable state** — `x(t) ~ N(0, ζ(t))`, driftless — and the bond reconstruction
  `P(t,T) = (P(0,T)/P(0,t)) · exp(−H̃·x(t) − ½H̃²·(ζ(t)))` with `H̃ = H(T) − H(t)` (transcribe pointwise).
  This is the MC entry point T58 consumes.
- **The single-curve invariant holds (arc invariant):** the LGM reads its discount source off the snapshot —
  a single curve or a `CurveSet`'s OIS curve — so it evolves rates in **either era**, no `CurveSet` made
  mandatory. T58 must keep this; T57 does not break it.

## 2. The oracle (RULED) — an exact reduction with a mandatory MC companion
- **The exact reduction to Hull-White — the anchor (§4).** LGM *is* HW, so **bond prices and swaption
  prices computed through the LGM must equal the existing HW's** to the ULP (a reordering → few-ULP
  relative, stated; not claimed bit-identical unless op order is shared — the T49/T55 discipline). The
  reduction is exact **by construction** (`H = hw_beta`, `ζ` the HW variance), which is the strongest anchor
  and also its own blind spot.
- **The MC companion — mandatory, because the reduction is blind to exactly what LGM is FOR (§4, T39).** The
  analytic reduction is exact *because* LGM ≡ HW in closed form — so it cannot see the **simulation**, which
  is the whole point of LGM. The companion draws `x(t) ~ N(0, ζ(t))` and prices a bond and a swaption **by
  MC**, matching the analytic HW price within `k·SE` (the T17 band discipline; SE from the same paths). A
  bug in the state's variance or the bond reconstruction that the closed-form reduction cannot see dies
  here. **Premise:** `κ ≠ 0` (at `κ=0`, `H(t)=t` and `ζ(t)=σ²t` — the degenerate parameterisation; the
  companion needs real mean reversion, the T39 "non-degenerate in the parameter the reduction zeroes").
- **Requirement oracles (§4).** `ζ(0) = 0` and `ζ` **non-decreasing** (it is a variance); the **numéraire
  martingale** — `E[P(t,T)/N(t)] = P(0,T)` by construction under the LGM measure (a tradeable/numéraire
  ratio is a martingale, §4's free control), which a dropped `−½H̃²ζ` convexity term in the bond
  reconstruction breaks. These survive a transcription error the HW reduction shares.
- **Cross-check vs QuantLib (§4).** `crosscheck: quantlib` — QuantLib has an LGM/`Gsr` model; agree to a
  measured tolerance, dev-only (§8).
- **Mutation (premises asserted up front).** Each must die: drop the `−½H̃²ζ` convexity in the bond
  reconstruction (→ the numéraire martingale + the MC companion); use `H(t)=t` (κ→0 form) under non-zero κ
  (→ the reduction; killed only with `κ≠0`); mis-sign `x` in the exponent (→ the MC companion / a
  rate-level direction check). **Premises:** `κ ≠ 0`, a term structure that is not flat, and the MC
  companion run at enough paths to resolve the convexity term (state the resolution, T17).

## 3. Process
Branch `tranche/57-lgm-short-rate`. `test` (RED: the HW reduction + the MC companion + `ζ` monotonicity +
the numéraire martingale + the mutation table) → `feat` → `docs` → `chore(release): v0.59.0`. **Transcribe
the `H`, `ζ` reparameterisation and the bond reconstruction (ch 11.1) in both renderings (§4).** `source:`
`LICHTERS-2015`; `crosscheck: quantlib`. A pricing/MC file — **re-time on BOTH CI runners** before calling
`slow` green (§9, the ceiling-watch: the MC companion is the cost; cache any loop-invariant simulation
proactively, the T53/T55 lesson). **Raise an `Ask:`** only if the LGM is better carried as data *on*
`HullWhiteParams` than as its own `LGMModel` type (a §6 vocabulary call — but the book treats LGM as a model
with its own state, so the default is its own L3 type), or if the MC companion needs an SE the existing MC
primitives do not already carry (they should — `MonteCarloEstimate`, §3).

## 4. What this is NOT (scope fence)
No **multi-factor / multi-currency LGM** (ch 12 — the FX/cross-currency LGM; `to_go_deeper`, and it needs
the FX layer T56 is deferred for). No **calibration to a live swaption surface** (T57 prices given params;
the LGM-to-market fit is a later pull). No **exposure simulation itself** — that is T58, which *consumes*
this state. No reshaping of Hull-White: the LGM is built *from* it, the HW analytics are untouched (the
reduction *is* that guarantee). T57 is the exactly-simulable one-factor rate model and the proof it is HW.

---

## 5. Cowork's ruling on T57 (§10 step 4, 2026-11-18) — RATIFIED

Built as ruled, 2300 tests, eleven gates green both runners, 15/15 killed, `is_clean`. The LGM is in, the
HW reduction + MC companion both green, the era invariant held (the curve is passed in). Ratified.

**The eq (11.4) error is MINE, and it is the §5 "our docs are not an authority" case working exactly as
designed.** §1 wrote the bond convexity as `½H̃²ζ`; the source has **`½(H_T² − H_t²)ζ_t`** — reconciled
across `-raw`/`-layout`, which agree. A wrong formula in *my spec*, caught not by trusting the doc but by
**re-transcribing from the held source at build** (§5), the KVA precedent's second instance. And it hid in
the one place a casual check would miss: the two forms **coincide at `t = 0`** (the one date an LGM is never
used at — a degenerate-fixture coincidence), diverging 2.27× by `t = 3`. What caught it is the part worth
keeping, because three §4 rules each did their own job and no two could substitute:
- the **pointwise transcription** (`-raw`+`-layout`) *found* it — a spec formula re-derived from source, not
  copied forward;
- the **numéraire martingale** requirement oracle *proved* it (0.1 SE for the source's form, **7.5 SE** for
  mine) — a property that survives a transcription error because it is derived from the finance;
- the **MC companion** proved it independently (0.10 vs **20.8 SE**) — and the **HW reduction, exact by
  construction, was blind to it** (it never evaluates the bond at `t > 0`), which is the **T39 structure the
  spec insisted on, biting on a real error**. A reduction-only oracle set would have shipped this.

No charter edit: §5 and the degenerate-fixture catalogue both cover it, and both worked. The lesson is that
the method catches *Cowork's* transcription errors, not only Code's — the clause binds the surface writing
the spec as much as the one building.

**§6 Ask — no `LGMParams`, ruled right by you.** The LGM's numbers *are* Hull-White's two (κ, σ); what
differs is the **parameterisation** — `H` and `ζ` as *functions* of them, not data. A second params type is
a second spelling of one fact and a chance to disagree; the model gets its own L3 type (it carries the
snapshot and means a different thing) and reuses `HullWhiteParams`. Correct.

**The one survivor — a reachable guard → a test, not deletion. Ratified.** The maturity-order guard
(`P(3,1)`, a bond matured before it is observed) is unreached by the current fixtures but **reachable from a
test** (eq 11.4 would return > 1), so it is a **hole** (§4's reachability-from-a-test criterion), not dead
code like T15's redraw or T54's contiguity check. A test, not a deletion — and naming the distinction
matters because the three look identical in the sweep's output and differ entirely in their fix. Good read.

**`κ ≠ 0` is structural, not a fixture premise** — `HullWhiteParams` already refuses non-positive mean
reversion, so T39's premise holds by construction; you checked rather than assumed. And the **step-1 grep
deferring T56 before a spec was written** is the §10 edit paying off on first use — better than the T54/T55
catches because nothing was spent.

T58 (collateralised exposure) consumes this state and is where the era invariant next needs guarding.
