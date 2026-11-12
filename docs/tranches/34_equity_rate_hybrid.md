# Tranche 34 — the equity-interest-rate hybrid (equity under Hull-White)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.41.0`.
**Source:** `BROCKHAUS-2016` ch 11.1 (constant equity volatility) — the log-normal cash-bond/stock
model (11.1.1, eq 11.1) specialised to one-factor Hull-White rates (11.1.2). **Pre-registered RED.**
Fifth unit of the equity & hybrids arc, and **the first true hybrid** — the equity core (T30–T33) meets
the rates vertical (`models/hull_white`, built in the JY/nominal-rates arc). It registers into the
`(product, model)` dispatch T33b just built, which is precisely why T33b came first.

## 0. The product worth having
**The European equity option when the short rate is stochastic (one-factor Hull-White), priced in
closed form.** For a long-dated option, part of the equity's realised volatility comes from the
discount factor moving, and the equity/rate correlation is a real, unhedgeable risk (Brockhaus's
"hybrid risk"). The deterministic-rate equity vanilla (T30) prices this as if the curve were frozen;
T34 is the product that prices the rate volatility and the correlation **exactly**, because for
Gaussian rates and lognormal equity the vanilla still has a Black-Scholes-form closed solution — no
PDE, no MC grind for the price itself.

Brockhaus 11.1.1: with cash bond `B` and stock `S` jointly lognormal,

```
E[B_t^{-1} (S_t − K)^+] = P(t) [ F(t)·N(d₊ + ρ_{B,S} σ_B √t) − K·N(d₋ + ρ_{B,S} σ_B √t) ]      (11.1)
d± = (1/(σ_S √t)) [ log(F(t)/K) ± ½ σ_S² t ]
```

where **σ_S is the *total* volatility of log S** (it *includes* the bond's contribution), `F(t)` is the
equity forward (our T30 dividend forward), `P(t)` the discount factor, and the d± are shifted by
`ρ_{B,S} σ_B √t`. So the hybrid price is Black on the forward with (a) the variance augmented by the
rate vol and the equity-rate cross term and (b) a correlation shift in the d±.

Brockhaus 11.1.2 specialises to one-factor Hull-White: the total vol `σ_S` is assembled (eq 11.2) from
three pieces,

```
σ_S² = σ_B² + σ_{S/B}² + 2 ρ_{B,S/B} σ_B σ_{S/B}                                            (11.2)
(σ_t^B)²  t = ∫₀ᵗ (v_s(T) − v_s(t))² ds                     — the bond-vol contribution
(σ_t^{S/B})² t = ∫₀ᵗ σ_s² ds                                — the pure equity vol
ρ^{B,S/B} σ_t^B σ_t^{S/B} t = ∫₀ᵗ ρ_s σ_s (v_s(T) − v_s(t)) ds   — the cross term
with   v_s(T) = σ_r · β(s,T),   β(s,T) = (1 − e^{−κ(T−s)})/κ   — OUR `hw_beta`, already verified (T4/T10a)
```

and `ρ_{B,S}` (the raw stock/bond correlation used in the d± shift) from the decomposition (eq 11.3).
**The first hybrid carries constant parameters** (`σ`, `σ_r`, `ρ` constant), so the three integrals
evaluate to closed forms in `κ, σ_r, σ, ρ, T`; the time-dependent/piecewise case is `to_go_deeper`
(see §4).

## 1. The pieces and their layers
- **L3 `models/` — the hybrid model, reusing `HullWhiteParams`.** A frozen bundle
  (`EquityHullWhiteModel`, §7 vocabulary) carrying the `MarketSnapshot`, the **`HullWhiteParams`
  (κ, σ_r)** for the rate factor (reused, not re-declared — one Hull-White convention in the library),
  the equity volatility `σ_{S/B}`, and the equity-rate correlation `ρ`. Four fields (`market`,
  `rate_params`, `equity_vol`, `correlation`) — under the §7 limit; the dividend yield `q` lives on the
  `EquityAsset` already (T30), not a model field. **Do not** invent a second product type — a European
  equity option is one product whatever prices it (the §2/§6 ruling from T33).
- **L4 `engine/` — the closed-form hybrid vanilla pricer**, `@register`ed under
  `(EuropeanEquityOption, EquityHullWhiteModel)` — a third pricer for the equity option, coexisting
  with Black (T30) and Heston (T33) *because T33b made the registry key on the model*. It assembles the
  three HW variance pieces (constant-param closed forms), builds `σ_S` via (11.2) and the d± shift via
  (11.3)/(11.1), and returns Black on the forward — reusing the verified `black_price`.
- **L4 `engine/` — the hybrid Monte-Carlo** (the independent non-degenerate cross-check): a correlated
  two-factor simulation of the Hull-White short rate and the equity, on the Jäckel path engine (RNG,
  correlation, variance reduction reused from T15–T22), repricing the option and carrying its SE.

## 2. The oracle (RULED)
The `σ_r → 0` reduction is exact but a no-op satisfies it; the non-degenerate case is pinned by an
**independent method** (MC, with ρ pinned where it is directly observable) plus **requirement
identities** that survive a transcription error, plus the book's own **derivation-step** cross-check.

- **The `σ_r → 0` exact reduction — the preferred anchor (§4), with the mandatory companion.** At
  rate-vol zero, `σ_B = 0`, the cross term vanishes, `σ_S = σ_{S/B}` (the pure equity vol), the d±
  shift `ρ_{B,S} σ_B √t → 0`, and `B_t^{-1} → P(t)` deterministically — so (11.1) collapses
  **bit-identically** to T30's `P(t)[F·N(d₊) − K·N(d₋)]`, the deterministic-rate equity vanilla.
  Exact by construction; but a no-op (ignoring the rate dynamics) also satisfies it, so it is vacuous
  alone. **Companion — `σ_r > 0, ρ ≠ 0`:** the price must differ **materially** from the frozen-curve
  T30 price (both the variance augmentation and the correlation shift move it), asserted non-degenerate
  per the §4 catalog — assert the *magnitude* of the gap, not merely that it is nonzero.
- **Requirement oracles (§4, survive a transcription error).** (a) **Put-call parity** on the hybrid
  call and put at one `(K,T)`: `C − P = P(t)(F − K)` exactly — derived from the finance, blind to a CF
  transcription slip. (b) **The forward/martingale identity** `E^T[S_T] = F(t)` — the drift is pinned
  independently of the vol dynamics (and asserted of the **MC** too, so the simulation's drift is
  checked separately from its variance; the T16/T17 "pin the drift where it is directly observable"
  rule). (c) **The decomposition identity (11.2) as a requirement**: the assembled `σ_S²` must equal
  `σ_B² + σ_{S/B}² + 2ρσ_Bσ_{S/B}` — an algebraic identity the code must satisfy for all parameters,
  and (crucially) a **derivation step** of the source, so it is an *independent* check on the final
  (11.1), stronger than a second rendering of one line (§4's T12b rung). Transcribe (11.1) pointwise in
  **both renderings** (`-raw` + `-layout`) **and** assert (11.2)/(11.3) — the two catch different
  failures (§4 "requirement oracles complement a pointwise transcription, never replace it").
- **The independent-method cross-check — the non-degenerate correctness anchor.** The closed-form
  hybrid price vs the **hybrid MC** (correlated HW rate + equity, independent code and method) within
  `k·SE`, on `σ_r > 0, ρ ≠ 0`. A variance-assembly or d±-shift error that survives the reduction and the
  requirement identities is caught here — a different method entirely. Band = `k·SE(MC)` with the
  closed form's own (exact) value as the centre; **assert the band's resolution** (the smallest
  variance/correlation error a green rules out at the committed path count), per the T17 banded-
  resolution rule — never a flat percentage absorbing both the MC noise and a real defect.
- **ρ is pinned where it is DIRECTLY observable, not only through a price (§4, T17).** The correlation
  enters the **factorisation** of the two correlated increments, so a single price pins only one
  correlation-weighted scalar and is a weak test of ρ (the T17 trap — a price gave a 13.9% band where
  the direct pin gave 1.66%). Pin ρ by the **realised covariance of the simulated HW-rate and equity
  increments** reproducing `σ_r σ_S ρ · dt` per step (every pair, individually), so a factorisation
  bias shows as failure-to-improve as `√N`. This is the T17 instrument, and it applies here because ρ
  is in the diffusion, not the drift (the one-seed drift-differencing identity does **not** apply).
- **Mutation (prove the oracle has power).** Each must die: flip the `ρ` sign (→ the cross term and the
  d± shift flip; caught by the companion, the MC, and the realised-covariance pin directly); drop the
  cross term `2ρσ_Bσ_{S/B}` in (11.2) (→ `σ_r→0` still green, companion + MC die, (11.2) identity dies);
  drop the bond-vol `σ_B²` term (→ reduction still green, companion + MC die); use `σ_{S/B}` (pure
  equity vol) where `σ_S` (total) belongs in d± (→ reduction green since they coincide at `σ_r=0`,
  companion dies — the classic degenerate-fixture trap, so the companion fixture MUST have `σ_r ≠ 0`);
  drop the d± correlation shift `ρ_{B,S}σ_B√t` (→ put-call parity may survive, MC + companion die).
  **Assert the non-degeneracy premises explicitly** (`σ_r > 0`, `ρ ≠ 0`, `κ` finite so `β` is not its
  `T−t` limit) so none of these mutants hides behind a fixture where the term it breaks is invisible.

## 3. Process
Branch `tranche/34-equity-rate-hybrid`. `test` (RED: the `σ_r→0` reduction + put-call parity + the
forward identity + the (11.2) decomposition identity + the MC cross-check + the realised-covariance ρ
pin + the mutation table) → `feat` → `docs` → `chore(release): v0.41.0`. **Open the draft PR early;
pre-register the requirement oracles first** (the identities need no implementation — the T32 RED-first
fix, kept clean at T33). **Transcribe eq 11.1 in both renderings and reconcile; use eqs 11.2/11.3 (the
page-161 derivation steps) as the independent check on the final (11.1)** — §4's stronger rung, which
the source hands us because it derives the result in steps. **Re-time `slow`** — the hybrid MC is new
cost, and the margin is not bankable (T19); if `test_local_volatility`'s ~7.7s/23% margin is still the
tightest, note it. `source:` `BROCKHAUS-2016` (ch 11.1); `crosscheck: quantlib
AnalyticBSMHullWhiteEngine` (a real target — QuantLib implements exactly this hybrid). **Raise an
`Ask:`** if (a) the tranche is too large and the closed form + the MC want splitting (T33a/T33b
precedent), or (b) the constant-parameter integral forms in 11.1.2 disagree between renderings in a way
the derivation steps cannot reconcile (then rule which form by dimension/convention, §4).

## 4. What this tranche is NOT (scope fence)
No **time-dependent / piecewise equity volatility** — the 11.1.2 integrals with non-constant `σ_s`,
`ρ_s` (a quadrature; oracle-able, `to_go_deeper`). No **two-factor or CIR rates** (ch 10.5; the
one-factor HW is what the arc reuses — `to_go_deeper`). No **local or stochastic equity vol under
stochastic rates** (the LSV-hybrid, ch 11.3 — `to_go_deeper`, a genuinely harder object). No **Gauss
copula** hybrid (ch 11.2). No **equity/rate exotics** (the equity-linked swap structures Brockhaus
names — MC consumers pulled later if interest wants one). T34 is the one clean closed-form hybrid: the
European equity vanilla under one-factor Hull-White, constant parameters. T35 (defaultable equity)
begins the credit seam.
