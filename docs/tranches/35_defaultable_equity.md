# Tranche 35 — defaultable equity (the equity-credit seam, independent default)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.42.0`.
**Source:** `BROCKHAUS-2016` ch 13.1.1 (Black-Scholes and independent default). **Pre-registered RED.**
Sixth unit of the equity & hybrids arc — the equity core meets the **credit** vertical (the
`SurvivalCurve` built in the XVA arc, T25). The second hybrid, after the equity-rate hybrid (T34).

## 0. The product worth having
**The European option on a stock that can default, when default is independent of the diffusion** —
the elementary, exact defaultable-equity vanilla, a **modified Black formula**. Brockhaus 13.1.1:
`S_t = S̃_t · 1{τ > t}` with `S̃` lognormal (forward `F̃`, vol `σ̃`) and the default time `τ`
**independent** of `S̃`. Writing `p(t)` for the risk-neutral survival probability — **our
`SurvivalCurve.survival(t)`** — the forward of the surviving process is `F̃(t) = F(t)/p(t)` (eq 13.3),
and the option prices collapse to **Black on a scaled strike**:

```
E[(S_t − K)^+] = p(t)·BSCall(F(t)/p(t), K, σ̃, t) = BSCall(F(t), p(t)·K, σ̃, t)              (call)
E[(K − S_t)^+] = (1 − p(t))·K + p(t)·BSPut(F(t)/p(t), K, σ̃, t)
              = (1 − p(t))·K + BSPut(F(t), p(t)·K, σ̃, t)                                      (put)
```

The call is Black with the **strike scaled by the survival probability** (default lowers the payoff);
the put carries an extra **`(1 − p(t))·K` default term** — the lump the put holder collects when the
name defaults and the stock goes to zero. The jump-to-default introduces negative skew in `S` that
`S̃` does not carry, which is the product's whole economic point. This is the first time credit enters
an *equity* price, and it reuses two verified pieces whole: `SurvivalCurve` (T25) and `black_price`
(T5/T30).

## 1. The pieces and their layers
- **L3 `models/` — the defaultable-equity model.** A frozen bundle (`DefaultableEquityModel`, §7
  vocabulary) carrying the `MarketSnapshot`, the equity volatility `σ̃` (the vol of the *surviving*
  process `S̃`), and the **`SurvivalCurve`** (reused, not re-declared — the credit convention is the
  survival curve's, one definition in the library). The `SurvivalCurve` already carries its own time
  measure; the model carries the snapshot the equity forward is quoted against (§3). Three fields
  (`market`, `volatility`, `survival`) — well under the §7 limit. **Do not** invent a new product type
  — the `EuropeanEquityOption` is one product whatever prices it (the §2/§6 ruling from T33, held at
  T34).
- **L4 `engine/` — the modified-Black pricer**, `@register`ed under
  `(EuropeanEquityOption, DefaultableEquityModel)` — the **fourth** pricer on the equity option (Black
  T30, Heston T33, equity-rate hybrid T34, this), coexisting because T33b keyed dispatch on the model.
  It reads `p(t) = survival.survival(t)`, forms the scaled strike `p(t)·K` (call) and the `(1−p)K`
  default term (put), and reuses `black_price` on `F(t)` — no new Black arithmetic. Discounting per §3
  as the other equity engines do.

## 2. The oracle (RULED)
The `h → 0` (⇒ `p → 1`) reduction is exact but a no-op satisfies it; the non-degenerate case is pinned
by requirement identities that survive a transcription error, the modified-Black **homogeneity**
identity, and an independent MC.

- **The `h → 0` exact reduction — the preferred anchor (§4), with the mandatory companion.** At zero
  hazard, `p(t) = 1`, the scaled strike `p(t)·K → K`, the put's `(1−p)K → 0`, and both formulas are
  **bit-identically** T30's default-free equity vanilla (`BSCall(F, K, σ̃, t)`, `BSPut(F, K, σ̃, t)`).
  Exact by construction; a no-op (ignoring the survival curve entirely) also satisfies it, so it is
  vacuous alone. **Companion — `h > 0, p(t) < 1`:** the call must be *strictly below* its default-free
  value (strike scaled up) and the put *strictly above* (the `(1−p)K` floor), asserted **material** per
  the §4 degenerate-fixture catalog — assert the magnitude of the gap, and use a **non-flat**
  `SurvivalCurve` (the piecewise-constant hazard read beyond its first segment, mirroring the T25
  survival-curve oracle) so a mutant reading only `hazard(0)` or a flat `e^{−λt}` is separated.
- **Requirement oracles (§4, survive a transcription error).** (a) **Put-call parity with the default
  terms cancelling:** `C − P = F(t) − K` (undiscounted bracket), because
  `BSCall(F,pK) − [(1−p)K + BSPut(F,pK)] = (F − pK) − (1−p)K = F − K` — the survival-scaling and the
  default lump cancel **exactly**, a parity that must hold for every `p` and is blind to a transcription
  slip in either formula. (b) **The survival-weighted forward identity** `E[S_t] = F(t)` (eq: the
  forward "includes the possibility of default") — the default-risky forward is the plain forward, which
  pins the `F̃ = F/p` scaling independently of the vol; assert of the **MC** too so its default-weighting
  is checked separately from its diffusion. (c) **The modified-Black homogeneity identity**
  `p·BSCall(F/p, K, σ̃, t) = BSCall(F, p·K, σ̃, t)` — Black is degree-1 homogeneous in `(forward,
  strike)`, so the two printed forms of the call are the *same* object; assert they agree to ULP (a
  pure-arithmetic identity, the §4 reordering case — not necessarily bit-identical), which is an
  independent check on the strike-scaling transcription.
- **The independent-method cross-check (the non-degenerate correctness anchor).** The closed-form price
  vs an independent **Monte-Carlo**: simulate `S̃` lognormal **and an independent default indicator**
  `1{U < p(t)}` (`U` uniform, independent of the equity draw — the independence is the model premise, so
  the MC must draw them independently), payoff `(S̃·1{survive} − K)^+`, within `k·SE`. A strike-scaling
  or default-term error that survives the reduction and the identities is caught here — different method.
  Band `k·SE` with asserted resolution (§4 banded-resolution). **Assert the MC's own `E[S_t] = F(t)`**
  so its drift/weighting is pinned before it is trusted as the price anchor.
- **Mutation (prove the oracle has power).** Each must die: drop the strike scaling (use `K` not `p·K`
  in the call → `h→0` still green since `p=1` there, companion + parity + MC die — the classic
  degenerate-fixture trap, so the companion fixture MUST have `p < 1`); drop the put's `(1−p)K` term
  (→ parity dies, companion put-side dies, MC dies); use `hazard(0)` for the whole horizon instead of
  `survival(t)` (→ dies only on the **non-flat** curve read beyond segment one — assert that premise,
  the T25 survival-curve degeneracy); invert the scaling (`K/p` for `pK`) (→ companion + MC die).
  **Assert the non-degeneracy premises:** `p(t) < 1` (hazard positive), the survival curve non-flat and
  read past its first segment, and the companion strike chosen so the scaling moves the price materially.

## 3. Process
Branch `tranche/35-defaultable-equity`. `test` (RED: the `h→0` reduction + put-call parity `C−P=F−K` +
the survival-weighted forward identity + the homogeneity identity + the MC cross-check + the mutation
table) → `feat` → `docs` → `chore(release): v0.42.0`. **Open the draft PR early; pre-register the
requirement identities first** (they need no implementation — the RED-first discipline, clean since
T33). **Transcribe eqs 13.2/13.3 and the two option formulas in both renderings** (`-raw` + `-layout`)
and reconcile; the modified-Black forms are given in **two equivalent shapes** (`p·BS(F/p,K)` and
`BS(F,pK)`) — that is the book handing us the homogeneity identity as a built-in cross-check (assert
both shapes agree). **Re-time `slow`** — the defaultable-equity MC is new cost (T19); `test_local_volatility`'s
~7.7s/23% margin is still the arc's tightest, so watch it. `source:` `BROCKHAUS-2016` (ch 13.1.1);
`crosscheck: quantlib` — **none** is the honest marker here unless a QuantLib defaultable-equity vanilla
is confirmed to implement *this* independent-default modified-Black (do not name a target that may not
match the convention, the §8 "a real target or `none — <reason>`" rule); `crosscheck: none — the
independent-default modified-Black is a composition of Black and a survival probability, each already
cross-checked; no single QuantLib engine implements this exact convention`. **Raise an `Ask:`** only if
a QuantLib `AnalyticBSMHullWhiteEngine`-style defaultable vanilla is found that *does* match (then it
becomes a real target), or if the equity forward `F(t)` under the survival measure needs a dividend
convention ruling where risky dividends (`D₀(t) = E[B_t⁻¹ D_t 1{τ>t}]`, the book's risky-dividend note)
interact with the survival curve — which would be a finding, not a refactor, and is explicitly out of
scope below.

## 4. What this tranche is NOT (scope fence)
No **correlated default** — Brockhaus 13.1.2 (Black-Scholes with a Gaussian hazard rate correlated to
the equity at `ρ(X,Y)`, via the truncated-Gaussian approximation `H_T ≈ (a_T + σ_H Y)^+` and the
**bivariate-normal** four-term call) is the *equity/credit correlation*, which the arc assigns to **T36
(equity CVA + wrong-way risk)**; it is the modelling substrate T36's WWR pulls, not T35's. If T36 builds
WWR by MC rather than by 13.1.2's closed form, 13.1.2 becomes a `to_go_deeper` entry (oracle-able — it
reduces to 13.1.1 at `ρ=0` and to default-free at `h=0`, and the truncated-Gaussian approximation error
is measurable against an exact-model MC). No **risky dividends** (the `D₀(t)` survival-weighted dividend
extension, ch 13.1.1's closing note — a convention question deferred). No **stochastic equity vol under
default** (the ch 6/7.4 vol models applied to `S̃` — `to_go_deeper`). No **CDS/defaultable-bond
calibration** of the survival curve (an L1 bootstrap, pulled when a product needs it, §6). T35 is the
one clean closed form: the defaultable-equity vanilla under *independent* default. T36 (equity CVA + WWR)
is the capstone and brings the correlation.

---

## 5. Cowork's ruling on T35 — (reserved for §10 step 4, after the build)
