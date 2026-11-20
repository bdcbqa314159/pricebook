# T66 — the Vasicek short-rate model (+ Merton as its reduction)

**Cowork spec, 2026-11-20.** First tranche of the single-curve short-rate arc (project nine,
`MCINERNEY-ZASTAWNIAK`; `docs/stochastic_rates_arc.md`). The canonical mean-reverting Gaussian short rate —
and deliberately first because it is the **most reuse**: the short rate is Ornstein–Uhlenbeck, the affine
`B(t,T)` is the `hw_beta` the tree already holds, and the whole thing is the constant-parameter sibling of
the Hull-White model built at T10.

## Target
Price a zero-coupon bond and a caplet under the Vasicek model —
`price(product, VasicekModel) → PricingResult` — with the short rate `dr = κ(θ − r)dt + σ dW`.

## Pieces, layer, reuse
- **The model** (new, L3) — `VasicekModel` carrying the `MarketSnapshot` (§3) and the three parameters
  `(κ, θ, σ)` plus `r₀`. Bundle into a frozen value object if market + params exceed the field limit (§7 —
  `HullWhiteParams` is the precedent; `VasicekParams` may *be* `HullWhiteParams` + `θ` + `r₀` or its own
  type — Code rules by whether the grouping is the same idea). **Distinct from `HullWhiteModel`** — Vasicek
  has *constant* θ, HW has θ(t) fitted to the initial curve; same SDE family, different idea (the extended
  model carries a curve, the base does not), so no merge (the §6 call, and the LGM/HW precedent).
- **The affine bond** (new, L3) — `P(t,T) = A(t,T)·e^{−B(t,T)·r}`. **`B(t,T)` IS `hw_beta(κ, t, T)`**
  (`(1−e^{−κ(T−t)})/κ`) — reuse it, do not re-derive; `A(t,T)` is the new closed form. The short rate is OU,
  so the distribution of `r` and the bond are Gaussian/lognormal in closed form.
- **The caplet** (new engine, L4) — a Vasicek caplet is Black on the forward bond price with the model's bond
  variance, exactly as the HW caplet is; **reuse `zero_bond_option` / `bond_option_variance`** (T10) where
  the variance structure matches, or the new Vasicek variance if it differs (ch 3.4). Reuses `black_price`.
- **The MC companion** reuses **`simulate_lgm_states`** (the OU simulator) — the Vasicek short rate is OU
  after the deterministic drift, exactly as the Schwartz log-spot was at T64.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **McInerney–Zastawniak ch 3.4 (Vasicek)** and **ch 3.3 (Merton)** in `-raw` and `-layout` and reconcile
(§4). The load-bearing closed forms are the bond `A(t,T)` and `B(t,T)`, the short-rate mean and variance
(`E[r_T]`, `Var[r_T]` — OU, so `Var → σ²/2κ` as T→∞), and the caplet. Mark every formula "re-transcribe from
eq X"; the spec is a pointer, not the authority. **The spec names the obligation, not the form** (§4): e.g.
oracle 4 below says *measure* the stationary variance against the closed form, it does not pre-assert a
value.

## Oracles (RED-first, §4)
1. **Pointwise transcription of `A(t,T)` and `B(t,T)`** — the test forms its own `(1−e^{−κτ})/κ`
   independently. `B` is `hw_beta`, so **cross-check Vasicek's `B` against `hw_beta`** (same object — a
   consistency guard, **not** independence; §4's same-formula caveat, flag it). `A(t,T)` is the genuinely-new
   formula and gets its own pointwise transcription.
2. **Reduction: κ → 0 ⇒ the Merton model** (ch 3.3) — the driftless Gaussian short rate, `B(t,T) → (T−t)`,
   `r_T ~ N(r₀ + θ·?, σ²t)` per the source. Exact by construction. **Companion required (§4/T39):** a
   **genuine κ > 0** Vasicek case checked against the pointwise transcription — the κ→0 reduction kills
   exactly the mean reversion, so it is blind to the whole OU structure. State the premise that the
   companion's κ is materially non-zero.
3. **Cross-check against Hull-White** — Vasicek with constant θ is the un-fitted sibling of `HullWhiteModel`;
   at matched `(κ, σ)` the two `B(t,T)` agree exactly (both `hw_beta`) and the bond variances share structure.
   Assert the agreement as a **consistency guard** (same object, not cross-source independence — the T17
   caveat). This is the reuse made honest: the tree's HW machinery and the new Vasicek must not disagree
   where they compute the same thing.
4. **Requirement oracle — the stationary variance.** The OU short rate has `Var[r_T] → σ²/(2κ)` as T→∞;
   **measure** `Var[r_T]` across a maturity sweep (from the simulated paths and from the closed form) and
   assert they agree and the finite limit holds. Do **not** pre-assert the approach shape (§4 obligation
   clause). Distinguishes a mean-reverting short rate from a driftless one (the κ→0 Merton limit, whose
   variance grows linearly).
5. **The MC companion (reusing `simulate_lgm_states`)** prices the bond/caplet under genuine mean reversion,
   banded per §4 (`k·SE`, closed form exact, resolution stated). Pin the simulated OU **mean and variance**
   against the model's closed forms first (the T17/T19 unbiased-before-credited discipline).
6. **Requirement oracle — put–call parity** on the caplet (cap–floor = the forward value), and the bond
   boundary `P(T,T) = 1` exactly.

## Degeneracy premises (§4 / §9)
- The κ→0 companion is **non-degenerate in κ** (T39) — state it; the model's whole point is the parameter the
  reduction zeroes.
- **Non-trivial `T`** (not T=1, where `κ·T` hides a factor) and **distinct maturities** for the bond/variance
  term structure. Prefer a **mutation to a reading**: drop the `e^{−κτ}`, flip κ's sign, drop the `1/κ` in
  `B` or the `1/(2κ)` in the variance, swap θ for r₀.
- The HW cross-check fixture must use **matched** `(κ, σ)` where Vasicek's constant θ and HW's θ(t) coincide
  (a flat initial curve, or the degenerate θ(t)=θ), so the two are genuinely the same object there.

## Provenance / gates
- Header `source: MCINERNEY-ZASTAWNIAK §3.3, §3.4`; `oracle:`; `crosscheck:` (QuantLib has a Vasicek model
  and analytic engine — a real target).
- `references.md`: `MCINERNEY-ZASTAWNIAK` is already `worked-from`; this adds `src/` citers. `verify.py
  references` holds.
- Version → **v0.68.0**; one floor bump; one `CHANGELOG` entry.

## Out of scope (carried / deferred → T67+)
- **Two-factor HW (G2++)** — T67; **Bermudan-in-HW** — T68; **HJM** — T69/T70; **the smile models** — T71.
- **CIR / square-root short rates** — not in this book's detailed treatment (ch 3.2 lists them in passing);
  `to_go_deeper` if ever pulled, needs a source that derives the non-central-χ² bond option.
- Calibration of Vasicek to a market curve — Vasicek has constant θ so it cannot fit an arbitrary initial
  curve (that is exactly what Hull-White's θ(t) is for); the model is the spine, the fit is HW's job.
