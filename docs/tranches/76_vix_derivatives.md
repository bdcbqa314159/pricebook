# T76 — VIX derivatives: VIX² as the forward variance-swap rate, VIX futures and options

**Cowork spec, 2026-11-22.** Third tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`;
`docs/vol_derivatives_arc.md`). The pivot from *realized* variance (T74/T75, backward-looking) to the
*forward* variance-swap rate: **VIX² is the forward variance-swap rate**, so this tranche reuses the same
transform and the same replication strip pointed at a *future* window, and adds VIX futures and options off
the conditional law of the VIX index. It also discharges ruling 0401(2): **the `SmileLike` protocol is
extracted here**, because this is the second consumer of "present a model's smile to T31's strip".

## Target
Price, at valuation date 0, for a VIX window `[T, T+Δ]` (Δ = 30/365 for the index):
- the **VIX² level** `VIX²(0; T, T+Δ)` — the forward variance-swap rate — two ways (Heston closed form and
  the model-free replication strip), as the engine behind the futures/options;
- a **VIX future** `E₀[VIX_T]` and a **VIX call/put** `E₀[(VIX_T − K)⁺]`, each
  `price(product, HestonModel) → PricingResult`.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **Kwok–Zheng ch 3** in `-raw` and `-layout` and reconcile (§4):
- **§3.1 (and §3.1.1 for the jump correction)** — VIX² as the forward variance-swap rate, and its closed
  form under the (jump-)affine variance dynamics. This is the load-bearing identity.
- the **VIX-futures and VIX-options** sub-sections of ch 3. *(Sourcing honesty: the arc plan pins ch 3 /
  §3.1.1 for the VIX²/variance-swap-rate identity, confirmed from the TOC at 0397, but I did **not**
  re-pin the exact sub-section number for the futures/options pricing here. Confirm it from the TOC in
  `-raw`+`-layout` at step 1; if ch 3 does not carry the futures/options closed form where expected,
  raise an `Ask:` rather than transcribing from the analogous T75 structure — the session's standing
  method-watch: derive the oracle from THIS source's section, never from the prior tranche.)*
- **§3.3 (direct VIX modelling)** is **out of scope** — T76 takes the consistent / variance-swap-rate
  route (the model-free VIX² plus Heston), not a stand-alone VIX SDE (arc plan's deferred list).

Transcribe the VIX² closed form and the futures/option forms pointwise (the F-6.1 rule); the spec names the
obligation, not the form (§4).

## Pieces, layer, reuse
- **The VIX² engine** (new, L4) — the forward variance-swap rate under Heston, i.e. the *forward-window*
  expected annualised integrated variance `(1/Δ)·E₀[∫_T^{T+Δ} V_s ds]`. This is a **reuse-and-compose** of
  T75's machinery: `integrated_variance_moments` / `heston_fair_variance`
  (`engine/realized_variance_option.py`) and `integrated_variance_transform` (`engine/volatility_swap.py`)
  give the `[0,s]` integrated-variance mean; the forward window is the **difference** of two such integrals
  (`[0,T+Δ]` minus `[0,T]`), affine in `V₀`. The new code is at most that forward-window arrangement.
- **The model-free VIX² strip** (reuse, for the cross-check) — VIX² is *also* the `1/K²` replication strip
  (T31/T75's `fair_variance`) over the forward window / the forward smile. This is the independent-method
  anchor, the same shape as T75's oracle 1: Heston's forward smile → the strip → compare to the Heston
  closed form. **It is the second consumer of the strip-takes-a-model-smile pattern → extract the
  protocol (below).**
- **VIX future + VIX option** (new products L2 + engine L4) — under Heston, `VIX²_T = a·V_T + b` is affine in
  the terminal variance `V_T` (the same forward-rate coefficients), so `VIX_T = √(a·V_T + b)` is a
  deterministic function of `V_T`, and `E₀[VIX_T]` and `E₀[(VIX_T−K)⁺]` are **one-dimensional integrals
  against the law of `V_T`** (non-central χ² / the transform). The products: a `VIXFuture` and a `VIXOption`,
  frozen dataclasses (window `T`, `Δ`, strike, notional; ≤5 fields, §7 — bundle the window into a value
  object if over). Siblings of the realized-variance products, **not** subclasses (the §6 real-leg-vs-YoY
  line — a forward-looking index payoff is a different accrual).
- **The `SmileLike` protocol extraction** (ruling 0401(2), §6 rule-of-two at the second consumer) —
  extract a minimal `@runtime_checkable SmileLike` protocol, the single real member
  `volatility(strike, forward, years)` (**memberful**, per T33b — an empty protocol is satisfied by
  everything and could not reject a non-smile), and retype `fair_variance` / `generalized_fair_variance`
  to consume it. `ImpliedVolSkew` implements it as-is; T75's Heston subclass **collapses into a presenter**
  in the same change (the C16 raise-at-review pattern). **Layer:** the protocol lives where
  `ImpliedVolSkew` does — **L1 `market/`** — so the L4 strip depends *down* on it and the L3/L4 model-smile
  presenters implement a lower-layer protocol (downward dependency, clean). Grep `acyclic`/`layers` after.
- **The MC companion** reuses `heston_mc` — simulate `V_T`, form the realized VIX² / the forward integrated
  variance, price the future and option against the closed forms.

## Oracles (RED-first, §4)
1. **The VIX²/variance-swap-rate identity — a REQUIREMENT oracle** (it is forced by the *definition*: VIX²
   is the forward variance-swap rate, so it survives a transcription error the source-value oracle would
   share — the F-6.1 reasoning). Transcribe it and assert it pointwise.
2. **The cross-METHOD anchor (the T75 oracle-1 shape, one window forward).** The Heston VIX² closed form
   == the model-free `fair_variance` strip fed Heston's **forward** smile, different code each side.
   **Measure the grid per weight** (the arc lens — VIX's forward-window integrand is not T31's or T74's or
   T75's; do not inherit a width/node count, measure this one, and assert whether the strip converges from
   above or below so the agreement is not two cancelling errors, as T75 did).
3. **The forward-window composition — a cross-product oracle.** The forward fair variance equals the
   `[0,T+Δ]` expected integrated variance minus the `[0,T]` one (compose T75's `integrated_variance_moments`
   a *different* way than the engine does), exact to machine precision — a within-Heston check independent of
   the strip.
4. **VIX future = `E₀[VIX_T]`**, and the **convexity requirement** `E₀[VIX_T] ≤ √(VIX²(0))` (Jensen on √,
   forced by the finance — a requirement oracle, **strict** under genuine vol-of-vol, an equality at
   `σ_v → 0`). The futures value via the 1-D integral against the law of `V_T`, cross-checked by the MC
   companion (banded `k·SE + measured error`, resolution stated, §4).
5. **The VIX option — MC companion measures any approximation error** (if the option uses a quadrature or a
   moment match; if it is the exact 1-D integral, the MC is the independent cross-check). Band per §4,
   resolution stated, swept where it is non-degenerate; **put–call parity** on VIX options.
6. **Reductions (§4/T39):** at **`σ_v → 0`** (deterministic variance) VIX² ⇒ the deterministic forward
   integrated variance (exact, with a genuine `σ_v > 0` companion), and the convexity gap
   `√(VIX²) − E[VIX]` ⇒ 0; the VIX option at `σ_v → 0` ⇒ a deterministic payoff.
7. **The protocol extraction is a pure refactor** — prove it by the §4 pure-refactor rule: the strip's
   output on an `ImpliedVolSkew` is **bit-identical** before and after the retype (op order preserved — it
   is a type change, not an arithmetic one), checked in-process; the committed regression keeps its existing
   tolerance. T75's cross-method oracle must stay green through the collapse of the Heston subclass into a
   presenter.

## Degeneracy premises (§4 / §9)
- The cross-method and convexity fixtures use a **genuine vol-of-vol and a non-flat forward skew** — under
  flat vol the strip/closed-form agreement is trivial and the Jensen gap is zero (the T74 flat-skew / the
  σ_v → 0 degeneracies). State it; prefer a mutation (drop the forward-window offset so VIX² reads the spot
  window — the **translation-invariance** trap, T17: a forward window that is *the same length* as a spot
  fixture makes `[0,T+Δ]−[0,T]` look like `[0,Δ]` unless the mean-reversion makes the two windows carry
  **different** expected variance, so the fixture needs `V₀ ≠ θ` — mean-reverting away from the long-run
  level — or the offset mutation survives).
- The `SmileLike` extraction's **non-degeneracy**: supply the protocol an object that is **not** a smile and
  assert it is rejected (the T33b input-population rule — the memberful protocol must actually discriminate;
  an empty `@runtime_checkable` protocol passes everything).
- The protocol purity premise: the bit-identity is a **measurement, in-process** (T49) — the committed test
  carries a tolerance, not the bit patterns.

## Provenance / gates
- Header `source: KWOK-ZHENG §3.1 (+ §3.1.1 jump correction; + the futures/options sub-section once pinned)`;
  `oracle:`; `crosscheck:` (QuantLib / a known library has no standard VIX-under-Heston engine — likely
  `crosscheck: none — <reason>`, but check before asserting it).
- `references.md`: `KWOK-ZHENG` is already `worked-from` (T74); no flip needed.
- Version → **v0.78.0**; one floor bump; one `CHANGELOG` entry. **Re-time the Heston MC / the VIX quadrature
  on BOTH CI runners (§9)** — this is the seventh file in the arc's heavy-numerics family; measure CI, not
  the local clock (F-8.9's running concern — a lossless pass before an exemption).

## Out of scope (carried / deferred → T77+)
- **Direct VIX modelling** (§3.3) — a stand-alone VIX SDE; `to_go_deeper` (the arc plan's deferred list).
- **The 3/2-model** — T77 (a new stochastic-vol model); VIX here is Heston-only.
- **VIX term structure / the futures curve across windows** — one window this tranche; a multi-window curve
  is a product-pull away, `to_go_deeper` if not needed.
