# Checkpoint — the volatility-derivatives arc (project ten, `KWOK-ZHENG`)

**Closed 2026-11-25 at `v0.81.0`, 3286 tests, eleven gates green.** The tenth vertical: the *products on*
variance and volatility, extending the variance/vol-swap base (T31) and the Heston Fourier machinery (T33/
T48/T49) project six left. Distinct from `GATHERAL-2006` (the vol *surface*); this is the vol *derivatives*.
Scope Bernardo set: **the bigger spine (~6 tranches)** — the core four plus a new stochastic-vol model (3/2)
and the time-changed-Lévy framework. Built in **six** tranches, T74–T79.

## What was built
- **T74 — generalized variance swaps** (gamma + corridor), static replication re-weighting T31's strip
  (`1/K` for gamma, restricted `1/K²` for corridor). `v0.76.0`.
- **T75 — options on *continuous* realized variance under Heston** + the Heston fair variance, which gave
  T74 the **independent cross-method anchor** it lacked (replication-of-Heston-skew vs the MGF closed form).
  The `SmileLike` extraction was seeded here. `v0.77.0`.
- **T76 — VIX derivatives** — VIX² as the variance-swap rate (**§1.5.2/eq 1.31 = T31's strip verbatim**),
  futures/options as 1-D integrals against the non-central χ² law of `V_T`, reduced by integration-by-parts
  to the **survival** function (F-8.6's deep-tail work paying off). `SmileLike` protocol extracted at L1.
  First arc `slow-test-exempt` (**OI-8**). `v0.78.0`.
- **T77 — the 3/2-model variance swap**, two independent routes to 8.1e-10: the reciprocal-CIR Poisson
  ladder (scipy-free production) and the paper's hypergeometric transform (oracle). `v0.79.0`.
- **T78 — time-changed Lévy variance swap + the discrete-monitoring adjustment** (a pricing tranche, per
  0401), the discrete→continuous O(1/N) rate as the adjustment's oracle. `v0.80.0`.
- **T79 — perpetual timer options**, the capstone — the MC first-passage timer, scope (C): `PricingResult`
  with SE, anchored by the exact `ρ=0,r=q=0` BS reduction + the §6.2.3 perturbation + the joint-law `ρ` pin
  + the exponential-martingale carry identity. Second arc exemption (**OI-9**). `v0.81.0`.

## What the arc reused
`integrated_variance_transform` (T48) · the `fair_variance` `1/K²` strip (T31) · `heston.py` + the Fourier
inversion (T33/T49) · `noncentral_chisquare` (T72/T77) · `black_price` · the T62 moment-approximation+MC
discipline · the T25 rate-detector. The vol-derivatives arc is the most reuse-heavy to date — most tranches
added an engine over existing machinery, not new L0.

## Method events worth carrying
- **Source defects (two this arc, both acquired/diagnosed, not quarantined blind):** `KWOK-ZHENG` eq (3.30)
  was a confluent-hypergeometric **argument-order typo**, resolved by acquiring `ZHAO-CUI-FLORESCU-2018`
  (the paper's eq (19) gives `L(0)=1` exactly); Prop 4.8 printed the **second moment** where `var(L₁)` is
  required, ruled by the drift-free-quadratic-variation requirement identity. Both in the `references.md`
  defect log.
- **My spec anchor-errors (four, all caught at build, none shipped):** T76 forward-window, T77 stationary
  mean=θ, T78 adjustment-vanishes-with-jumps, T79 `ρ`-non-degeneracy. All the same shape — pre-committing a
  limit/anchor's *value or property* instead of naming the obligation and letting the build measure it
  (obligation-not-form, on the spec-writer's side). The two-surface split caught every one.
- **T43 read generally (T79):** an oracle *set* can share a collective blind spot beyond scale-invariance —
  here every anchor forced `carry=0`. Second instance of the shared-blind-spot shape; vindicates T43's
  generality rather than adding a clause.
- **`SmileLike` protocol (T76):** the first cross-cutting §6 extraction of the arc — `fair_variance` now
  consumes an L1 protocol, `ImpliedVolSkew` and model-smile presenters both implement it.

## Deferred (the forward hooks)
- **§6.2.2 perpetual-timer CLOSED FORM → the numerical track's first pull** (`to_go_deeper`): needs complex
  `I_ν`, a Bromwich density-inverter, a 2-D quadrature — primitives the structural/numerical arc builds; it
  retro-anchors T79's MC when built. Eq (6.23) transcription settled.
- **C17 `implied_carry`** — a 5-site duplication that bit T78; re-audit #9's named duplication target.
- gamma/corridor MGF cross-checks (T75); finite-maturity/discrete timers (§6.3); direct VIX models (§3.3);
  the §5.3 sharper options-on-variance approximation — all `to_go_deeper`.
- **OI-8 + OI-9** — two heavy-MC slow exemptions, both with live-path/φ-sharing removal triggers; F-8.9 (the
  slow-ceiling margin) is Bernardo's, now three data points.

## Where this sits in the bigger plan
This closes the **literature** cadence for now. Per the 2026-11-24 direction conversation (Bernardo), the
next move after **re-audit #9** is the first **structural arc (R1 — representation & mark-to-market)**: the
Trade/Position/PricingRequest spine §0 promised and the arcs deferred, which turns the catalog-above-the-
pricers into a library and makes the whole verified catalogue desk-usable. FX waits as the literature arc
after R1 (the test that the representation is currency-general). The two-track rhythm (literature ⇄
structural, switched at the re-audit) and a purpose-defined 1.0 are drafted for a charter ruling at that
boundary.
