# Tranche 38 — the credit default swap (premium + protection legs, par spread, RPV01)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.45.0`.
**Source:** `OKANE-2008` ch 5–6 (the CDS and its valuation model), with ch 3.10's credit triangle as
the reduction anchor. **Pre-registered RED.** First unit of the credit-derivatives arc (project five) —
the single-name product the XVA arc deliberately deferred. `OKANE-2008` flips `in-hand → worked-from`
here (the `references` gate enforces it).

## 0. The product worth having
**The single-name credit default swap, marked and par-quoted off a survival curve.** The canonical
credit derivative: buyer pays a running spread `S` on a risky annuity; seller pays `(1−R)` at default.
Two legs, a mark-to-market, and a par spread:

```
Premium leg PV   = S₀ · RPV01(t,T)
RPV01(t,T)       = Σₙ Δ(tₙ₋₁,tₙ)·Z(t,tₙ)·½(Q(t,tₙ₋₁) + Q(t,tₙ))                         (6.4, accrual-on-default, midpoint)
Protection PV    = (1−R)·½·Σₖ (Z(t,tₖ₋₁)+Z(t,tₖ))·(Q(t,tₖ₋₁) − Q(t,tₖ))                  (6.x, trapezoidal)
V(t)  (long prot)= Protection PV − S₀·RPV01(t,T)
S_par(t,T)       = Protection PV / RPV01(t,T)
```

where `Z` is the discount factor (our discount curve) and `Q` the survival probability (**our
`SurvivalCurve`, T25**). The practitioner forms (6.4, the trapezoid) are *discretisations* of the exact
integrals (6.3: `RPV01 = ∫ … Z·Q + ∫ … Z(−dQ)` for the accrual; protection `(1−R)∫Z(−dQ)`), with a
stated `O(r/M)` error — so the exact integral is the anchor and the discretisation's error is measured
(§4), never one flat tolerance.

## 1. The pieces and their layers
- **L2 `products/` — `CreditDefaultSwap`.** Frozen data: the premium schedule (payment dates +
  accruals, the `Term`/schedule vocabulary already in the tree), the contractual spread `S₀`, notional,
  maturity, and the protection direction (buyer/seller — its own enum, §6, not reusing a call/put).
  Prices nothing. Recovery `R` is **not** a product field — it is a market/model assumption (the
  standard 40%), carried on the model.
- **L3 `models/` — the credit model (thin).** Carries the `MarketSnapshot` (which must hold the
  `SurvivalCurve` Q and the discount curve Z — §3, the engine reaches market only through
  `model.market`) plus the assumed **recovery rate `R`** (a model parameter, since it is an assumption
  not a quote; `0 ≤ R < 1`). No new survival machinery — the hazard/survival is T25's, reused whole.
  Fields ≤ 5.
- **L4 `engine/` — the CDS pricer**, `@register(CreditDefaultSwap, CreditModel)`. Returns a
  `PricingResult` carrying the value; the **par spread and RPV01** are named surface (separate
  functions or result fields) so the bootstrap (T39) and the risk (T40) can reach them without
  re-deriving — the §4 "extract a buried step into named surface" rule, applied before the consumer
  exists because two named consumers (T39, T40) are already in the committed queue (rule-of-two).
  The integration step count `M` is numerical config arriving with this first consumer (§3.5).

## 2. The oracle (RULED)
The flat-hazard **credit triangle** is the exact closed-form reduction; the protection leg is
cross-checked against the **T25 CVA integral** (different code, different arc); the discretisation error
is measured against the exact integral; par-reprice is the telescoping guard.

- **The credit triangle `S_par = λ(1−R)` — the preferred anchor (§4), exact in the flat limit.** Under
  a **flat hazard `λ`, flat rate, and the continuous limit** (`M → ∞`, or the exact integral), the par
  spread is **exactly** `λ(1−R)` (O'Kane 3.10, derived: `Protection = λ(1−R)∫ZQ` and `RPV01 = ∫ZQ` in
  the continuous no-accrual limit, so their ratio is `λ(1−R)`). A closed-form reduction, exact by
  construction. **Companion (non-degenerate):** a **term-structured** (non-flat) hazard and a
  **non-flat** discount curve, where `S_par ≠ λ(1−R)` and the full schedule matters — asserted material,
  with the survival curve read beyond its first segment (the T25/F-3.1 non-flat premise, **baked in from
  the start** given the τ=1 history — the premium accruals `Δ` must be non-unit too).
- **The protection leg = the CVA integral (cross-product oracle, §4 — different code, different arc).**
  `(1−R)∫Z·λQ dt` is `LGD·∫DF·λS dt` with EPE ≡ 1 — exactly what `risk/credit_valuation_adjustment.py`
  / `mc_exposure_cva` (T25/T26) compute. The CDS protection leg reconciled against the XVA CVA integral
  on a flat unit exposure is a cross-arc check no within-CDS oracle can replace (the T3/T5 cross-product
  precedent, now across arcs). This is also the capstone seam — project three's integral reused.
- **The discretisation error is MEASURED, not toleranced (§4 banded-resolution + the [L,U] bracket).**
  O'Kane gives the protection leg's **lower and upper Riemann sums `L, U`** explicitly; the trapezoid
  must sit **inside `[L, U]`** (a requirement for all parameters) and converge to the exact integral at
  the stated `O(r/M)` rate — assert the *rate*, not a flat band (the T25 convergence-plateau lesson).
  The exact integral (fine `M`, or the flat closed form) is the reference; state the smallest pricing
  error the committed `M` resolves.
- **Requirement oracles (survive a transcription error).** `RPV01 > 0` and **decreasing in the hazard**
  (more default risk shortens the expected annuity); the **protection leg increasing in the hazard**;
  `V(S_par) = 0` by construction (the par-reprice — a **telescoping** self-consistency guard, kept but
  not the correctness anchor, §4); `V` linear and decreasing in `S₀` (`∂V/∂S₀ = −RPV01`); the
  accrual-on-default term **non-negative and bounded** by one premium period. The accrued-coupon
  midpoint form (6.4) is an **approximation** to the exact accrual (6.3) — assert its error against the
  exact integral (O'Kane Fig 6.1: it lowers the breakeven spread), never assume it.
- **Cross-source check (§4's same-formula caveat).** `CAPINSKI-ZASTAWNIAK` ch 4 also gives a CDS par
  spread; **confirm it renders the *same* formula** as O'Kane ch 6 before treating it as an independent
  transcription (they may differ in the accrual/recovery convention — a measured gap to assert, not
  assumed cross-source independence — the T17 lesson). Where they agree, it is a second rendering.
- **Mutation (prove the oracle has power).** Each must die: drop `(1−R)` (→ triangle + CVA cross-check
  die); drop the accrual-on-default term (→ dies on non-unit `Δ`, the baked-in premise); use `Q(tₙ)`
  instead of the `½(Q_{n-1}+Q_n)` average (→ dies on term-structured hazard, NOT on flat — the
  degenerate-fixture premise); swap `Z` and `Q` roles (→ triangle dies); drop a discount factor in the
  trapezoid (→ [L,U] bracket + convergence die). **Premises asserted:** `λ` term-structured, `Δ ≠ 1`,
  `Z` non-flat, survival read past segment one.

## 3. Process
Branch `tranche/38-credit-default-swap`. `test` (RED: the credit-triangle reduction + the CVA cross-
product + the [L,U] bracket & convergence rate + the requirement identities + the CZ cross-source check
+ the mutation table) → `feat` → `docs` → `chore(release): v0.45.0`. **Open the draft PR early;
pre-register the credit triangle and the requirement identities first** (they need no implementation).
**Transcribe RPV01 (6.4) and the protection trapezoid in both renderings** (`-raw` + `-layout`), and use
O'Kane's **derivation** of the credit triangle (3.10, derived in steps) as the independent check on the
leg formulas (§4's stronger rung). `source:` `OKANE-2008` (flips to `worked-from`); `crosscheck:
quantlib CreditDefaultSwap / IsdaCdsEngine` (a real target — QuantLib implements the ISDA CDS model).
**Raise an `Ask:`** if the premium schedule needs a convention our `Term`/schedule vocabulary does not
carry (the ISDA accrual/business-day details — rule which are in-scope vs deferred), or if the recovery
`R` belongs on the market rather than the model (a §3 placement question).

## 4. What this tranche is NOT (scope fence)
No **CDS bootstrap** — inverting par quotes for `Q` is T39 (L1, the T8 round-trip pattern). No **risk /
CS01** or **forward CDS / swaption** — T40. No **upfront/standardised-coupon** CDS conventions (ch 6.7,
a later convention pull) beyond what par pricing needs. No **digital/loan CDS** (ch 6.8–6.9). No
**multi-name** anything (phase 2, T41+). No **bonds / asset swaps** (ch 4, `to_go_deeper`). T38 is the
one vanilla CDS: two legs, a mark, a par spread, off a reused survival curve.

---

## 5. Cowork's ruling on T38 — (reserved for §10 step 4, after the build)
