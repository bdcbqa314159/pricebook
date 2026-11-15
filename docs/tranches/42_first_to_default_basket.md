# Tranche 42 — the first-to-default basket (homogeneous loss)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.49.0`.
**Source:** `OKANE-2008` ch 15.1–15.2 (pricing default baskets; modelling first-to-default baskets),
on the §12.2.4 basket-survival-curve construction. **Pre-registered RED.** Fifth unit of the credit arc
and **the first product priced through the copula** — the copula (T41), the CDS legs (`credit_legs`,
T39) and the entity keying (T41b) all meet here.

## 0. The product worth having
**The first-to-default basket: protection that triggers on the FIRST of N names to default.** A
CDS-like contract on a basket of `N_C` credits — the protection leg pays `(1−R)` at the first default
(if before maturity), the premium leg pays a spread `S₁` on a risky annuity that knocks out at that
first default. O'Kane's method (§15.2) is exact and reuses everything: build the **basket survival
curve** `Q_B(t) = Pr(no name defaults by t)`, then **price it as a CDS** with the T39 leg analytics.

```
Conditional on the market factor Z the names are independent, so the conditional basket survival is
the product of the names' conditional survivals:
    Q_B(t | Z) = Π_i (1 − p_i(t|Z)) = Π_i Q_i(t|Z)                          (15.x, conditional independence)
    Q_B(t)     = ∫ Q_B(t|Z) φ(Z) dZ                                        (integrate over the market factor)
Then, exactly as a single-name CDS on Q_B (reuse credit_legs, T39):
    Premium PV    = S₁ · RPV01(Q_B)
    Protection PV = (1−R) ∫ Z(t) (−dQ_B(t))
    Par FtD spread S₁ = Protection PV / RPV01(Q_B)
```

**Homogeneous loss** (`L_i = L = (1−R)` for all names, §15.1) — the simpler case, and the scope here;
inhomogeneous loss (per-name recovery, tracking the defaulter's identity) is `to_go_deeper`.

## 1. The pieces and their layers
- **L3 `models/` (or `market/`) — the basket survival curve `Q_B`.** Computed from the `GaussianCopula`
  (T41's conditional survivals `Q_i(t|Z)`) by the product-over-names, integrated over `Z`
  (Gauss-Hermite, reused, T41/T12b). A named, oracled quantity (a survival curve the legs consume),
  not a new primitive — it *is* a `SurvivalCurve`-shaped object (monotone, `Q_B(0)=1`), so it can feed
  `credit_legs` directly.
- **L2 `products/` — `FirstToDefaultBasket`.** Frozen: the `ReferenceEntity` list (T41b), the premium
  schedule, the contractual spread `S₁`, notional, maturity, recovery `R` (homogeneous). ≥ 2 names
  (a one-name basket is the reduction, not a product to encode specially). Prices nothing.
- **L4 `engine/` — the FtD pricer**, `@register(FirstToDefaultBasket, GaussianCopula)`. Builds `Q_B`,
  then the premium/protection legs via **`credit_legs`** (reused whole — the FtD is a CDS on `Q_B`),
  returns value + par spread. **Deterministic** (the market-factor integral), so no SE; the Monte-Carlo
  is the *cross-check oracle*, not the engine (the T41 pattern — the exact integral prices, the MC
  verifies).

## 2. The oracle (RULED)
The three exact limits O'Kane gives are the closed-form anchors; the **single-name reduction** is a
cross-product to T38; a **requirement sandwich** `Π Q_i ≤ Q_B ≤ min Q_i` survives transcription; the MC
is the non-degenerate anchor with ρ pinned where observable.

- **Single-name reduction (`N_C = 1`) — a cross-product to T38 (§4, different code).** A one-name basket
  has `Q_B = Q₁`, so the FtD basket prices **exactly** as the single-name CDS (T38) — the FtD engine
  reconciled against the CDS engine on the same name. The cleanest statement that the basket machinery
  reduces to what it generalises.
- **The two correlation limits — exact reductions (§4), O'Kane §15.2.** (a) **Independence `ρ=0`:**
  `Q_B(t) = Π_i Q_i(t)` (the market factor drops out). (b) **Comonotone `ρ=1`:** `Q_B(t) = min_i Q_i(t)`
  (all defaults driven by `Z`; the basket survives iff the most-default-prone name survives, so the FtD
  is the widest name). Both exact. **Companion (non-degenerate):** `0<ρ<1` with **distinct** names
  (distinct `β_i`, distinct `Q_i`), where `Q_B` sits strictly between the two bounds — asserted material,
  and the T41 distinct-name discipline carries (equal names hide an index bug in the `Π`).
- **The requirement sandwich (survives a transcription error, §4).** For all parameters:
  `Π_i Q_i(t) ≤ Q_B(t) ≤ min_i Q_i(t)` — the lower bound is independence, the upper is "the basket
  defaults no later than its first name", and they are the `ρ=0`/`ρ=1` endpoints, so the sandwich is
  both a requirement and the monotonicity-in-ρ check. Plus `Q_B` monotone decreasing, `Q_B(0)=1`, and
  the **FtD spread between `max_i S_i` and `Σ_i S_i`** (a widest-name floor and a sub-additive cap — a
  finance requirement no `Q_B` transcription can fake).
- **The Monte-Carlo cross-check — the non-degenerate anchor, ρ pinned where observable (§4, T17).**
  Simulate `A_i = β_iZ + √(1−β_i²)ε_i`, take `τ_FtD = min_i τ_i`, price the FtD leg, within `k·SE`.
  Because `ρ_ij = β_iβ_j` enters the factorisation of the `A_i`, pin it by the **realised correlation of
  the simulated `A_i`** (T17), not only by the basket spread. Assert the MC's own `E[τ_FtD]` / basket
  survival matches `Q_B` within SE before trusting it as the price anchor.
- **The market-factor integral — rate-asserted (§4, T41/T25).** `Q_B = ∫ Π_i Q_i(t|Z) φ(Z) dZ` by
  Gauss-Hermite; assert its **convergence rate** in the node count (and the unit-β step issue T41 fixed
  does not recur here — the product of smooth conditional survivals is smooth in `Z` for `|β_i|<1`;
  state the `|β_i|=1` handling, reusing T41's analytic limit if a name is at full loading).
- **Mutation (prove power).** Each must die: drop a name from the `Π` (→ the single-name reduction still
  passes at `N_C=1` but the companion + MC die — assert `N_C≥2`); use `p_i` not `(1−p_i)` in the
  conditional survival (→ `Q_B` inverts, every anchor dies); swap `min`/`Π` in the two limits (→ the
  reductions die); drop the `(1−R)` or the `Q_B` in the protection leg (→ reuses `credit_legs`, whose
  own oracles T39 already pin, but assert the FtD composition). **Premises:** `N_C≥2`, distinct names,
  `0<ρ<1`, semi-annual schedule (τ=1 history), term-structured curves.

## 3. Process
Branch `tranche/42-first-to-default-basket`. `test` (RED: the single-name reduction + the two
correlation limits + the `Π Q ≤ Q_B ≤ min Q` sandwich + the spread bounds + the MC with the realised-ρ
pin + the rate-asserted integral + the mutation table) → `feat` → `docs` → `chore(release): v0.49.0`.
**Open the draft PR early; pre-register the reductions, the sandwich and the spread bounds first**
(closed-form/requirement, no implementation needed). **Transcribe the `Q_B` construction (§15.2, §12.2.4)
in both renderings**; reuse `credit_legs` for the legs (confirm the reuse is clean — the FtD *is* a CDS
on `Q_B`, so no leg re-transcription). **Re-time `slow`** — the MC + the per-`Z` product over names is new
cost (T19). `source:` `OKANE-2008`; `crosscheck: quantlib` — QuantLib's `NthToDefault` / basket machinery
is a real target. **Raise an `Ask:`** if the basket-survival object wants to live at L1 (`market/`, a
produced curve) vs L3 (`models/`, a copula output) — the §2 calibration-split question (it *produces* a
survival curve from a model, so likely L3 output consumed as market-data-shaped; rule it at build if the
import direction forces a choice).

## 4. What this tranche is NOT (scope fence)
No **nth-to-default** for `n>1` (second-to-default etc., §15.3 — the same `Q_B` machinery with
`Pr(n(t)<n)`, a clean later pull, `to_go_deeper`). No **inhomogeneous loss** (per-name recovery, tracking
the defaulter's identity — §15.1, `to_go_deeper`). No **multi-factor** copula (§15.5) or **Student-t**
(§15.6) — `to_go_deeper`. No **tranches** (T43). No **basket risk/hedging** (§15.7). T42 is the one
product: the homogeneous-loss first-to-default basket, priced as a CDS on the copula's basket survival
curve, oracled by the reductions + the sandwich + the MC.

---

## 5. Cowork's ruling on T42 — (reserved for §10 step 4, after the build)
