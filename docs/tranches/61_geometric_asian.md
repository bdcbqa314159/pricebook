# T61 — geometric Asian options, exact closed form

**Cowork spec, 2026-11-19.** Second tranche of the commodity arc (project eight, `CLARK-2014`;
`docs/commodity_arc.md`). The first *exotic*: an average-rate option, priced exactly because the
**geometric** average of lognormals is itself lognormal — so it is Black-76 on an adjusted forward and an
adjusted variance, reusing T60's machinery whole.

## Target
Price a European-style **geometric-average** commodity option over a schedule of fixings —
`price(CommodityGeometricAsian, CommodityBlackModel) → PricingResult`. The **discrete** Kemna–Vorst form
(ch 2.7.3) is the spine (real Asians fix on a schedule); the continuous form (ch 2.7.1) is an optional
limit check.

## Pieces, layer, reuse
- **The product** (new, L2) — a frozen `CommodityGeometricAsian`: the underlying `CommodityAsset` (T60),
  strike, direction (`CommodityOptionKind`, **reuse**), notional, and a **fixing schedule** (the averaging
  dates). It is a **sibling** of `CommodityOption`, not a subclass — the payoff is the average, a different
  idea (§6, the real-leg-vs-YoY-leg rule, same call just made for the two Black models). Keep ≤5 fields
  (§7): the schedule is one field (a tuple of dates / a `Term`-like object); bundle if it pushes over.
- **The engine** (new, thin, L4) — computes the geometric average's **adjusted forward** `F_G` and
  **adjusted total variance** `σ²_G` in closed form, then **reuses `black_price(F_G, K, σ²_G, kind)`** and
  projects into a `PricingResult` (§3), discounted on the model's curve. No new Black; the only new code is
  the two adjustment formulas. Reuses the T60 forward (`CommodityAsset`/convenience yield) for each fixing's
  forward.
- **Model:** `CommodityBlackModel` (T60) — reused unchanged (flat vol; the term-structure vol is T64's).

## The mathematics (re-transcribe from the source — do NOT build from this spec, §5)
The geometric average `G = (∏ᵢ F(tᵢ))^{1/n}` over `n` fixings is lognormal, so `ln G` is Gaussian with a
mean and variance that are closed-form sums over the fixing times. Read **Clark ch 2.7.3 (Kemna–Vorst,
discrete)** — the adjusted forward `F_G` and the adjusted variance `σ²_G` — in **`-raw` and `-layout`** and
reconcile (§4 dual rendering). The variance carries the **double sum** `Σᵢ Σⱼ min(tᵢ,tⱼ)` (the covariance
of the log-prices), which is the term a mutation will most plausibly get wrong — so it is the load-bearing
formula to transcribe pointwise. The spec deliberately does not restate the formulae; the oracle is the
source, not this doc.

## Oracles (RED-first, §4)
1. **Pointwise transcription of `F_G` and `σ²_G`** — independent reference (the test forms its own
   `Σ min(tᵢ,tⱼ)` double sum from the fixing times), not read back from the engine. This is the one new
   load-bearing formula; §4 requires it transcribed pointwise at least once (the T59/F-6.1 lesson — a
   formula that only ever appears inside a composite is never actually pinned).
2. **Reduction: a single fixing (`n = 1`) ⇒ the T60 European** exactly — `F_G = F(t₁)`, `σ²_G = σ²·t₁`, so
   the price equals a direct T60 commodity-option call on the same expiry. Exact by construction.
   **Companion required (§4, and T39):** the reduction degenerates the averaging, so it is blind to the
   whole double-sum structure — a **genuine multi-fixing** case (`n ≥ 3`, spread fixing times) checked
   against the pointwise transcription, so the anchor is not vacuous. The companion must be non-degenerate
   in exactly what `n = 1` kills: **multiple, distinct fixing times**.
3. **MC companion on the geometric average** (reuse `foundation/mc_estimator.py`) — simulate GBM forwards,
   form the geometric average per path, price by discounted payoff; the closed form must sit inside the
   band. Band per §4: `k·SE + 0` (the closed form is exact, so the only error is the MC SE — state the
   resolution, the smallest defect a green rules out at the committed path count, and add a higher-path
   power test). This is the oracle that exercises the real averaging the `n = 1` reduction cannot.
4. **Requirement oracle — variance reduction by averaging:** `σ²_G ≤ σ²·t_n` (averaging strictly reduces
   variance for `n ≥ 2` with distinct times), with **equality iff all fixings coincide** — assert both the
   inequality (general schedule) and the equality (degenerate schedule = the `n = 1` reduction). Survives a
   transcription error the source-value oracles might share.
5. **Requirement oracle — put–call parity** on the adjusted forward: `C − P = df·(F_G − K)`, swept off the
   money (parity at `K = F_G` is degenerate, C = P).
6. **Optional — the continuous limit** (ch 2.7.1): as `n → ∞` on a fixed window the discrete Kemna–Vorst
   price converges to the continuous-averaging closed form; a convergence check (by **rate**, the T25
   detector, since it is an order limit not a value) is a free companion if cheap. Mark optional — do not
   let it gate.

## Degeneracy premises to assert (§4 / §9 catalog)
- The `n = 1` companion is **non-degenerate in the fixing count and spacing** (T39) — state it; a fixture
  with all fixings at expiry is the `n = 1` case in disguise and must NOT be the companion.
- **Distinct, non-uniform fixing times** and a **non-trivial `T`** — the double sum `Σ min(tᵢ,tⱼ)` is
  translation-/scale-sensitive, so a mutation dropping a cross term or mis-indexing the min must change the
  number (the T17 time-decay lesson: decay/weight in the actual times, not index distance).
- Prefer a **mutation to a reading** on the double-sum (drop a cross term, swap `min`→`max`, drop the
  `1/n²`); a surviving mutant is the hole.

## Provenance / gates
- Header `source: CLARK-2014 §2.7.3` (and §2.7.1 if the continuous limit is built); `oracle:`;
  `crosscheck:` marker (QuantLib has a geometric-Asian analytic engine — a real target, or
  `crosscheck: none — <reason>`).
- Version → **v0.63.0**; one floor bump; one `CHANGELOG` entry.

## Out of scope (carried / deferred → T62+)
- **Arithmetic** Asians (Turnbull–Wakeman) — T62 (the next tranche; geometric is the exact base it leans on,
  including the `geometric ≤ arithmetic` ordering oracle).
- Average-**strike** Asians, discrete-fixing weighting schemes — `to_go_deeper` unless pulled.
- Term-structure / seasonal vol in the averaging — T64's model territory.
