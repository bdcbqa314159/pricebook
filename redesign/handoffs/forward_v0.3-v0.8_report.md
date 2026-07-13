# Handoff Report — forward migration on the corrected contract (v0.3.0 → v0.8.0)

Date: 2026-07-13   ·   Version: `0.8.0`   ·   Tests: 118 green (`verify.py tests --layer 6`)

Second build → design report (per `redesign/08_handoff_protocol.md`), covering the forward
work after the A1/A2/A3 amendments landed (`v0.2.0`). Everything below is built on the
corrected `price(product, model, numerics)` contract. Read §5 — it carries the rulings.

---

## 1. Slices landed

| Slice (branch) | Version | Layer | Oracle | Tol met |
|---|---|---|---|---|
| `risk-to-l5` (Pricable + generic greeks) | 0.3.0 | L5 | generic `dv01` == analytic (cashflow, bond); generic over a raw closure; swaption rebuilds the model | `< 1e-6` |
| `credit-hazard-bootstrap` | 0.4.0 | L1 | each CDS reprices to zero at its par spread | `< 1e-10` |
| `cds-product` | 0.5.0 | L2/L3/L4 | par CDS → 0 through the engine; matches L1 `cds_pv`; seller = -buyer | `< 1e-3` |
| `cds-credit01` (CS01) | 0.6.0 | L5 | buyer credit01 > 0; seller = -buyer; matches an independent hazard FD | `< 1e-9` |
| `swaption-mc` | 0.7.0 | L4 | MC converges to the S08 Jamshidian analytic (payer + receiver); exact at σ=0; reproducible | `~2% at 200k paths` |
| `float-leg-fixings` | 0.8.0 | L4 | seasoned swap uses the realized fixing; already-paid period excluded; missing fixing fails; spot unchanged | `< 1e-6` |

All red-before-green (behaviour-preserving refactors run the existing oracles; e.g. the
`coupon_bond_cashflows` extraction in `swaption-mc` kept the S08 analytic oracle green).

## 2. Ledger deltas (new-tree entries)

| Entry | Layer | Quarry provenance | Notes |
|---|---|---|---|
| `risk/pricable.py` (`Pricable`, `discounting_pricable`, `hull_white_pricable`) | L5 | `risk/` | the `snapshot -> PV` protocol; replaced the Slice-0 `risk/dv01.py` |
| `risk/greeks.py` (`bump_rate`, `dv01`) | L5 | `risk/` | generic bump-and-reprice |
| `market/survival_curve.py` (`SurvivalCurve`, CDS leg math, `bootstrap_survival_curve`) | L1 | `core/survival_curve.py` | mirrors the S03 discount bootstrap |
| `models/credit_model.py` (`CreditModel`) | L3 | `models/` credit | carries market + survival + recovery |
| `products/cds.py`, `engine/cds.py` | L2/L4 | `credit/`, `pricing/` | CDS product + engine |
| `risk/credit_greeks.py` (`bump_hazard`, `credit01`) | L5 | `risk/` | CS01; does NOT go through `Pricable` (see §5.2) |
| `engine/swaption_mc.py` (`SwaptionMCEngine`); `NumericalConfig.mc_paths/mc_seed` | L4/L0 | `numerical/_mc.py`, `pricing/` | T0-forward MC; **stdlib `random`, not the quarry MC module** (§5.4) |
| `products/swap.py` `FloatLeg.day_count`; `engine/swap.py` fixings | L2/L4 | `fixed_income/` | consumes `FixingHistory` (A1) |

## 3. Oracles used
Closed-form self-consistency throughout, except the MC slice (convergence to the analytic
within a few standard errors, seeded/deterministic). Credit uses reprice-to-par (S03
pattern). Greeks use analytic-vs-FD (dv01) and independent-FD / sign / symmetry (credit01).

## 4. Debt logged this layer
- **None.** `verify.py debt` = 0. Two tentative `# type: ignore` markers (HW curve cast,
  Pricable `_pv`) were removed rather than logged — no type checker runs in CI, so the
  annotations are simply left slightly unsound with a runtime `assert`/guard.

## 5. Design drift — for Cowork to rule

1. **The hazard curve lives in `CreditModel`, not the `MarketSnapshot` — drift from A1.**
   A1 ratified: *market data (curves + fixings) lives in the snapshot; the model is built
   from it; risk perturbs the snapshot and rebuilds.* The hazard/survival curve **is** market
   data (the credit curve), but we put it in `CreditModel(market, survival, recovery)`, not in
   the snapshot. Consequence: **`credit01` cannot use the `Pricable` protocol** (which bumps
   the *snapshot*) — it is a bespoke `credit01(cds, model)` that bumps `model.survival` and
   calls `CDSEngine` directly. **Ruling needed:** either (a) promote `SurvivalCurve` to a
   first-class member of `MarketSnapshot` (alongside `discount_curve` + `fixings`), so credit
   greeks flow through the same `Pricable` as `dv01`, or (b) accept per-model-family greek
   functions and document that the `Pricable` covers only snapshot-borne risk factors. I lean
   (a) — it restores the A1 principle and unifies greeks — but it touches `MarketSnapshot`.

2. **Pricing math at L1 (recurring L1/L3/L4 boundary).** The CDS leg math (`RPV01`,
   protection PV, `cds_par_spread`, `cds_pv`) lives in **L1** `survival_curve.py`, reused by
   the bootstrap (L1) *and* the L4 `CDSEngine` — exactly as the S03 discount bootstrap put
   par-swap math at L1. Same soft spot as the HW analytic ZCB-option formula on the L3 model.
   **Ruling:** ratify "curve-construction pricing math may live at the curve's layer and be
   reused upward by the engine," or insist all pricing math sits at L4 (which would duplicate
   or invert the bootstrap dependency).

3. **Multiple engines per product → the registry/facade is now overdue-ish.** A swaption has
   two engines (`SwaptionEngine` analytic, `SwaptionMCEngine`). Selection is explicit today.
   The spine's engine-registry open-question will need to key on (product, model, **method**).
   Not blocking; flag for when the facade lands.

4. **MC used stdlib `random`, not the quarry `numerical/_mc.py`.** Demand-migration pulled the
   smallest thing that works (a seeded `random.Random`, no numpy) rather than migrating the MC
   toolkit module. Variance reduction / Sobol / a general path engine are deferred. Confirm
   this is the intended "migrate the minimum" reading (I believe it is per `ng-migration-mode`).

5. **Swap-level accrued/clean not reported.** A2 gives *fixed* legs an accrued/clean
   decomposition; `SwapEngine` returns only the dirty NPV (the float current period accrues but
   isn't surfaced). Minor; add when a consumer needs swap accrued.

6. **Still pending from the last report:** realign `redesign/04_slice_plan.md` numbering to the
   build's real history (these slices use descriptive branch names + descriptive provenance
   tags, not S-numbers).

## 6. Quarry status
Unchanged crossings this round are new-tree construction on the corrected contract; L0 toolkit
remains demand-migrated (`norm_cdf`, `bisect_root` earlier; MC via stdlib this round — §5.4).

## 7. Ready for next?
- Forward migration on the corrected contract: healthy, 118 green, no debt.
- Recommended next: **rule §5.1** (hazard curve placement) before more credit/greeks, since it
  decides whether credit risk unifies under `Pricable`. Then a new asset class (FX / equity) or
  curve refinements (dual-curve OIS discount, quarterly CDS) each as an oracle-gated slice.
- Blockers: none. Questions for design: §5 items 1–6.

---

### One-line return message (paste into Cowork)

> Forward work v0.3→v0.8 landed (risk→L5 Pricable, credit curve+CDS+CS01, swaption MC vs
> analytic, seasoned-swap fixings); 118 green, no debt. Drift: §5.1 hazard curve sits in
> CreditModel not the snapshot (so credit01 bypasses Pricable) — rule promote-to-snapshot vs
> per-family greeks; plus L1-pricing-math boundary, multi-engine registry, stdlib-MC,
> slice-plan renumber. See redesign/handoffs/forward_v0.3-v0.8_report.md.
