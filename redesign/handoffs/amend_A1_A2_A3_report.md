# Handoff Report — Amendments A1 / A2 / A3 (engine contract + valuation model)

Date: 2026-07-12   ·   Version: `0.2.0`   ·   Branches (landed): `slice/amend-model-only`,
`slice/amend-temporal`, `slice/amend-hierarchy`

This is a **build → design** report (per `redesign/08_handoff_protocol.md`), returning the
three ratified amendments now that they are implemented against the already-shipped code
(the build had reached `v0.0.10`, HW swaption, on the *old* contract). Read §5 first — it is
the part that asks Cowork for rulings.

---

## 1. Slices landed

| Slice (branch) | Amendment | Version | Oracle | Tol met |
|---|---|---|---|---|
| `slice/amend-model-only` | A1 — engine depends on model, not market | 0.0.11 | binding test (no `market` arg; model prices only its own snapshot) + unchanged PVs | exact / behaviour-preserving |
| `slice/amend-temporal` | A2 — temporality-aware valuation | 0.1.0 | seasoned bond excludes paid coupons; forward-starting; `dirty = clean + accrued`; cashflow on valuation is historical | closed-form `< 1e-6` |
| `slice/amend-hierarchy` | A3 — Product/Trade/Book + benefit table | 0.2.0 | realized = Σ paid (undiscounted); realized + remaining = total; end-of-life realized = total & mark = 0; Book aggregates | exact `< 1e-6` |

All three red-before-green (A1/A3 also carry a behaviour-preserving refactor commit). Full
suite: **88 tests green** at `verify.py tests --layer 6`; `acyclic` / `debt` / `provenance` /
`version` / ruff `PLR0913` green.

## 2. Ledger deltas (new-tree entries the amendments introduced)

| New-tree entry | Disposition | Quarry provenance | Notes |
|---|---|---|---|
| `models/discounting_model.py` (`CalibratedModel`, `DiscountingModel`) | new | `core/pricing_context.py` | the "built-market-state" input; model carries `market` |
| `market/snapshot.py` (`FixingHistory`) | new (vocabulary only) | `core/fixings.py` | first-class in `MarketSnapshot`; **no consumer yet** (see §5) |
| `foundation/cashflow.py` (`Accrual`) | new | `fixed_income/fixed_leg.py` | coupon accrual period + day count |
| `foundation/results.py` (`PricingResult.accrued`, `.clean`) | extended | ad-hoc returns | decomposition, not scalar |
| `shell/booking.py` (`Trade`, `Book`, benefit table) | new / re-homed | `core/book.py` + `pnl_history` | realized P&L, never discounted |
| L2 `instruments/` → `products/`; `FixedCashflowTrade`→`FixedCashflow` | rename | — | "Trade" freed for L6 |

## 3. Oracles used
- **A1:** behaviour-preserving — reused every S00–S08 oracle; added a structural binding test
  (`inspect.signature` has no `market`; a higher-rate snapshot alone lowers PV). No numerical
  reference beyond the existing closed forms.
- **A2:** closed-form. Seasoned 5Y semi-annual bond valued mid-period: engine dirty PV equals
  the independent discounted sum over the 6 remaining flows; accrued = coupon·0.25/0.5 = 10,000
  on the test bond; `clean = dirty − accrued`.
- **A3:** exact nominal identities over the trade's life (issue → mid-life → post-maturity);
  `Book` linearity across two trades.

## 4. Debt logged this layer
- **None.** No suppressions added (the one tentative `# type: ignore` in `HullWhite.curve` was
  removed in favour of a runtime `assert isinstance(..., FlatDiscountCurve)` — see §5.6).
  `verify.py debt` = 0.

## 5. Design drift — for Cowork to rule

1. **Slice-plan numbering is stale (documentation).** `redesign/04_slice_plan.md` labels the
   amendments S06–S08 and says "the forward migration (S1 day-count onward) re-slots behind
   these." But S1–S8 (day-count, curve, bond, swap, HW, swaption) are **already built and
   merged**; the amendments are refactors *on top* of them. Build used descriptive branch names
   (`slice/amend-*`) and tagged provenance with the amendment id (`A1`/`A2`/`A3`). **Ask:**
   realign the slice-plan doc to the build's real history, and read "resume forward migration"
   as *continue past the swaption* (original plan S6 CDS, S7 risk→L5, S8+ asset classes), **not**
   restart at S1.

2. **`FixingHistory` is a ratified type with no consumer.** A1 added it first-class to
   `MarketSnapshot` per directive, but nothing reads it: A2's seasoned bond is fixed-rate, and no
   seasoned **float** product exists. It is an empty frozen type + `.get()`, no resolution
   machinery (honouring "no speculative fixing machinery"). **Ask:** accept the ratified-but-idle
   type until a seasoned-float slice, or defer the type itself to that slice?

3. **Per-product model dispatch is deferred.** `Trade.value` builds one `DiscountingModel` and
   prices every product through the `DiscountingEngine`. That is correct only while all products
   are linear. A trade mixing families (e.g. a bond **and** a swaption) needs the engine/model
   **registry/facade** the spine anticipates (open question #3). **Ask:** confirm the registry
   lands with the first genuinely mixed trade, not before.

4. **L3/L4 boundary for analytic-model option formulas.** HW's ZCB-option value
   (`zero_bond_option`) lives on the **L3 model** as a closed-form property of the dynamics (like
   `B(t,T)`), while the **L4 `SwaptionEngine`** binds the *product* to it via Jamshidian. This
   mirrors QuantLib's `discountBondOption`, but it is a soft spot vs. "pricing lives in L4."
   **Ask:** ratify the boundary (model exposes analytic building blocks; engine prices products),
   or move analytic option values into an engine.

5. **`PricingResult` decomposition is minimal.** It carries `pv` (dirty) + `accrued` ⇒ `clean`.
   The vocabulary specced it to also hold a **per-cashflow breakdown, sensitivities, and
   diagnostics** — not added (no consumer). **Ask:** fine to grow these fields on demand?

6. **Clean/dirty & flat-curve HW.** (a) `accrued` is **nominal (undiscounted)** and
   `clean = dirty − accrued` (market convention) — confirm that is the intended semantics.
   (b) `HullWhite` still carries a concrete `FlatDiscountCurve` (runtime `assert` in `.curve`);
   the general (bootstrapped) HW fit is deferred — flag for when a bootstrapped-curve model is
   needed.

7. **L5 risk is still a stub.** Only `dv01` exists (bump-the-snapshot / rebuild-the-model /
   reprice, now correct under A1). The spine's L5 (greeks/XVA/RWA on a `Pricable` protocol) is
   unbuilt. **Ask:** is "relocate risk to L5 on the `Pricable` protocol" the next structural
   slice (original plan S7), ahead of more asset classes?

## 6. Quarry status
Unchanged by the amendments (these are new-tree refactors, not quarry crossings). The L0
numerical toolkit remains demand-migrated: `norm_cdf` (S07) and `bisect_root` (S08) landed;
the rest is pulled as products need it (ruling `ng-migration-mode`, 2026-07-12).

## 7. Ready for next?
- Amendment thread A1/A2/A3: **complete**, contract corrected, all green.
- Recommended next (build's real position, post-swaption): either **risk → L5** (structural,
  spine open item) or **CDS + hazard bootstrap** (next asset class), each its own oracle-gated
  slice on the corrected `price(product, model, numerics)` contract.
- Blockers: none. Questions for design: §5 items 1–7.

---

### One-line return message (paste into Cowork)

> Amendments A1/A2/A3 landed at v0.2.0 (engine→model, temporal valuation, Product/Trade/Book +
> benefit table); 88 green; drift: slice-plan numbering stale + 6 rulings in §5 (idle
> FixingHistory, deferred model-registry, L3/L4 analytic-option boundary, PricingResult fields,
> clean/dirty semantics, L5-risk-is-next). Quarry unchanged. Ready to resume forward past the
> swaption. See redesign/handoffs/amend_A1_A2_A3_report.md.
