# Claude Code directive — S06 (engine → model-only) + S07 (temporal valuation)

Run AFTER S05 (signature bundling). Read CLAUDE.md §0, §2 (esp. the amended contract +
invariant 6), §3 (Models / Time semantics), and redesign/02_spine.md Amendments A1 & A2.
Two sequential slices, each its own branch, each oracle-gated.

---

## S06 — engine depends on model, not market  (branch slice/06-engine-model-only)

```
Amend the stateless engine to price(instrument, model, numerics) — market is NO LONGER a
peer argument.

1. feat(models): introduce CalibratedModel — a model carries the MarketSnapshot it was
   calibrated to, exposed as `model.market` (frozen). Introduce DiscountingModel(curve) at
   L3: a thin model for linear products whose "calibration" adopts a discount curve; it
   exposes model.market and the curve via CurveHandle. Real consumers: every linear product
   (rule of two) — not ceremony.
2. refactor(engine): DiscountingEngine.price(instrument, model, numerics) reads the curve
   through model, not a market arg. Update ALL call sites (Slice-0 path, bond, book.value,
   risk DV01). book.value(date) builds/binds the model for that date's snapshot, then prices.
3. Guard: a model can only price against its own snapshot — make model/market mismatch
   structurally impossible (there is no second market to pass). Add a test asserting the
   engine has no market parameter and that PVs are byte-identical to pre-refactor values
   (behaviour-preserving) — reuse the existing Slice-0 and bond oracles.
4. Risk: a market bump now flows through re-deriving/rebinding the model (bump the snapshot →
   rebuild DiscountingModel → reprice). Update DV01 accordingly; oracle unchanged
   (analytic vs finite-difference DV01 < 1e-6).
5. verify.py all + ruff (PLR0913) green on both OSes; docs/provenance; CHANGELOG (Changed);
   __version__ PATCH; rebase-and-merge.
```

## S07 — valuation is temporality-aware  (branch slice/07-temporal-valuation)

```
Make the engine reason about start / valuation / payoff dates. valuation_date =
model.market.valuation_date.

1. feat(engine): PV = Σ over cashflows with date > valuation_date of amount·DF(valuation→date).
   Cashflows with date ≤ valuation_date are HISTORICAL — excluded from PV (never discounted
   with non-positive t). Do not mutate the instrument; partition at pricing time.
2. feat(foundation/vocabulary): make accrued interest and clean vs dirty price explicit
   (dirty = clean + accrued). Seasoned coupon accrues from period start to valuation_date.
3. test FIRST (RED before GREEN — hard rule), closed-form oracles:
   - seasoned bond valued mid-life: PV excludes the already-paid coupon; matches the
     closed-form sum over remaining flows.
   - forward-starting instrument (start > valuation): prices only future flows.
   - accrued/clean/dirty: dirty − clean == accrued, to < 1e-12.
   - a cashflow exactly on valuation_date is treated as historical (boundary case, documented).
4. Fixings: reset dates ≤ valuation_date use realized FixingHistory; > valuation forward-implied.
   (Only wire what a present instrument needs — no speculative fixing machinery.)
5. verify.py all + ruff green both OSes; docs/provenance; CHANGELOG (Added/Changed);
   __version__ MINOR (new capability); rebase-and-merge.
```

---

These two amend the ratified §2 contract (spine Amendments A1/A2). After S07, resume the
forward migration (S1 day-count onward) on the corrected engine.
