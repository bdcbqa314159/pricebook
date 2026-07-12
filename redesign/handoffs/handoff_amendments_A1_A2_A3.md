# Hand-off to Claude Code — implement Amendments A1 / A2 / A3

**Context:** the design workspace amended the ratified contract while the build reached
`v0.0.10` (HW European swaption) on the *old* contract. So these are **refactors to
already-built code**, not just guidance for new slices. Source of truth = `CLAUDE.md`
(§0/§2/§3) + `redesign/02_spine.md` Amendments **A1/A2/A3** (already on disk; read them
first). Do **not** author a parallel design note — report findings via the handoff §5
drift section and Cowork ratifies.

**Branch names are descriptive** (not S05–S08 — those numbers are already used by shipped
build slices). Three ordered, oracle-gated slices off `main`:

---

```
Read CLAUDE.md §0/§2/§3 and redesign/02_spine.md Amendments A1, A2, A3. Then implement the
three refactors below, in order, each on its own branch off main, rebase-and-merge, red-
before-green where behaviour is added, verify.py all + ruff (PLR0913) green on Ubuntu+Windows.

BRANCH slice/amend-model-only   (Amendment A1) — behaviour-preserving
  - Change the engine to price(product, model, numerics). Market is NOT a peer arg.
  - build(snapshot) -> CalibratedModel: a model carries the MarketSnapshot it was calibrated
    to (model.market, frozen). Introduce DiscountingModel(curve) at L3 for linear products.
  - FixingHistory is first-class in MarketSnapshot (the economy = curves + fixings).
  - Refactor ALL existing engines/products/models built so far (cashflow, bond, HW swaption)
    and call sites (book.value, risk/DV01) to the new signature; risk bumps the snapshot and
    rebuilds the model.
  - Oracle: PVs byte-identical to pre-refactor (reuse existing oracles) + a binding test that
    a model can only price against its own snapshot. Version PATCH.

BRANCH slice/amend-temporal     (Amendment A2) — adds capability
  - Retire the "fail on past cashflow" guard; replace with segment-and-settle: partition each
    product's cashflows into past (settle), current (accrue), future (price).
  - PricingResult becomes a decomposition: dirty PV + cashflow/accrual breakdown => clean,
    accrued. valuation_date = model.market.valuation_date.
  - RED first, closed-form oracles: seasoned bond mid-life excludes the paid coupon; forward-
    starting prices only future flows; dirty = clean + accrued to <1e-12; cashflow exactly on
    valuation_date is historical. Version MINOR.

BRANCH slice/amend-hierarchy    (Amendment A3) — L6 shell
  - Rename the L2 atom instrument -> product. Trade (L6) holds a collection of products + a
    start date; Book = collection of trades; BookedTrade remembers the benefit table
    (realized-cash P&L, ties to quarry pnl_history). Never discount realized P&L.
  - Oracle: realized (benefit table) + mark (engine) reconcile to total economics over a
    trade's life; realized P&L = sum of actually-paid cashflows (no discounting). Version MINOR.

Honour all standing rules: max 5 args (bundle into value objects), tolerance-based oracles
(no bit-exact across OS), provenance headers, no silent suppression. After amend-hierarchy,
resume the forward migration on the corrected contract and emit a handoff report noting any
drift for Cowork.
```

---

*Note for Cowork:* the slice-plan labels S06–S08 in `redesign/04_slice_plan.md` should be
read as these three named refactors, not the build's numeric slices. Worth realigning the
slice plan to the build's real history on the next pass.
