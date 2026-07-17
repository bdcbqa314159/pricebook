# Cowork → Build rulings — CP-2b review

Reviews `redesign/handoffs/CP2b_checkpoint.md` (v0.42.0, 243 green). **Accepted — strong
checkpoint.** General-curve HW closed the biggest parity gap; oracles all closed-form.

## §4 rulings
1. **FRA forward composed in the engine** (no `curve.forward_rate`) — keep. Add
   `curve.forward_rate` as a curve building block **when OIS (2nd forward consumer) lands** in
   CP-2c. Rule-of-two.
2. **HW duck-types `instantaneous_forward`** — introduce a **`RateCurve` protocol**
   (`df` + `zero_rate` + `instantaneous_forward` + `forward_rate`) **in CP-2c**, where OIS is the
   2nd rate-curve consumer. Accept the duck-type until then (tracked).
3. **Date-based `forward_short_rate` / path sim** — ✓ confirmed.
4. **Deletable-bar rigor** — ✓ **RATIFIED as a standing rule** (now in CLAUDE.md §4): every parity
   slice ends by reading its quarry counterpart end-to-end and listing the residual; a module
   ticks to *deletable* only when the residual is empty.

## §5 smells — accepted
Duck-type → `RateCurve` in CP-2c; `_bracket_slope` O(n)-per-call ceiling noted (cache when a hot
loop bites); "create the slice branch first" process note. No debt.

## NEW — signature/field hardness (Cowork-initiated, ratified)
`PLR0913` guards function args but **not dataclass fields**; 8 ng dataclasses exceed 5 fields
(FRA 7, CDS/options/inflation 6; plus the legit-wide `MarketSnapshot` 6 and `XvaReport` 10).

- **`verify.py fields`** (new, merge-gate): a `products/` or `foundation/` value dataclass has
  **≤5 fields** unless it carries `# fields-exempt: <reason>`. (CLAUDE.md §3b updated.)
- **Exempt now** (add the marker): `MarketSnapshot` (A5 shape), `XvaReport`, `XvaReportConfig`
  (output record / config aggregates).
- **`slice/product-field-bundling`** (behaviour-preserving refactor): reduce the products by
  bundling into value objects that ALREADY EXIST — `Money` for `notional`+`currency`, `Accrual`
  (`foundation/cashflow.py`) for `accrual_start`+`accrual_end`+`day_count`. FRA 7→4; same for CDS,
  equity_option, commodity_option, inflation swap. Guarded by every existing product oracle
  (PVs byte-identical). Land `verify.py fields` green in the same slice.

## CP-2c (ruled): fixings-first
Confirmed. `FixingHistory`-consuming seasoned float unblocks the seasoned FRA/swap spine **and**
the L6 float-leg realized P&L (benefit-table gap). Plus **deposit** and **OIS/RFR compounding** —
OIS brings the `RateCurve` protocol (§4.2) and `curve.forward_rate` (§4.1). Each vanilla ends with
the deletable-bar parity confirmation against its quarry module.

Suggested order: `product-field-bundling` (quick, closes your flag) → fixings/seasoned-float →
deposit → OIS (+ RateCurve). Report per #11; refresh `quarry_reconciliation.md`.
```
