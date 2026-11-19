# T60 — the commodity forward (cost-of-carry + convenience yield) + the commodity option

**Cowork spec, 2026-11-19.** First tranche of the commodity arc (project eight, `CLARK-2014`;
`docs/commodity_arc.md`). Deliberately the **lightest** tranche: the genuinely-new object is the commodity
**forward** with a convenience yield; the option engine **reuses `black_price`** (Black-76 already exists).
`CLARK-2014` flips `in-hand → worked-from` here.

## Target
Price a European option on a commodity spot/futures under Black-76, with the forward built from
cost-of-carry and convenience yield — `price(CommodityOption, model) → PricingResult`.

## Pieces, layer, reuse
- **The commodity forward** (new) — `F(T) = S·exp((r − δ)·T)`, `r` the risk-free rate (from the discount
  curve), `δ` the **convenience yield** (storage benefit net of cost). Clark ch **2.1.2** (forwards) and
  **2.2.3** (Black-Scholes with convenience yield). Layer: a forward constructor reading the discount curve
  + a `δ` — **L1/L2** (it produces market-data-like forwards from a curve and a carry parameter; Code rules
  the exact layer by what it reads and produces, §2 — a pure `S,r,δ,T → F` value is L2 product data, a
  curve-reading constructor is L1). Keep the convenience yield a **carried field only if constitutive**
  (§6) — it is (it distinguishes two otherwise-identical forwards), so it is exercised by the round-trip.
- **The commodity option product** (new) — a frozen L2 dataclass: underlying forward reference, strike,
  expiry, call/put (`omega` via the existing `OptionDirection`/`CapFloorKind` protocol — **reuse**, do not
  redefine). ≤5 fields (§7); bundle into a value object if over.
- **The engine** (new, thin) — **reuses `from pricebook.models.black import black_price`** exactly as
  `engine/equity_vanilla.py` does. Black-76 total variance is `σ²·T`; the engine assembles
  `(forward, strike, σ²T, omega)` and projects the discounted `black_price` into a `PricingResult` (§3).
  **L4.** No new Black.

## Oracles (RED-first, §4; transcribe from `CLARK-2014`, NOT from this spec — §5)
Read the forward and the convenience-yield Black-Scholes form from Clark ch 2.1.2 / 2.2.3 in **`-raw` and
`-layout`** and reconcile (§4 dual rendering). Then:

1. **Pointwise transcription of the forward** `F = S·e^{(r−δ)T}` — independent reference (the test computes
   its own `e^{(r−δ)T}`), **not** read back from the code. This is the one load-bearing new formula, so it
   gets its pointwise oracle (§4 "every load-bearing formula transcribed pointwise at least once").
2. **Reduction: δ = 0 ⇒ pure cost-of-carry** `F = S·e^{rT}` (exact by construction). **Companion required**
   (§4 — a reduction a no-op also satisfies is vacuous): a **non-zero δ** case checked against the pointwise
   transcription, so the reduction is not the only witness. The companion must be non-degenerate in **δ**
   specifically (T39 — δ=0 is exactly what the reduction kills).
3. **Reduction: a single-fixing / European option ⇒ `black_price` directly** (exact) — the commodity engine's
   output must equal a direct `black_price(forward, strike, σ²T, omega)` call on the same forward. This is
   the companion proving the engine is a faithful Black-76 wrapper, not a re-derivation.
4. **Requirement oracle: put–call parity** `C − P = df·(F − K)` — survives a transcription error the
   source-value oracles might share (§4). Assert across a strike sweep.
5. **Cross-source check (optional, §4 same-formula caveat):** Clark's Black-76 and our `black.py`
   (`BERHAN-2012`) render the **same** Black-76 — confirm they agree (they must, it is the same object), so
   this is a consistency guard, **not** independence. Note it as such; do not claim cross-source independence
   where none exists (the T17 lesson).

## Degeneracy premises to assert (§4 / §9 catalog)
- The δ=0 reduction's companion is **non-degenerate in δ** (T39) — state it.
- Use a **non-trivial `T`** (not `T=1`, where `(r−δ)·T` hides a factor) and a **non-flat** `(r,δ)` so no
  mutation dropping `T` or swapping `r`/`δ` survives (the τ=1 catalog — assert the premise, prefer a
  mutation to a reading).
- Put–call parity at a **single strike = forward** is degenerate (C=P); sweep strikes off the money.

## Provenance / gates
- Header `stock:` / `source: CLARK-2014 §2.1.2, §2.2.3` / `oracle:` / `slice: tranche 60` on every new
  `src/` module (§8). A `crosscheck:` marker on any new `market/`/`models/`/`engine/` module (§8 — real
  QuantLib target for the forward/Black-76, or `crosscheck: none — <reason>`).
- `references.md`: flip `CLARK-2014` to `worked-from` (it gains a `src/` citer here); `verify.py references`
  then holds.
- Version → **v0.62.0**; one floor bump; one `CHANGELOG` entry.

## Out of scope (carried / deferred)
- American options, Asians, spread options, the Schwartz model — later tranches (T61–T64).
- Stochastic / seasonal convenience yield — `to_go_deeper` (a deterministic δ is the spine).
- Any market-specific calibration (ch 3–9) — `to_go_deeper`/`out_of_method`.
