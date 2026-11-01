# Tranche 8 — the nominal curve bootstrap (§5.1), the cross-check presence gate, and the dispositions

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.8.0`.
**Source:** `BERHAN-2012` (`worked-from`) §5.1 — the nominal discount curve bootstrapped from par
swap rates ("recovered from bootstrapping the nominal discount curve", p. 5). Single-curve, as the
thesis is.

This tranche carries three things that belong together: the §5.1 feature, the cross-check
**presence gate** (ruled in 0045), and the **per-module dispositions** that gate needs. All land in
one PR with the parked charter §8 edit and the §2 correction below.

## 0. The feature — bootstrap the nominal curve from par swaps

Today the nominal curve arrives **built** (handed to the snapshot). §5.1 constructs it: given par
swap quotes `S(Tᵢ)`, solve for the discount-factor pillars that reprice each swap to par. This is
**model-free market-data construction** — quotes in, a `DiscountCurve` out, no model, no engine.

## 1. Layer — L1, and the charter §2 line is corrected to say so

The bootstrap is **L1** (`market/`), beside T2's break-even bootstrap. It imports only L0 (the
solver seam, day counts, `Money`) and L1 (`DiscountCurve`); it does **not** touch a model or the
engine. §2 currently calls a curve bootstrap "L3" and names T2's break-even inversion as the
example — but that lives in `market/inflation_curve.py` at **L1**, so the clause misdescribes its
own example. **Charter §2 corrected** (on disk, lands with this PR): a *curve bootstrap* (quotes →
curve, L0+L1 only) is **L1**; *model-free parameter* calibration is **L3**; *engine-inverting*
calibration is **L4**. The split is by what the calibration produces and reads, not by taste.

## 2. Pieces

```
L0  (reuse) foundation/solvers.py — the sequential root-find per pillar
L1  a par-swap quote representation (schedule + the quoted par rate) — minimal, ≤5 fields
    market/nominal_bootstrap.py — quotes → DiscountCurve, composing ONE annuity atom
    the annuity/`Σ τ·df` atom, shared — so a future vanilla-IRS pricer composes the same one (§3d)
```

No new product priced through the engine; no model; nothing above L1.

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **the anchor — known-curve round trip, pointwise.** From a known `DiscountCurve`, compute the par swap rates it implies **by an independent formula in the test**, bootstrap a curve from those rates, and recover the known `df` at every pillar to 1e-12 | **1** | non-telescoping: the test computes the par rates itself, so a wrong bootstrap recovers the wrong curve |
| 2 | **companion: a sloped par-rate input yields a NON-flat curve** | structural | without it, oracle 1 passes an implementation that returns flat / ignores the later quotes |
| 3 | **reduction:** a single flat par rate ⇒ the curve whose constant zero reproduces it, closed form | **1** | the degenerate case with a known answer |
| 4 | **consistency guard (telescopes — kept as a guard, not the anchor):** the bootstrapped curve reprices the input quotes to par | self-consistency | guards that construction and the par formula share one annuity (§3d); cannot see a shared convention error, which is why oracle 1 is the anchor |
| 5 | **negative rates do not crash — failure is a value.** A negative par rate bootstraps to a `df > 1` fine; a genuinely unsolvable pillar returns `PricingFailure`, never a raise | structural | the stock's bootstrap hard-codes a `[1e-6, 1.0]` bracket and raises on a negative-rate root — **do NOT inherit it** (recorded in `_stock`'s own audit); the bracket must admit negative rates |
| 6 | monotone/duplicate/empty quotes → `PricingFailure` | structural | ill-posed input is a value, not an exception |

Identifiability is **not** a concern here as it was in T6: a sequential bootstrap is *exactly
determined* — pillar `i` is fixed by quote `i` given pillars `< i`, a triangular Jacobian — so there
is no valley. Oracle 1's pointwise recovery is the proof; no multi-start needed. State that in the
doc so the absence of a T6-style discriminator is a reason, not an omission.

## 4. The cross-check presence gate + the 7 dispositions (ruled)

The parked §8 edit (presence gate, explicit exemption over `market/`/`models/`/`engine/`) lands
**with its implementation** here. The `provenance` gate gains the presence check (a filter on the
`_src_modules()` loop it already runs). The dispositions Code asked me to rule — a target or
`none — <reason>`, my call per §8's criterion:

| module | disposition |
|---|---|
| `market/curve.py` | **`crosscheck: quantlib InterpolatedZeroCurve / log-linear zero interpolation`** |
| `market/nominal_bootstrap.py` (new) | **`crosscheck: quantlib PiecewiseYieldCurve / par-swap bootstrap`** |
| `market/curve_set.py` | `crosscheck: none — a keyed container of curves, no numerics` |
| `market/snapshot.py` | `crosscheck: none — a frozen market-state value bag, no numerics` |
| `models/discounting_model.py` | `crosscheck: none — carries the MarketSnapshot, computes nothing` |
| `engine/cash.py` | `crosscheck: none — PV = N·df; the df convention is cross-checked at market/curve.py` |
| `engine/jy_calibration.py` | `crosscheck: none — inverts the cap/floor forward map (tagged at engine/capfloor.py); QuantLib's inflation vol is surface-based, not a JY-to-floor calibrator` |
| `engine/registry.py` | `crosscheck: none — product→pricer dispatch, no numerics` |

The `jy_calibration` reason is a positive statement about QuantLib's design, not "I am unaware of
one" — the inverse has no distinct counterpart because QL's inflation vol is surface-based; its
forward map (the cap/floor) is where the external check lives, and that is tagged.

## 5. Stock
**Mine for content:** the stock's nominal bootstrap for the par-swap par formula and the sequential
method. **Do NOT inherit** its `[1e-6, 1.0]` bracket and uncaught `brentq` raise — the audit in
`_stock` records it crashes on negative rates, which violates failure-is-a-value (oracle 5 exists
because of it). Build the solve on the `foundation/solvers.py` seam with a bracket that admits
negative rates.

## 6. Process
Branch `tranche/08-nominal-bootstrap` (the §8 presence-gate commit is already parked there). `test`
(RED) → `feat` → `docs` → `chore(release): v0.8.0`. One PR carries: the bootstrap, the presence
gate + the 7 dispositions, the parked §8 edit, and the §2 correction. `source:` cites
`BERHAN-2012` §5.1.
