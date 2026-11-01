# Tranche 7 — the index series and seasoned pricing (valuation at t > 0)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.7.0`.
**Source:** `BERHAN-2012` (`worked-from`) — §2.5 eq **2.5.12** (the seasoned YoY form), §1.1.2
(indexation lag), §1.2 (the index `I(t)` as the FX-analogy spot). Rung-1 reduction anchor.

## 0. The gap, stated precisely (verified against the tree, not the coverage prose)

The ZCIIS (`engine/inflation.py`) and the linker already price the **seasoned** form: each reads
`I(t) = index_fixings[index]` and scales by `I(t)/I0`, which is eq 1.2.13 — at inception `I(t)=I0`
and it collapses to 1.2.14, but the general path is there. **The one inception-only instrument is
the YoY swap**: its current, in-progress period pays `I(Tᵢ)/I(Tᵢ₋₁) − 1` where `I(Tᵢ₋₁)` is an
*already-published* reset (eq 2.5.12, `ι(t) > 1`), and the single-scalar `index_fixings` cannot
supply a historical print. That is the gap.

The fix is one new market-data shape — an **index fixing series** — plus the **lag resolution**
that turns a date into the right print. The single current fixing becomes the degenerate case of
the series (one point), which is the reduction this tranche is anchored on.

## 1. The pieces, by layer

```
L0  InflationIndex gains `publication_lag` + `interpolation`  — see §3.1 (the pull returns them)
    lag resolution: (reference_date, lag, interpolation) → level from a series   — §3.2
L1  FixingSeries (reference-month → published level), keyed by index, in the snapshot
    — replaces the scalar `index_fixings`; the scalar is a one-point series
L2  InflationLinkedBond.indexation_lag — carried since T3, now READ for the first time
L4  YoY pricer: the seasoned form (2.5.12) for the current period; future periods unchanged
    ZCIIS / linker: resolve I(t) from the series via the lag, instead of a hand-fed scalar
```

Nothing new at L3 (no model change — this is all market data + the engines reading it). No new
product: the YoY swap and linker are the ones that gain seasoning.

## 2. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **the anchor — seasoned reduces to inception.** With no elapsed period (`ι(t)=1`, valuation at inception), the seasoned YoY (2.5.12) equals the inception YoY (2.5.13) already oracled in T4, exactly; and a ZCIIS resolving `I(t)` from a one-point series equals the T2 ZCIIS given that `I(t)` directly | **exact reduction** | if the seasoned path mis-handles the current period, or resolution returns a different `I(t)`, the two diverge |
| 2 | **companion (charter §4): a real elapsed period MOVES the price** vs inception — the seasoned path is not a no-op | structural | without it the reduction passes an implementation that ignores the historical fixing entirely |
| 3 | **lag resolution, pointwise:** the level resolved for date `d` equals the series print at `d − lag`, computed independently in the test | **1 — requirement** | an off-by-one-month in the lag, the defect this whole area is about |
| 4 | **interpolation has the power to fail:** daily (TIPS) vs flat/prior-month (UK ILG) tested on a **sloped** series, where the two give different numbers; a flat series cannot discriminate them (the T3 fixture-power lesson) | structural | a flat fixture would pass any interpolation, or none |
| 5 | **the current YoY period uses the HISTORICAL reset, not a forward** — `I(Tᵢ₋₁)` is the published print, not `Pr/Pn`-implied | structural | the seasoned-vs-forward confusion is exactly where 2.5.12 differs from 2.5.13 |
| 6 | **invariant 6 across the schedule:** fully-elapsed periods excluded (PV 0, shell's business); the in-progress period included with its historical reset | structural | the partition now has three cases, not two |
| 7 | **failure is a value:** a reference date before the series begins, or a missing series, → `PricingFailure` — never a guessed or extrapolated level | structural | "refusing to assume the base level", extended to the series |
| 8 | round-trip + version for any new serialised shape (the series, the lag/interp fields) | structural | the representation contract |

**Oracle 1 is the correctness anchor and oracle 2 its mandatory companion**, per the charter's
reduction clause. Oracle 4 is the charter's "domain must contain the failure" applied to a fixture:
a flat series is the wrong fixture exactly as a flat curve was in T3.

## 3. Design decisions (ruled by Cowork)

### 3.1 `publication_lag` and `interpolation` return to the `InflationIndex` identity — now that they have a consumer
T3 dropped `publication_lag` from `InflationIndex` because nothing read it — the pull discipline.
Seasoned resolution is its first real consumer, so it **returns now**, together with the
**interpolation** convention (TIPS daily-interpolate, UK ILG do not — a property of the index, not
the contract). This is the carried-field rule working forwards: the field arrives *with* the code
that reads it, not ahead of it. The **indexation lag** stays on the contract (`InflationLinkedBond`,
and it is read for the first time here) — two linkers on one index can specify different ones (the
T3 ruling), so it is contract data; the publication lag is the index's own.

### 3.2 Resolution is one shared atom, composed — not re-derived per pricer
`reference_level(series, date, lag, interpolation)` is defined once and every pricer composes it, so
the ZCIIS, the linker and the YoY swap cannot disagree about which print `I(t)` is — the §3d
shared-atom discipline, and the thing a cross-module convention bug would otherwise be (it goes in
the conventions register with its guard).

### 3.3 The scalar fixing becomes a one-point series, not a parallel field
`index_fixings: Mapping[index, float]` is replaced by the series; a single current print is a series
of length one. This keeps one representation of "what the index has printed", and makes oracle 1's
reduction (series → the old scalar behaviour) the proof the replacement is faithful rather than a
rewrite that could drift. The T2–T5 pricers change from "read the scalar" to "resolve from the
series", guarded by that reduction.

### 3.4 Not building
A full real-world fixing *calendar* (which business day a print lands on); seasonality adjustment
(a monthly seasonal curve — its own tranche if a product pulls it); any index beyond the
ACT/365F + monthly-print shape the thesis uses.

## 4. Stock
**Purify for content:** `_stock/.../inflation_indices.py` carries the lag/interpolation facts per
index (the `FLAT`/`LINEAR` enum, the 3-vs-8-month values) — mine the *facts*, not the conflated
single-field shape T3 already diagnosed. **Build fresh:** the series type and the resolution atom;
the stock's fixing handling is not in the ratified shape.

## 5. Process
Branch `tranche/07-seasoned-index-series`. `test` (RED) → `feat` → `docs` → `chore(release): v0.7.0`.
`source:` cites `BERHAN-2012` §2.5/§1.1.2 with equation numbers. First rung of finishing the book.
