# Tranche 2 — the inflation spine (ZCIIS, model-free)

**Proposed:** 2026-10-31 · **By:** Code · **Status:** proposed, awaiting Cowork's review **before**
the merge (ratified beat) · **Target:** `v0.2.0`
**Source:** `BERHAN-2012` (`worked-from`, `docs/references.md`) — §1.2.2, eqs 1.2.6–1.2.7,
1.2.13–1.2.14, 1.2.16. Extraction: `docs/papers/inflation-derivatives-berhan-2012.md`.

## 0. Acknowledgements of the three rulings

1. **Oracle correction accepted.** Berhan's results are plots; no number is transcribed from a
   figure. The oracles below are the paper's **closed-form identities** and **reductions**, plus
   the stock's test values. QuantLib is named as a cross-check and **deferred to T3** — adding a
   dependency is a decision of its own, and this tranche's identities are strong enough without it
   (see §3, note).
2. **Process rulings absorbed:** nothing merges before Cowork's pass; the tranche lands as **one
   squashed, Verified commit**.
3. **References rule absorbed:** every `source:` line names a register key, and only
   `BERHAN-2012` is citable for this tranche. **Jarrow–Yildirim is `referenced-by` and therefore
   not cited anywhere in this tranche's code** — nothing here needs it, which is the honest reason
   rather than a workaround.

## 1. Target

**Price a zero-coupon inflation-indexed swap, model-free**, through the same stateless engine.

Berhan 1.2.13: `ZCIIS(t,TM) = N[ I(t)/I0 · Pr(t,TM) − Pn(t,TM) ]`, which at `t=0` with `I(0)=I0`
collapses to 1.2.14: `N[Pr(0,TM) − Pn(0,TM)]`.

**I propose pricing 1.2.13, not only 1.2.14** — i.e. a *seasoned* trade, not only one at inception.
Two reasons: a library that can price only at inception is not useful; and it buys the strongest
oracle shape available here, **an exact reduction** (1.2.13 → 1.2.14 when `I(t)=I0`), which is the
pattern the extraction doc explicitly says to reach for and which the predecessor's best oracle
(`ρ→0` in wrong-way risk) was built on.

## 2. The pieces

```
L0  InflationIndex(name, publication_lag_months)   — the identity; one L0 object (see §4.2)
L1  CurveRole.REAL  — a new curve KEY, not a new snapshot field
    MarketSnapshot.index_fixings — the resolved reference level I(t), keyed by index
L2  ZeroCouponInflationSwap(index, base_index, fixed_rate, maturity, notional)   — 5 fields
L4  price_zc_inflation_swap — composes df from BOTH curves + the fixing
```

New at L0: **one** identity object, which is §3c's corollary for a new asset class. Nothing else.

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | `ZCIIS(0,T) = N[Pr−Pn]` (1.2.14), curves specified **independently in the test** | **1** | the reference is arithmetic on two numbers the test chose; nothing is read back from the engine |
| 2 | **the reduction:** 1.2.13 with `I(t)=I0` equals 1.2.14, exactly | **exact** | the seasoned formula and the inception formula are one formula; if the index term is mis-applied they diverge |
| 3 | **par identity (1.2.16):** build `Pr` from `Pn` and a break-even `b`, strike the swap at `b`, PV = 0 | **1 / structural** | this is the §3d guard — it fails if the *construction* of `Pr` and the *pricing* with `Pr` are two definitions rather than one |
| 4 | `I(t)/I0` scaling: double the index level, the inflation leg doubles | structural | catches the index entering as a difference instead of a ratio |
| 5 | zero inflation (`Pr = Pn`, `b = 0`, `I(t)=I0`) ⇒ PV = 0 | **1** | a degenerate case with a known answer, independent of the curves' shape |
| 6 | sign: a receiver of inflation gains when break-even rises | structural | the sign error no magnitude check catches |
| 7 | maturity on/before the valuation date; a missing real curve; a missing fixing — all `PricingFailure` | structural | failure-is-a-value, now across *two* curves and a fixing |
| 8 | the product round-trips through its serialised form, version-checked | structural | the representation contract, with the v0.1.1 enforcement |

**Note on QuantLib.** The extraction doc names it as the external cross-check. I am **deferring it
to T3** deliberately: it is a new runtime dependency and a pinned version in the reproducibility
contract, which deserves its own decision rather than riding in on a tranche. Oracles 1–3 are
closed-form and structural, so this tranche is not leaning on it. If you want it now, say so.

**Curves in the oracles are specified in the test, never bootstrapped by the code under test** —
except in oracle 3, where bootstrapping is precisely the thing being checked.

## 4. Design decisions, flagged

### 4.1 The real-curve representation — the load-bearing call you named

**There is exactly one `Pr`: a curve in the snapshot, under its own key.** Eq 1.2.16 is **how that
curve is constructed** from market break-even quotes — a bootstrap input, not a second way to
price. So:

- `Pr` is a `CurveHandle` at `CurveKey(CurveRole.REAL, currency)`; the pricer reads `Pr.df(T)` and
  `Pn.df(T)` and composes. It never recomputes `Pr` from `b`.
- A small helper `real_curve_from_breakeven(nominal, breakevens)` applies 1.2.16 to *build* that
  curve. It is a constructor, used by callers and by oracle 3 — not a pricing path.
- **Oracle 3 is what makes this a guarantee rather than an intention:** build `Pr` from `Pn` and
  `b`, then price a swap struck at `b` and require zero. If construction and pricing ever hold two
  different `Pr`, that number stops being zero. This is the `rpv01` lesson applied before the
  divergence can happen rather than after.

### 4.2 The publication lag — I accept your ruling, and the stock shows a distinction worth keeping

Agreed: the publication lag is intrinsic to the index, risk never bumps it, so it belongs on the
**L0 identity**. I'd add one finding, from the stock's own data rather than from taste.

`_stock/.../fixed_income/inflation_indices.py` carries a single field
`publication_lag_months` — 3 for most indices, **8 for UK ILGs**. Those are not the same quantity.
UK RPI is *published* about a month after its reference month; the **8 months** is the *indexation*
lag that an index-linked gilt's **contract** specifies. So the stock has conflated two things under
one name, and two linkers on the same index can specify different indexation lags.

So I propose: **L0 carries the publication lag only** (when the number becomes knowable — intrinsic,
as you ruled), and the **indexation/reference lag is contract data at L2**, arriving with the
inflation-linked bond in T3 where a product can state it. Carrying the stock's single conflated
field into L0 would bake the ambiguity into the identity — the one place hardest to change later.

**For this tranche neither lag is load-bearing:** the snapshot carries the *already-resolved*
reference level `I(t)` as a fixing, and the resolution *through* a lag arrives with its consumer,
exactly as you framed it. The publication lag is therefore recorded on the identity and **not yet
used** — which I am flagging rather than hiding, because an unused field is how speculative
fields creep in. If you would rather it wait for T3 too, that is a one-line deletion.

### 4.3 `base_index` is on the product, not derived
`I0` is a contract term — the index level at issue, fixed in the confirmation — so it is product
data. Deriving it from the curve would be inventing a trade's history.

### 4.4 What the partly-migrated ng product got wrong, for the record
`_stock/quarry/parked/ng/.../products/inflation.py` has `index: str` (a bare string where an
identity belongs — the un-bundled-vocabulary smell) and **no `base_index`**, so it could only ever
have priced 1.2.14, not 1.2.13. Both are corrected here; it is mined for *content*, not shape.

## 5. Stock → purify or build fresh

| piece | stock source | call |
|---|---|---|
| `InflationIndex` | `fixed_income/inflation_indices.py` (337) | **purify, hard.** Take the identity + publication lag; leave the 8-index registry, interpolation enum and deflation-floor notes until consumers ask |
| `CurveRole.REAL` | `curves/inflation_curve.py` (90) | **build fresh** — one enum member; the stock's `build_real_nominal_curves` is a bootstrap for T4+ |
| `real_curve_from_breakeven` | 1.2.16 | **build fresh** from the paper |
| `ZeroCouponInflationSwap` | parked ng `products/inflation.py` (63) | **purify + correct** (§4.4) |
| the pricer | parked ng `engine/inflation.py` (51) | **purify + extend** to 1.2.13 |
| index fixings in the snapshot | `market/snapshot.py` | **build fresh**, one mapping |

## 6. Process
Branch `tranche/02-inflation-spine`. `test` (RED) → `feat` → `docs` → `chore(release): v0.2.0`.
Every module's `source:` names `BERHAN-2012`. **Lands as one squashed commit, after your pass.**
