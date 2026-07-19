# Artifact #19 — Market Data: the cross-asset design (rev 2, post-adversarial review)

**Status:** Ratified. Cross-asset — every future asset class consumes it, so it is settled *before*
the multicurve build. Supersedes A5's snapshot shape. **Rev 2** incorporates the adversarial pass;
the five findings and their resolutions are recorded in §8 so the reasoning is not lost.

---

## 1. Two layers — quoted and built (RATIFIED)

```
QuoteSet         raw observations — deposit/par-swap rates · basis spreads · FX spots
    │            carries `as_of`; retains each quote's RAW external identifier (§6)
    ▼  calibrate  →  CalibrationResult (quotes used · residuals · convergence · Jacobian)
MarketSnapshot   the state pricing reads — built curves/surfaces AND passed-through data
                 carries `valuation_date`
```

**Honest caveat (finding 1):** the snapshot is **not** purely "built" — FX spots pass through
unchanged and fixings are raw observations. It is *"everything pricing needs"*, some built, some
passed through. Do not pretend otherwise.

### The dual-role rule (finding 1, RATIFIED)
Some data is **both** a calibration input and a direct pricing input — FX spot builds xccy curves
*and* prices an FX forward. Bumping it in each role gives a different number. The rule:

> **The `QuoteSet` is authoritative. A datum appears in the snapshot only as a *derived* copy.**
> A bump is applied **at the `QuoteSet`, and everything downstream is rebuilt** — so an FX-spot bump
> moves both the xccy curves and the forward, consistently. Snapshot-level bumps
> (`bump_curve`/`bump_pillar`) exist only for **zero/pillar risk**, which is *defined* as a
> perturbation of the built curve holding quotes fixed.

Two risk families, deliberately distinct and both correct: **par/market risk** = bump quotes,
rebuild. **Zero/pillar risk** = bump the built curve. Never mix them in one number.

## 2. The snapshot: CLOSED shapes × OPEN keys (RATIFIED — finding 3)

The meta-rule, one level up. Market data has a **finite set of shapes** and an **open set of keys**:

| shape (closed → field) | what it is | examples |
|---|---|---|
| **term structure** | a curve over time | discount · projection · survival · inflation · repo · dividend-yield |
| **surface** | value over two axes | vol (strike × expiry) |
| **scalar** | a single number | FX/equity/commodity spot · recovery · **correlation (keyed by a pair)** |
| **series** | observations over past dates | fixings · historical spots |
| **schedule** | discrete dated events | **discrete dividends** · reference-bond coupons |

```python
@dataclass(frozen=True)
class MarketSnapshot:          # fields-exempt: aggregate — the closed shape set
    valuation_date: date
    curves:    CurveSet                      # term structures  (§3)
    surfaces:  Mapping[SurfaceKey, Surface]
    scalars:   Mapping[ScalarKey, float]
    series:    Mapping[SeriesKey, TimeSeries]
    schedules: Mapping[ScheduleKey, Schedule]
```
**Keys are open** (currency · collateral · index · entity · underlying · pair) — a new asset adds
keys, never a field. Correlations are *scalars keyed by a pair*; dividends are *schedules keyed by an
underlying*. The category is now closed rather than a bet on four buckets.

## 3. `CurveSet` — one storage, typed accessors per family (RATIFIED — finding 2)

Discount/projection/survival are factor-shaped (a value in [0,1]); **an inflation curve returns an
index *level*** — different units, different interpolation (seasonality, monthly-flat vs
daily-linear). One untyped container would just relocate stringly-typing into the value type. So:

```python
class CurveSet:                                   # one backing store, keyed by CurveKey
    def discount(self, ccy, collateral=None) -> YieldCurve: ...
    def projection(self, index: RateIndex)   -> YieldCurve: ...
    def survival(self, entity)               -> SurvivalCurve: ...
    def inflation(self, index)               -> InflationCurve: ...
```
Uniform storage, type-safe access, and a level curve is never pretended to be a discount factor.
Single-curve stays the degenerate configuration (`projection(idx) → the discount curve`).

## 4. Ingestion — the adapter boundary
No universal market-data standard exists (FpML and the ISDA CDM are trade-centric; vendor feeds are
proprietary). Ports-and-adapters, living in the **data spine** — never imported by the core.

```python
class QuoteSource(Protocol):
    def quotes(self, as_of: date) -> QuoteSet: ...
```
The adapter's real work is **identity resolution**: *"EUR 3M swap 2.5%"* → look up our registries →
`ParSwapQuote(currency=EUR, tenor=Tenor(3, M), index=EURIBOR_3M, rate=0.025)`. It is the **only**
place allowed to know vendor naming.

Topic 1 uses **checked-in synthetic quote sets** on real conventions. ECB/FRED/BOE and vendor
adapters are the separate market-data topic; they implement the same protocol and change nothing
above it.

## 5. Provenance
```python
calibrate(quotes, spec) -> (CurveSet, CalibrationResult)
```
`CalibrationResult`: quotes used · method · residuals · iterations · convergence · **par→zero
Jacobian**. Curves stay clean (`df`/`zero_rate`/`forward_rate`); provenance sits **beside** them —
never a mutable attribute on the curve (the quarry's anti-pattern).

## 6. Resolution safety — the oracle blind spot (RATIFIED — finding 4)

> **`reprice-to-par` CANNOT detect mis-resolution.** Resolve *"EUR 3M swap"* to ESTR instead of
> EURIBOR 3M and the curve builds **self-consistently with the wrong assumption** — every pillar
> reprices to par perfectly, and the error propagates silently into every price.

Because the headline oracle is blind here, resolution needs its own guards:
1. **Fail loud on ambiguity.** The resolver never guesses a default index, currency or convention.
2. **Retain the raw external identifier** on every quote, so resolution is auditable after the fact.
3. **Resolution is asserted, not inferred** — a quote's resolved `RateIndex` is checked against the
   curve it is being used to build (a projection pillar must reference that curve's index).

## 7. Anchor consistency (finding 5)
A curve carries its own anchor date; the snapshot carries `valuation_date`. **Assert they agree** on
snapshot construction. Roll-down (same curve, later valuation) is an **explicit operation** producing
a new curve *and* a new snapshot together — never a silent divergence.

## 8. The adversarial pass — findings and resolutions
| # | finding | resolution |
|---|---|---|
| 1 | snapshot mixes built and passed-through data; a datum can be both calibration and pricing input, and bumping means different things | §1 dual-role rule: `QuoteSet` authoritative; par risk bumps quotes and rebuilds; zero/pillar risk bumps the built curve; never mixed |
| 2 | "one CurveSet for all curves" over-unifies — inflation is a *level* curve | §3 typed accessors per family over one store |
| 3 | four concept-fields is a bet that all market data fits four shapes; correlations and discrete dividends do not | §2 closed SHAPES × open KEYS |
| 4 | **reprice-to-par is blind to mis-resolution** — wrong index still reprices perfectly | §6 fail-loud resolver · raw identifier retained · resolution asserted against the target curve |
| 5 | curve anchor vs snapshot `valuation_date` can silently disagree | §7 assert on construction; roll-down is explicit |

## 9. The test this design must pass
Credit · equity · commodity · inflation · FX each slot in by **adding keys, never a field**:
- credit → `Survival(entity)` term structure + recovery **scalar**
- equity → spot **scalar** + vol **surface** + discrete-dividend **schedule**
- commodity → spot **scalar** + carry/convenience term structure + delivery **schedule**
- inflation → `Inflation(index)` term structure (typed accessor) + seasonality
- FX → spot **scalar** keyed by pair + `Discount(ccy, collateral)` for collateral discounting

If any of them forces a new field, the shape set is wrong — and we would rather learn that here than
three topics in.
