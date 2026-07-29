# Artifact #19 — Market Data: the cross-asset design (rev 3, post-foundation-audit)

**Status:** Ratified. Cross-asset — every future asset class consumes it, so it is settled *before*
the multicurve build. Supersedes A5's snapshot shape. **Rev 2** incorporated the adversarial pass
(§8). **Rev 3** folds in what the foundation audit changed (doc 21): FX quote order & triangulation
(§2.1, `AC-3.6b` promoted from deferral into F1 scope), the serialisation convention (§5.1, audit
3.4), and `TimeMeasure` as the sanctioned anchor (§7, ruling A1). The rev-3 amendments are recorded
in §10.

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
| **scalar** | a single number | equity/commodity spot · recovery · **correlation (keyed by a pair)** · **FX spot (directional — carries quote convention, §2.1)** |
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

## 2.1 FX spot is directional — quote order & triangulation (RATIFIED — rev 3, `AC-3.6b`)

A scalar keyed by `(EUR, USD)` does **not** say which quote convention the number is in. EURUSD 1.08
and USDEUR 0.926 are the same market; stored as a bare float they are indistinguishable, and an
**inverted spot builds an xccy curve that calibrates perfectly and prices everything wrong** — the §6
mis-resolution blind spot, in the one asset class where inversion is a coin-flip. FX did slot into §2
by "adding a key" and was *still* under-specified: §9's slot-in test is necessary, not sufficient.

> **An FX spot carries its quote convention (base/quote order). The snapshot never exposes a bare
> pair-scalar; it exposes a directional accessor that asserts, never guesses:**

```python
def fx_rate(self, base: Currency, quote: Currency) -> float: ...   # inverts internally; asserts
```

Backed by the **pair-conventions registry** promoted out of the ledger (`AC-3.6b`):
- **market-standard quote order** per pair (so a raw quote's direction is resolved, not assumed);
- **spot lag** (already moved off `CurrencyPair` identity in the audit, into this registry);
- **triangulation** for a cross, through a **declared** vehicle currency — a *stated* path, never an
  inferred one. §6.1 (fail loud on ambiguity) applies unchanged: an undeclared cross raises.

*Oracle:* `fx_rate(A,B) · fx_rate(B,A) == 1` to tolerance; a cross triangulated through the vehicle
equals the directly-quoted cross wherever both exist.

**Scope honesty (`AC-3.6b`, green-oracle gate).** `fx_spot_date` is a **joint-counting** algorithm,
not the full asymmetric ACI intermediate-day rule — the asymmetric rule is implemented *only* if a
primary source with a verifiable worked example can be cited, else the joint-count behaviour stands
and says so in its docstring. "FX spot FIXED" must not read as a complete ACI spot algorithm.

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

## 5.1 Serialisation — QuoteSet + spec reproduces the snapshot (RATIFIED — rev 3, audit 3.4)

The foundation audit (3.4) landed the convention: **identities serialise by name, atoms by value.** A
`MarketSnapshot` holds both — `RateIndex`/`Currency`/`Calendar` are identities (by name, rehydrated
through their registries); curves, spots and fixings are atoms (by value). `QuoteSet` and
`MarketSnapshot` serialise under this convention directly.

> **A serialised `QuoteSet` + a `CalibrationSpec` is sufficient to reproduce a `MarketSnapshot`
> exactly.** The built curves need not be serialised as primary state — they are *derived*, and §1
> makes the `QuoteSet` authoritative.

This is what makes engine invariant 1 (identical inputs ⇒ identical output) checkable **across
processes**, not only within one — the reproducibility contract extends to the market state, not just
the pricing call.

## 6. Resolution safety — the oracle blind spot (RATIFIED — finding 4)

> **`reprice-to-par` CANNOT detect mis-resolution.** Resolve *"EUR 3M swap"* to ESTR instead of
> EURIBOR 3M and the curve builds **self-consistently with the wrong assumption** — every pillar
> reprices to par perfectly, and the error propagates silently into every price.

Because the headline oracle is blind here, resolution needs its own guards:
1. **Fail loud on ambiguity.** The resolver never guesses a default index, currency or convention.
2. **Retain the raw external identifier** on every quote, so resolution is auditable after the fact.
3. **Resolution is asserted, not inferred** — a quote's resolved `RateIndex` is checked against the
   curve it is being used to build (a projection pillar must reference that curve's index).

## 7. Anchor consistency (finding 5; rev 3 — `TimeMeasure`, ruling A1)
A curve's anchor is a **`TimeMeasure(anchor, day_count)`** — the anchor date *and* its day count
together, the **only** sanctioned `date → t` map (ruling A1). An anchor date without its day count is
half a specification and lets two curves measure the same interval differently. The snapshot carries
`valuation_date`; **assert the `TimeMeasure` anchor agrees** with it on snapshot construction.
Roll-down (same curve, later valuation) is an **explicit operation** producing a new curve *and* a new
snapshot together — never a silent divergence.

Per A1, `TimeMeasure` lands as an **L0 module in Topic 1's first slice** (it has no consumer before
the curve layer); this doc records the dependency so F1 does not invent a second `date → t` mapping.

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
- FX → directional spot **scalar** (quote-convention-carrying, §2.1) + `Discount(ccy, collateral)`
  for collateral discounting

If any of them forces a new field, the shape set is wrong — and we would rather learn that here than
three topics in. **Caveat learned in rev 3:** FX *passed* this test (added a key, no new field) yet
was still under-specified on quote direction (§2.1). The slot-in test guards the *shape*; it does not
guarantee a key is fully specified.

## 10. Rev-3 amendments (foundation-audit reconciliation — doc 21)
| # | change | section | source |
|---|---|---|---|
| R1 | FX spot is directional — carries quote convention; snapshot exposes `fx_rate(base, quote)`; pair-conventions registry + declared triangulation | §2.1 (new) | audit `AC-3.6b`, promoted deferral → F1 scope |
| R2 | `QuoteSet` + `MarketSnapshot` serialise identities-by-name / atoms-by-value; QuoteSet+spec reproduces a snapshot exactly (cross-process invariant 1) | §5.1 (new) | audit 3.4 |
| R3 | the §7 anchor **is** a `TimeMeasure(anchor, day_count)` — the only sanctioned `date → t`; lands as an L0 module in T1's first slice | §7 | ruling A1 |
| R4 | §9 FX row corrected off the bare-scalar shape; slot-in caveat recorded | §9 | consequence of R1 |
