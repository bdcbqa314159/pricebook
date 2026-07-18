# Cowork AUDIT — Topic 0 foundation (v0.63.0)

**Single actionable document for the Topic 0 gate.** Supersedes the loose rulings files for gate
purposes (`rulings_topic0_s3.md`, `rulings_topic0_settlement_and_index.md`, `rulings_topic0_gate.md`
remain as the reasoning record).

14 modules · 1753 LOC · 4 blockers found. **Gate is NOT green.**

---

## PASS — verified module by module

| check | result |
|---|---|
| function args ≤5 | ✅ **zero** violations across the tree |
| provenance headers | ✅ all 13 modules |
| ANZAC per-rule override | ✅ `fixed(..., observed=False)`, exception documented in the docstring |
| ACT/365L | ✅ frequency-keyed via `CouponPeriod` (ISDA §4.16(i) form) |
| EM calendar honesty | ✅ `Coverage` marker present (secular-only flagged, not silent) |
| numeraire basis | ✅ `DiscountBasis` on `PricingResult` — a PV states what it is discounted on |
| settlement | ✅ `SettlementType` · `Delivery(Quantity)` · `SettlementTerms` · `settlement_date` |
| index widening | ✅ `ObservationStyle` + `spread_adjustment` present |
| L0 finance-free | ✅ no pricing leakage (the PV references in `results.py` are the engine-I/O value type naming its own contents — legitimate) |

The conventions work is genuinely good: tight, published-oracled, no debt.

---

## FAIL — 4 findings, all must clear before the gate

### F1 (blocker) — `NumericalConfig`: 16 fields, still `fields-exempt`
The decomposition ruling has not been applied. Required shape:
```python
MonteCarloConfig(paths, seed, antithetic, sobol, brownian_bridge)   # 5
LatticeConfig(time_steps, space_steps, n_std_devs, tree_steps)      # 4
IntegrationConfig(quad_tol, quad_max_iter, cos_n, cos_L)            # 4
SolverConfig(root_tol, root_max_iter, fd_bump)                      # 3
NumericalConfig(monte_carlo, lattice, integration, solver)          # 4 ✓
```
**Remove the marker.** `verify.py fields` must pass on merit.

### F2 (blocker, most serious) — `RateIndex` LOST its calendar; reverted to the quarry's flaw
`accrued_rate` does:
```python
calendar = calendar_for_currency(index.currency.value)
```
This is **exactly the quarry defect that #14/#16 ratified as fixed**: *"ng improvement — `RateIndex`
carries a `RollRule` (hence a calendar); the quarry has no calendar field and infers it from
currency."* The build inferred from currency.

Concretely wrong today: **SOFR fixes on the SIFMA/US-government-securities calendar, not generic USD
settlement.** Also unexpressible: two USD indices on different calendars; any index on a joint
calendar. This is a regression against a ratified decision, and it silently produces wrong business
days — the failure class the project exists to prevent.

**Fix:** `RateIndex` carries its own `RollRule`; `accrued_rate` reads `index.accrual.roll.calendar`.
No currency inference anywhere.

### F3 (blocker) — `RateIndex`: 12 fields + a self-granted `fields-exempt`
`# fields-exempt: index-identity aggregate` is precisely the pattern ruled against in §3b —
exemptions are for **irreducible output records**, never for convention/config records. It
decomposes cleanly, and adding the F2 calendar makes it 13 anyway. Required shape:

```python
IndexId(name, currency, tenor)                                      # 3
AccrualConvention(day_count, roll)                                  # 2  ← RollRule fixes F2
FixingRule(observation_style, compounding, fixing_lag)              # 3
RfrConvention(observation_shift, lookback, lockout, payment_delay)  # 4
RateIndex(id, accrual, fixing, rfr, spread_adjustment)              # 5 ✓  no marker
```
It reads better, which is the point:
```python
SOFR = RateIndex(
    IndexId("SOFR", USD, "ON"),
    AccrualConvention(ACT_360, RollRule(SIFMA, MODIFIED_FOLLOWING, eom=False)),
    FixingRule(BACKWARD_LOOKING, COMPOUNDED, fixing_lag=0),
    RfrConvention(observation_shift=2, lookback=0, lockout=0, payment_delay=2),
)
```
An IBOR declares `RfrConvention.none()` — the distinction becomes visible rather than encoded in
five zeros.

### F4 — no general index/underlying identity; sibling types absent
`RateIndex` stands alone. But **§3c** rules that a new asset adds *"an identity, as a sibling under
the general index/underlying concept"* — and there is nothing to be a sibling of. **Credit is next
and `ReferenceEntity` has no home.** The S5 ruling also required inflation / FX-fixing /
equity-commodity index types *defined* (populated later); not done.

**Fix (minimal — this does not need to be elaborate):** an `Underlying` protocol carrying `name` and
`asset_class`, which `RateIndex` satisfies and which `MarketKey` (A5) keys on. Then declare the
sibling types as thin records: `InflationIndex` (indexation lag · interpolation · base),
`FxFixing` (source · time), `EquityUnderlying`/`CommodityUnderlying` (observation). Populated later;
**defined now**, so credit and the rest have a shape to slot into.

---

---

# SOLIDITY REVIEW — what will force us back into L0

The goal is an L0 retouched *rarely*. Stress-tested against credit / equity / commodity / inflation /
FX / LatAm, six things **will** force a change. Two are inside the already-ratified scope.

### S1 (certain, soon) — `Currency` is a closed enum missing **BRL**
19 currencies; the calendars cover **37**. Unnameable today: **BRL, MXN, CLP, COP, PEN, ARS**, ZAR,
CNY, KRW, INR, SGD, HKD, IDR, MYR, THB, PHP. The scope contract says *"EUR/USD/GBP **and
LatAm/BRL**"* — so an explicitly in-scope currency **cannot be expressed**.
**Fix:** make `Currency` **open** — an ISO-4217 code + a registry carrying its conventions (minor
units, settlement lag). A new market becomes a *declaration*, never an L0 edit — the same rule we
already apply to indices.

### S2 (certain) — `Leg` cannot hold a physical `Delivery`
`Leg.cashflows: tuple[Cashflow, ...]`. `Delivery(date, Quantity)` exists in `settlement.py` but no
leg can contain one, so a commodity swap leg or a physically-settled equity leg is inexpressible.
**Fix:** `Leg` holds `flows: tuple[Flow, ...]` where `Flow = Cashflow | Delivery` (a small union or
protocol). Do it now — retrofitting `Leg` later touches every product built on it.

### S3 (certain) — `Frequency` cannot express 28-day, daily, or single-period
Members are `WEEKLY=0 · MONTHLY=1 · QUARTERLY=3 · SEMI_ANNUAL=6 · ANNUAL=12`, encoded as **month
counts**. That encoding cannot represent:
- **28-day** — Mexico's TIIE benchmark. **LatAm is in scope.**
- **daily** — overnight/compounding legs.
- **single-period / bullet** — zero-coupon swaps, ZCIS, discount instruments (there is no "once, at
  maturity").
`WEEKLY=0` already breaks the months encoding — the smell is visible.
**Fix:** represent the period as a **`Tenor`-like step** (count + unit: D/W/M/Y) rather than a month
integer, plus an explicit `BULLET`/single-period case.

### S4 (likely) — `Unit` is a closed enum
8 commodity units. Missing: carbon allowances (EUA), freight, lots/contracts, agricultural variants.
**Fix:** same as S1 — open registry keyed by symbol.

### S5 (likely, and structurally awkward) — `Calendar` has no early closes, and is at the 5-field limit
Equity and bond markets have **half-days** (Christmas Eve, day after Thanksgiving) — not holidays,
not full business days; they affect fixing cut-offs and settlement. Zero support today. And
`Calendar` already carries **5 fields**, so adding it later forces a *restructure*, not just a field.
**Fix:** decide now — either bundle the day-classification into one field
(`day_type(d) -> BUSINESS | HALF | HOLIDAY | WEEKEND`) or accept it explicitly as deferred with the
`Coverage` marker extended to say "no half-days".

### S6 (planned churn) — `PricingResult` decomposition vocabulary is undecided
Three fields (`pv`, `accrued`, `basis`). A4.4 ruled "grows on demand" — which means **every asset
class retouches L0**: credit wants upfront, options want greeks, commodity wants physical settlement.
**Fix:** apply *design complete, populate incrementally* here too — fix the **full decomposition
vocabulary now** (pv · accrued · clean · cashflow breakdown · sensitivities · diagnostics), fields
optional and unpopulated until a consumer arrives. Decide the shape once.

### S7 (certain — implied by S3) — `Tenor` must become a value type
We ruled "`Tenor` stays a string + helpers." **S3 overturns that.** If `Frequency` becomes a
tenor-step (count + unit) to express 28-day and daily, then `Tenor(count, unit)` *is* that primitive
— and it is already needed in three places: index tenors (`"3M"`), schedule steps, and curve pillar
tenors. Keeping it a string means parsing `"28D"` in three modules.
**Fix:** `Tenor(count: int, unit: D|W|M|Y)` with `parse("3M")`/`__str__`. S3 and S7 are one change.

### S8 (likely) — `ScheduleTerms` has no roll-day anchor
Schedules roll on a day-of-month by convention — a bond paying the 15th, CDS the 20th, IMM the 3rd
Wednesday. Today it is implied by `start`, which fails when the effective date differs from the roll
anchor (very common: trade date ≠ first accrual ≠ roll day).
**Fix:** an explicit optional `roll_day` on `ScheduleTerms` (day-of-month, or an IMM/CDS rule).

### S9 (decide now, probably NOT a Calendar change) — intraday time and cut-offs
FX fixes at **4pm London**; FX options cut at **10am NY**; commodity settlement has fixed times.
`Calendar` is date-only, and it is already at the 5-field limit.
**Ruling to record:** *fixing time is **index metadata*** (goes on the `FxFixing` sibling), *expiry cut
is **product data*** (L2). **Time-of-day does NOT enter `Calendar`.** Record it explicitly, or someone
adds a `time` field to `Calendar` in six months.

### S10 (do with F4) — sketch the sibling `Underlying` fields now
F4's protocol will be under-designed unless we know what has to fit it:
| sibling | fields it needs |
|---|---|
| `ReferenceEntity` (credit) | name · (seniority/restructuring are **L2**, not here) |
| `InflationIndex` | indexation lag · interpolation (daily-linear vs monthly-flat) · base index |
| `FxFixing` | pair · source (WM/R) · **fixing time** (S9) |
| `EquityUnderlying` | exchange · currency |
| `CommodityUnderlying` | unit · **delivery location** · **grade** (Brent≠WTI; gas delivery point) |
Delivery location and grade are the ones that would otherwise be discovered late.

### S11 (cheap now) — verify `DayCountConvention` completeness against ISDA 2006
We have 10. Check for `1/1` and `ACT/ACT AFB`. An enum edit later is an L0 edit; adding two members
now costs nothing.

---

### S12 (CERTAIN — the biggest remaining gap; absent from BOTH trees) — no rate-quotation basis
There is **no concept of how a rate is compounded for quotation** — `SIMPLE / ANNUAL / SEMI_ANNUAL /
QUARTERLY / CONTINUOUS` — and no conversion between them. `CompoundingMethod` is a *different* concept
(index averaging: `COMPOUNDED/EXPONENTIAL/AVERAGED/FLAT`), which makes the absence easy to miss and
the names collide.

**A rate is meaningless without its basis (compounding + day count).** The quarry papered over this by
hardcoding `zero_rate` to continuous on an internal ACT/365F axis; our `YieldCurve.zero_rate() -> float`
inherits exactly the same ambiguity. It bites **immediately in Topic 1**:
- market swap quotes are quoted with a compounding convention (USD fixed 30/360 semi-annual);
- bond yields are semi-annual bond-equivalent;
- curve internals are continuous;
- par, zero and forward rates are all "rates" on **different bases**.

Treating a semi-annual rate as continuous is wrong by ≈ r²/2 — numerically valid, silently incorrect:
precisely the failure class this project exists to prevent.

**Fix (L0 — it is a convention plus pure math):**
- `Compounding(SIMPLE | ANNUAL | SEMI_ANNUAL | QUARTERLY | MONTHLY | CONTINUOUS)` — a *finite standard
  set*, so an enum per the meta-rule. **Rename the index one** to `AccrualMethod` (or similar) to kill
  the collision.
- `convert_rate(rate, t, from_basis, to_basis) -> float`.
- **Record the invariant:** internal curve rates are **continuously compounded on the curve's day
  count**; *quotes carry their own basis* and convert at the boundary. That keeps `Rate` a plain float
  (as ruled) without the ambiguity.

### S13 (decide now — it changes `Leg`, which is already being reshaped) — pay/receive direction
Is direction carried by the **sign of `Money`**, or by an explicit `pay_receive` on `Leg`? Undecided.
A swap has a paying and a receiving leg; every product will need it. Since S2 already reshapes `Leg`,
decide in the same change.
**Recommend:** signed amounts (a payment is negative), *no* direction field — one representation,
no possibility of sign-and-flag disagreeing.

### S14 (cheap) — degenerate periods
`year_fraction` with `end < start`, and zero-length accruals: raise, or return negative? Forward-starting
and seasoned trades will hit it. **Recommend: raise** (an accrual is ordered by construction), and
record it.

### S15 (cheap, and required by artifact #10) — pin the RNG
Cross-platform reproducibility (#10) forces tolerance-based oracles for libm; **MC additionally needs a
pinned generator**. `MonteCarloConfig` carries a `seed` but not the RNG *family* — so a later switch
silently shifts every MC oracle. **Pin it now** (name the generator in `MonteCarloConfig`, or as a
documented invariant).

### S16 (record only) — business-day counting convention
`business_days_between` is start-exclusive / end-inclusive (inherited). Some markets differ.
**Record the invariant** rather than parameterise it — no present consumer needs the alternative.

---

## The meta-rule (why these keep recurring)

Every one of S1, S4, S11 is the same mistake in a different place: **a closed enum modelling an
open-ended domain.** So state the rule and stop rediscovering it:

> **Open-ended domains get a registry** (a new member is a *declaration*): currencies, units,
> underlyings/indices, calendars, markets.
> **Standardised finite domains keep an enum** — but are **completed now**: day counts,
> business-day conventions, compounding methods, stub types, settlement types, interpolation schemes.

Applying it mechanically is what makes L0 one-off: the registry categories never need an L0 edit
again, and the enum categories are closed by construction.

*Lower priority, noted:* `Money` has no rounding/precision convention (JPY 0dp vs 2dp) — fold
`minor_units` into the S1 currency registry while it is being built; it bites at settlement/booking.

---

## Gate checklist
**Correctness blockers (F):**
```
[ ] F1  NumericalConfig decomposed, marker removed
[ ] F2  RateIndex carries RollRule; accrued_rate uses index calendar; no currency inference
[ ] F3  RateIndex decomposed (IndexId/AccrualConvention/FixingRule/RfrConvention), marker removed
[ ] F4  Underlying protocol + sibling index types defined
```
**Solidity blockers (S) — so L0 is rarely retouched:**
```
[ ] S1  Currency OPEN (ISO code + registry, incl. minor_units). BRL nameable — it is in scope.
[ ] S2  Leg holds flows: tuple[Flow, ...]  where Flow = Cashflow | Delivery
[ ] S3  Frequency expresses 28-day, daily and single-period (tenor-step, not a month int)
[ ] S4  Unit OPEN (registry keyed by symbol)
[ ] S5  Calendar half-days: either day_type(d) classification, or explicitly deferred in Coverage
[ ] S6  PricingResult full decomposition vocabulary fixed now (fields optional, unpopulated)
[ ] S7  Tenor becomes a value type (count + unit) — same change as S3
[ ] S8  ScheduleTerms gains an explicit roll-day anchor
[ ] S9  RECORD the ruling: time-of-day is index/product metadata, never Calendar
[ ] S10 Sketch sibling Underlying fields (esp. commodity delivery location + grade) with F4
[ ] S11 DayCountConvention completeness vs ISDA 2006 (check 1/1, ACT/ACT AFB)
[ ] S12 Compounding(SIMPLE..CONTINUOUS) + convert_rate(); rename the index CompoundingMethod
        to AccrualMethod; record "internal rates are continuous on the curve day count"
[ ] S13 pay/receive = signed amounts, no direction field (decide with S2's Leg reshape)
[ ] S14 degenerate periods raise (end < start, zero-length)
[ ] S15 pin the RNG family in MonteCarloConfig (cross-platform reproducibility, artifact #10)
[ ] S16 record the business-day counting invariant (start-exclusive, end-inclusive)
```
**Apply the meta-rule mechanically:** open-ended domains → registry; standardised finite domains →
enum, completed now. That is what makes L0 one-off.
**Gate:**
```
[ ] verify.py fields green ON MERIT (zero fields-exempt markers in foundation/)
[ ] oracle: SOFR on the SIFMA calendar differs from a generic-USD-calendar index over a period
        containing a US-government-securities-only holiday   ← the F2 regression test
[ ] oracle: a BRL instrument is constructible end-to-end (Currency · BUS/252 · São Paulo calendar)
        ← the S1 regression test; proves the in-scope LatAm claim is real
[ ] then: park the set → parked/topic-00-foundation/ , refresh roll-up, report
```
Topic 1 does not begin until this is green.

**Why S1–S6 belong at the gate rather than "later":** each one is a *shape* decision, not a feature.
Adding BRL later is an L0 edit; making `Currency` open is a one-time change that ends the category.
Same for `Leg`, `Frequency`, `Unit`. This is the *design complete, populate incrementally* rule —
the exact principle that produced Topic 0 in the first place.
