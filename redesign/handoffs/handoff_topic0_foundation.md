# HAND-OFF — Topic 0: the Foundation  (current; start here)

**This is the active hand-off.** It supersedes `handoff_topic1_conventions.md` (multicurve-scoped
foundation — wrong) and the ng-parking scope in `rulings_ng_parking_and_topic1_design.md` §1
(we now park **all** of ng, not 30 modules).

## Read first
| doc | why |
|---|---|
| `CLAUDE.md` (root) | the guardrails — law, not suggestion |
| `redesign/16_topic0_foundation.md` | **the spec for this topic** — read in full |
| `redesign/13_topic_migration_and_parking.md` | §3 tick mechanism · §5 target/use/**apply the design policy** |
| `redesign/15_foundation_comparison.md` | where each quarry `core/` module actually belongs |
| `redesign/12_domain_build_order.md` | why block order; drawdown is reporting, never steering |
| `redesign/11_checkpoint_and_review_cadence.md` | when to stop; the five review inputs |
| `redesign/02_spine.md` | Amendments A1–A6 |

## Stance
**Nothing above L0 is built until Topic 0's gate is green.** The foundation is designed
**cross-asset complete** and populated incrementally — it must accommodate credit's IMM rolls,
commodities' `Quantity` and options' full `NumericalConfig` now, even though only rates gets
populated first. Retrofitting L0 is the most expensive change in the system.

**Standing rule:** *mine the quarry and `ng_parked` for CONTENT, never for STRUCTURE.* Both are
content sources; neither's organisation carries authority. The per-file transformation gate is
`13 §5.3` — applied at write time, not review time.

---

## Slice 0 — `slice/ng-parking`
```
git mv ALL of src/pricebook_ng/ -> ng_parked/ , tests included.
Write ng_parked/MANIFEST.md: module | topic it belongs to | its re-base oracle | parked-at version.
Rationale: the existing ng foundation was built without RollRule, without calendars-keyed-by-identity,
without dual adjusted/unadjusted schedules. Editing it forward would inherit those decisions —
the same error we forbid with the quarry. Rebuild clean; ng_parked is a CONTENT source only.
Drop "quarry regression" from CI (it guards code we are replacing).
Version MINOR. No behaviour claim.
```

## Slice 1 — `slice/calendars`
Holiday-rule DSL (`fixed`/`easter`/`orthodox`/`nth`/`monday`, `since`/`until`), weekend rule, the
**three observance regimes** (US 5 U.S.C. §6103 · Commonwealth · Johannesburg Sunday-only),
`JointCalendar`, Christmas/Boxing collision cascade, Tokyo *furikae*, Colombian Emiliani law.
**Keyed by IDENTITY** (`TARGET`, `LONDON`, `NEW_YORK_SIFMA`) — currency→calendar is a *lookup* (C1).
**Declare all 37 markets.** *Mine:* `core/calendar.py` (943 LOC).
**Oracles:** known holiday dates per market; observance shifts (Sat vs Sun, US vs UK divergence);
Juneteenth `since=2021`; Danish Store Bededag `until=2023`; joint-calendar union.

## Slice 2 — `slice/daycounts`
**10 conventions** — the quarry's 7 (ACT/360 · ACT/365F · **30U/360 US SIA with the Feb rules** ·
30E/360 · ACT/ACT ISDA · ACT/ACT ICMA · BUS/252) **plus the gaps: ACT/365L · 30E/360 ISDA · NL/365**.
ICMA anchors via **`CouponPeriod`** (never 3 loose args). **Calendar passed in** — BUS/252 with no
calendar **raises** (the quarry silently defaults to São Paulo). **Delete `strict_icma`** — its
non-strict path priced a UST coupon at 1.9836 instead of 2.0000.
**Oracles:** ISDA 2006 §4.16 worked examples · ICMA Rule 251 · **UST coupon = exactly 2.0000**
(regression) · 30U/360 February edge cases.

## Slice 3 — `slice/schedules`
`Frequency` · `StubType` · **`RollRule`**(calendar, convention, eom) · **`ScheduleTerms` → `Schedule`**.
**Returns BOTH adjusted and unadjusted dates** (C2 — accrual uses unadjusted, payment adjusted).
**EOM anchored ONCE from `start`** (ISDA §4.10). **`BusinessDayConvention` += `NEAREST` and the EOM
variant.** **NEW: IMM roll dates** (3rd Wednesday) **and CDS roll dates** (20th Mar/Jun/Sep/Dec) —
neither exists in the quarry; credit and futures both need them.
**Shed:** the long-stub merge heuristic (`first_gap < months*30*0.5`) — replace with an explicit stub spec.
**Oracles:** ISDA schedule examples · EOM anchoring (31-Jan monthly → month-ends) · all four stubs ·
adjusted ≠ unadjusted under a holiday · published IMM and CDS roll tables.

> ### ►► CHECKPOINT (cadence ≤6) — stop, report, wait for rulings

## Slice 4 — `slice/money-quantity`
`Currency` · **`CurrencyPair`** (+ spot settlement lag, quote convention) · **`Money`**(amount,
currency — mixing is a type error) · **`Quantity`(amount, unit)** *(new: commodities settle in
barrels/MWh/tonnes)* · **`Cashflow`**(date, Money) · **`Leg`** · **`Accrual`**(start, end, day_count).
**Oracles:** currency-mixing rejected at type level; accrual year-fractions against slice 2;
unit arithmetic closed under same-unit addition only.

## Slice 5 — `slice/index-identity`
A **declarative index/underlying identity**, of which `RateIndex` is the first instance (equity,
inflation, commodity underlyings follow the same shape later).
**Rule: a new index is a DECLARATION, never a code change.** `RateIndex` carries currency · tenor ·
day_count · `RollRule` · **full RFR set**: `fixing_lag` · `observation_shift` · `lookback` ·
`lockout` · `payment_delay` · `compounding`. **`FixingHistory` generic over index** (not rates-shaped
— Asian options and inflation need fixings). **Exercise/expiry conventions** (expiry→settlement lag).
**ONE generic** `accrued_rate(index, accrual, fixings)`; the only branching is on `CompoundingMethod`.
**Fix, do not inherit:** the quarry's `_REGISTRY` is rebound at import by a JSON load that *replaces
the whole dict* — one bad entry drops the other 27. Explicit construction, no import-time I/O.
**Oracles:** compounded RFR against a hand-computed fixing series · **lookback vs observation-shift
must give DIFFERENT rates** · SONIA 0/0 vs SOFR 2/2 on the same period.

## Slice 6 — `slice/numerics-config`
`Interpolator` (mechanism only — *curve* extrapolation policy is L1, C4) · root-finding &
optimisation · distributions. **`NumericalConfig` with the FULL knob set**: MC (paths/seed/
antithetic/sobol/bridge) · PDE (time/space steps, n-std-devs) · tree steps · quadrature (tol,
max-iter) · COS (n, L) · root-finder (tol, max-iter) — `# fields-exempt: config aggregate`.
*This is the retrofit we are pre-empting:* the 12 knobs deferred at retire #1 are
`deferred→_fourier/_pde/_trees/_integrate/_rootfinding`. **`PricingResult`** (decomposition) ·
**`PricingFailure`** · the **serialisation pattern** (per-class `to_dict`/`from_dict` +
`schema_version`; **no framework** — the quarry's 831 lines are not carried).
**Demand-migrated, NOT now:** MC engine · PDE · Fourier/COS · trees · quadrature.

> ### ►► TOPIC 0 GATE — then park the set

## Topic 0 close
Mark `covered` as each is superseded, then `git mv` the set to `parked/topic-00-foundation/`:
```
calendar · day_count · schedule · rate_index · currency · fixings
interpolation · solvers · data_registry · notional · serialisable · serialization
```
(`core/numerical_config` and `fixed_income/fixed_leg` are already retired.)
Refresh `quarry_reconciliation.md`. **Then, and only then, Topic 1 — yield curves.**

**Note:** Topic 0 is *part migration, part new construction* — `Money`, `Quantity`, `RollRule`,
`Accrual`, `CouponPeriod`, `ScheduleTerms`, IMM/CDS rolls, `PricingResult`/`PricingFailure` have **no
quarry counterpart**. Do not expect a 1:1 file mapping.
