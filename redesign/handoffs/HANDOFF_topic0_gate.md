# HAND-OFF — Topic 0 gate (the active document)

**Give Claude Code this file + `CLAUDE.md`. Nothing else is needed to act.**
Reasoning behind each item: `AUDIT_topic0_foundation.md` (F1–F4, S1–S17) and
`rulings_topic0_gate_spotcheck.md`. Those are the record; **this is the instruction**.

**Tags:** `[BLOCK]` = the multicurve/linear-rates world (Topic 1) cannot be built correctly without
it. `[solid]` = one-off solidity; Topic 1 does not depend on it, but we land it now so L0 is
retouched rarely.

---

## Slice 1 — `slice/l0-tenor-frequency-roll`
- **[BLOCK] S7** `Tenor(count, unit: D|W|M|Y)` value type + `parse("3M")`/`__str__` +
  `date + Tenor` with roll adjustment. *Curve pillars **are** tenors; this is the most-used operation
  in curve building.* Overturns the earlier "Tenor stays a string" ruling.
- **[BLOCK] S3** `Frequency` expressed as a **tenor-step**, not a month integer. Must cover
  **single-period / BULLET** (a deposit and an FRA are one-period; a zero-coupon has no schedule),
  plus **28-day** (MXN TIIE) and **daily**. `WEEKLY=0` already breaks the month encoding.
- **[BLOCK] S8** explicit **roll-day anchor** on `ScheduleTerms` (day-of-month, or IMM/CDS rule).
  Today it is implied by `start`, which breaks whenever trade date ≠ effective date ≠ roll day — the
  normal case for IMM-dated FRAs and futures.

**Oracles:** `date + 3M` under each roll convention vs published examples · a BULLET schedule has one
period · an IMM-anchored schedule lands on 3rd Wednesdays regardless of effective date.

## Slice 2 — `slice/l0-rate-basis`
- **[BLOCK] S12** `Compounding(SIMPLE | ANNUAL | SEMI_ANNUAL | QUARTERLY | MONTHLY | CONTINUOUS)`
  + `convert_rate(rate, t, from_basis, to_basis)`. **Rename the index `CompoundingMethod` →
  `AccrualMethod`** (it models index averaging — a different concept; the collision is why this gap
  hid). **Record the invariant:** *internal curve rates are continuously compounded on the curve's
  day count; quotes carry their own basis and convert at the boundary.* `Rate` stays a float.

*Why blocking:* par swap quotes are semi-annual 30/360, zero rates are continuous, forwards are
simple. **Every bootstrap converts between them.** Missing from both trees; treating a semi-annual
rate as continuous is wrong by ≈r²/2 — valid-looking and silently incorrect.

**Oracles:** round-trip `convert_rate` across every pair · a semi-annual 5% converts to the published
continuous equivalent · annual↔continuous matches `ln(1+r)`.

## Slice 3 — `slice/l0-index`
- **[BLOCK] F2** `RateIndex` **carries its own `RollRule`** (hence calendar). Delete
  `calendar_for_currency(index.currency)` from `accrued_rate`. **SOFR fixes on SIFMA**, not generic
  USD settlement — currency inference is the quarry defect we ratified as fixed.
- **[BLOCK] F3** decompose (removes the self-granted `fields-exempt`):
  `IndexId(name, currency, tenor)` · `AccrualConvention(day_count, roll)` ·
  `FixingRule(observation_style, compounding, fixing_lag)` ·
  `RfrConvention(observation_shift, lookback, lockout, payment_delay)` →
  `RateIndex(id, accrual, fixing, rfr, spread_adjustment)` = **5**.
- **[solid] F4 + S10** `Underlying` protocol (`name`, `asset_class`) that `RateIndex` satisfies and
  `MarketKey` keys on; sibling types **defined, not populated**: `InflationIndex`(lag · interpolation ·
  base) · `FxFixing`(pair · source · time) · `EquityUnderlying`(exchange · ccy) ·
  `CommodityUnderlying`(unit · **delivery location** · **grade**).

**Oracle [BLOCK]:** SOFR on the SIFMA calendar ≠ a generic-USD-calendar index over a period
containing a government-securities-only holiday. *If those are equal, the calendar is not being read
from the index.*

## Slice 4 — `slice/l0-flows`
- **[BLOCK] S13** pay/receive = **signed amounts, no direction field** (one representation; sign and
  flag can never disagree). A swap is two legs in opposite directions.
- **[solid] S2** `Leg` holds `flows: tuple[Flow, ...]`, `Flow = Cashflow | Delivery`.
- **[solid] S14** degenerate periods (`end < start`, zero-length) **raise**.

## Slice 5 — `slice/l0-numerics`  (S17 — scipy is ratified)
- **[BLOCK]** `foundation/interpolation.py`: `LINEAR`/`LOG_LINEAR` ours; `CUBIC_SPLINE`,
  `MONOTONE_CUBIC`(PCHIP), `AKIMA` as **scipy adapters**; **extrapolation policy stated per end**
  (`FLAT | CONTINUE_SLOPE | RAISE`) — closes C4's silent divergence.
- **[BLOCK]** `foundation/solvers.py`: Brent · Newton · secant · least-squares (LM) via scipy,
  **replacing** `bisect_root`/`nelder_mead`. `distributions.py` → `scipy.stats.norm`.
  **No duplicates.**
- **[BLOCK] F1** decompose `NumericalConfig`: `MonteCarloConfig`(5) · `LatticeConfig`(4) ·
  `IntegrationConfig`(4) · `SolverConfig`(3) → `NumericalConfig(monte_carlo, lattice, integration,
  solver)` = 4. **Remove the marker.**
- **[solid] S15** pin the RNG family in `MonteCarloConfig`.
- **Thin adapters only** — never call scipy from engines/models; this is the C++ swap point.
- **Hagan–West stays in Topic 1** (it interpolates from *interval averages* to build forward curves;
  the quarry keeps it in `forward_interpolation.py`, which is curve construction).

> ### ►► CHECKPOINT (cadence ≤6) — stop, report, wait

## Slice 6 — `slice/l0-open-domains`  [solid]
- **S1** `Currency` **open**: ISO-4217 code + registry (incl. `minor_units`). **BRL must be
  nameable** — it is in the ratified scope contract.
- **S4** `Unit` open registry keyed by symbol.
- **S11** `DayCountConvention` completeness vs ISDA 2006 (check `1/1`, `ACT/ACT AFB`).
- **Meta-rule:** open-ended domains → **registry** (a new member is a *declaration*); standardised
  finite sets → **enum, completed now**.

**Oracle:** a BRL instrument constructible end-to-end (Currency · BUS/252 · São Paulo calendar).

## Slice 7 — `slice/l0-results-and-invariants`  [solid]
- **S6** `PricingResult` full decomposition vocabulary (already largely present — confirm complete;
  its `fields-exempt` is **legitimate**: an output record of independent facets, not a config).
- **S5** `Calendar` half-days: either a `day_type(d) -> BUSINESS|HALF|HOLIDAY|WEEKEND`
  classification, **or** explicitly deferred via the `Coverage` marker. `Calendar` is at its 5-field
  limit, so decide deliberately.
- **S9** record: **time-of-day never enters `Calendar`** — fixing time is index metadata, expiry cut
  is product data.
- **S16** record: business-day counting is start-exclusive / end-inclusive.

## Slice 8 — `slice/l0-serialisation`
Demonstrate the per-class `to_dict`/`from_dict` + `schema_version` pattern on a **hard case** —
nested value object **and** enum **and** collection (`Schedule`'s date tuples or `FixingHistory`'s
nested mappings). One easy class is not a demonstrated pattern.

---

## Re-classifications (no code)
- `core/data_registry.py` → **`dead`**, not reassigned. Its purpose is import-time JSON registry
  loading; we **ruled that capability away** (S5: explicit construction, no import-time I/O). Nobody
  will build it.
- `core/notional.py` → value concept **absorbed** into `Leg`/`Money`; the scalar→list expansion is
  **L2** product convenience.
- `core/fixings.py` → **reassign→market-data/persistence topic** (mutable store + file I/O ≠ L0).
  ng's immutable `FixingHistory` read model is the correct L0 type. *Your split was right; the
  hand-off that listed it in Topic 0 was wrong.*

## Topic 0 gate
```
[ ] all [BLOCK] items landed; [solid] items landed
[ ] verify.py fields green ON MERIT — zero fields-exempt in foundation/ except PricingResult
[ ] verify.py layers green; acyclic · debt · version · provenance green; ruff green
[ ] the two regression oracles: SOFR-on-SIFMA ≠ generic-USD ; BRL constructible end-to-end
[ ] park the Topic 0 set → parked/topic-00-foundation/  (~14 modules: calendar · day_count ·
    schedule · rate_index · currency · fixings* · interpolation · solvers · data_registry ·
    notional · serialisable · serialization)   *fixings reassigned, see above
[ ] refresh quarry_reconciliation.md roll-up; report per redesign/11
```
**Then Topic 1 — multicurve + linear rates.** L0 covers its base once the `[BLOCK]` items land.
