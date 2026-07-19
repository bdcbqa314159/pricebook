# Topic 0 — Foundation (L0, cross-asset complete) · MANIFEST

Living tracker (#13 §3.3) for the **foundation** topic. Status: `target` → `covered` | `dead` |
`reassigned→X` → `parked`. Fan-in = production consumers in `python/pricebook/` (excl. own tests) —
**irrelevant to parkability** (#13 §2: ng never imports the quarry; the quarry's tests park with it).

Topic 0 is **part migration, part new construction** — `Money`, `Quantity`, `RollRule`, `Accrual`,
`CouponPeriod`, `ScheduleTerms`, IMM/CDS rolls, `Delivery`, `PricingResult`/`PricingFailure` have **no
quarry counterpart** (handoff §note). So the covered rows below are the migrated fraction; the rest of
Topic 0 is new L0 that stands on nothing to retire.

Spec: `redesign/16_topic0_foundation.md`. Canonical gate handoff: `HANDOFF_topic0_gate.md` (8 slices).
Gate audit (F1–F4 + S1–S17, all landed): `AUDIT_topic0_foundation.md`; spotcheck
`rulings_topic0_gate_spotcheck.md` (the `data_registry`=dead / `notional`=absorbed / `fixings`→market-data
re-classifications).

---

## THE SET (handoff `HANDOFF_topic0_gate.md` §"Topic 0 gate")

| quarry file | LOC | fan-in | domain role | status | covered-by (ng) | evidence / shed | slice |
|---|---|---|---|---|---|---|---|
| `core/day_count.py` | 274 | 228 | 7 day-count conventions + `year_fraction` | **covered** | `foundation/day_count.py` | ng is a **superset** (10 conventions: 7 + ACT/365L · 30E/360-ISDA · NL/365); calendar passed in (BUS/252 raises, no São Paulo default); ICMA via `CouponPeriod`; degenerate periods raise (S14). **shed:** `date_from_year_fraction` `deferred→topic-01` (t→date curve helper, no ng consumer). | daycounts |
| `core/calendar.py` | 943 | 54 | 37 calendars + holiday-rule DSL | **covered** | `foundation/calendar.py` + `market_calendars.py` | all 37 declared; DSL (`fixed`/`easter`/`orthodox`/`nth`/`monday`, `since`/`until`); `JointCalendar`; three `Observance` regimes incl. Japanese *furikae* + Colombian Ley Emiliani; `DayType` + half-days (S5). Identity-keyed; currency→calendar a lookup (C1). | calendars |
| `core/schedule.py` | 143 | 81 | schedules, stubs, EOM | **covered** | `foundation/schedule.py` + `tenor.py` | ng **supersets**: `Frequency`(Tenor) · `StubType` · `RollRule` · `ScheduleTerms`→**adjusted AND unadjusted** (C2) · EOM anchored once from start (ISDA §4.10) · **IMM + CDS rolls** (new). **shed:** long-stub merge heuristic (replaced by explicit stub spec, handoff §S3). | schedules |
| `core/rate_index.py` | 329 | 2 | RFR + IBOR indices | **covered** | `foundation/rate_index.py` + `underlying.py` | declarative index identity; F2/F3 decomposed (`IndexId`/`AccrualConvention`/`FixingRule`/`RfrConvention`); full RFR set (obs-shift/lookback/lockout/payment-delay); generic `FixingHistory`; `spread_adjustment`; **carries own `RollRule`** (F2 fix — no currency→calendar inference). **shed:** `overnight_indices`/`indices_for_currency` query helpers `deferred→topic-01` (curve-build queries, no ng consumer). | index-identity |
| `core/currency.py` | 135 | 4 | Currency enum + CurrencyPair | **covered** | `foundation/money.py` | `Currency` now an **open registry** (code + `minor_units`; S1/S4) · `CurrencyPair`(+spot lag) · `Money` (mixing = TypeError). **shed:** `CurrencyPair.forward_rate`/`forward_points` + `all_g10_pairs` `deferred→FX` (FX math on a convention object — reassigned in topic-01 manifest §REASSIGNED). | money-quantity |
| `core/settlement.py` | 398 | 0 | settlement types + per-product settle | **covered** | `foundation/settlement.py` | L0 **vocabulary** covered: `SettlementType` · `Delivery(date, Quantity)` · `SettlementTerms` · `settlement_date` (T+2); cash≠physical, settlement-ccy≠contract-ccy (4b). **shed:** per-product settlement *computation* (`cash_settlement`, `cds_settlement_*`, `option_settlement_*`, `futures_settlement_*`, `settlement_risk`) `deferred→products` (L4/L6 logic, not L0 — travel with each product). | settlement |
| `core/numerical_config.py` | 123 | 1 | numerical knob bag | **covered** | `foundation/numerical_config.py` | decomposed into `MonteCarloConfig`/`LatticeConfig`/`IntegrationConfig`/`SolverConfig` (gate §1 — no `fields-exempt`, passes on merit); tree folded into `LatticeConfig`. Full knob set pre-empts the retire-#1 retrofit. | numerics-config |
| `core/interpolation.py` | 298 | 56 | 5 interpolation schemes | **covered** | `foundation/interpolation.py` | L0 mechanism covered — **thin scipy adapters (S17)**: `Interpolation` enum (linear/log-linear ours; cubic/PCHIP/Akima via `scipy.interpolate`) + **per-end extrapolation policy** (`FLAT`/`CONTINUE_SLOPE`/`RAISE`, closes C4). **shed:** **Hagan-West monotone-convex** `deferred→topic-01` — curve construction (interpolates from interval averages), absent from scipy *and* both trees; **mine the quarry `forward_interpolation.py`** for it. | l0-numerics |
| `core/solvers.py` | 261 | 41 | root-find + optimise | **covered** | `foundation/solvers.py` | L0 mechanism covered — **thin scipy adapters (S17)**: `brent`/`newton`/`secant`/LM `least_squares` via `scipy.optimize`. The hand-rolled `bisect_root`/`nelder_mead` are **deleted** (S17 "no duplicates"). No shed residual (scipy supersets the quarry's Newton/secant/halley/itp/brentq). | l0-numerics |
| `core/serialisable.py` | 649 | 95 | serialisation **framework** | **covered** (pattern) | per-class `to_dict`/`from_dict` | framework **not carried by design** (#16 §2.6, handoff §S8 — "the 831 lines are not carried"). ng uses the per-class **pattern** (`schema_version` + `to_dict`/`from_dict`, **demonstrated on the `Leg` hard case** — nested value objects + enum + collection + `Cashflow\|Delivery` union, S8); the wider surface `deferred→persistence` (CP-3 ruling — never blocks a tick). | l0-serialisation |
| `core/serialization.py` | 182 | 4 | serialisation helpers | **covered** (pattern) | per-class `to_dict`/`from_dict` | same as `serialisable.py` — helper layer of the shed framework; superseded by the per-class pattern. | l0-serialisation |
| `core/data_registry.py` | 156 | 9 | import-time JSON convention registry | **dead** | — (ruled away) | the capability is **ruled away** (gate re-classification): its purpose is *import-time JSON registry loading*, which S5 forbids (explicit construction, no import-time I/O). **Nobody will build it** — no ng consumer, no future trigger. Parked `dead`. | l0-open-domains |
| `core/notional.py` | 56 | 4 | notional scalar/list expansion | **covered** (absorbed) | `foundation/money.py` + `foundation/cashflow.py` | the notional *value concept* is **absorbed** into `Money` (an amount) and `Leg` (per-flow amounts). **shed:** the scalar→list expansion for amortising profiles is **L2 product convenience** `deferred→L2` (a swap/bond notional schedule, not an L0 type). | l0-open-domains |

## REASSIGNED OUT OF TOPIC 0 (not parked here — moved to a named later topic)

| quarry file | LOC | → topic | why not L0 |
|---|---|---|---|
| `core/fixings.py` | 253 | **market-data / persistence** | ng's **immutable `FixingHistory` read model** (`foundation/rate_index.py`) is the correct L0 type and covers lookup + lag. The quarry's **mutable `FixingsStore` + CSV/JSON persistence** (`set`/`bulk_set`/`save`/`load_csv`) is file I/O and mutable state — **not L0**; it travels with a later market-data/persistence topic. *(My earlier L0/Topic-1 split was ratified; the correct target is market-data, not Topic 1.)* |

---

## Coverage

**12 covered · 1 dead · 0 blocking · 1 reassigned (`fixings`→market-data).** Every file in the handoff's
Topic 0 set is resolved. No omission is a deferred *product* (would block, §3.2.4); all shed items are
deferred *capabilities* (Hagan-West→topic-01, notional-expansion→L2, serialisation→persistence),
forward-linked. → **Topic 0 COVERED.**

**Physical parking (git mv):** **13** files → `parked/topic-00-foundation/` — the 11 covered + `data_registry`
(dead) + `notional` (absorbed). `fixings` stays in the quarry (reassigned→market-data).

**Drawdown:** first physical parking of the build (the pre-replan `fixed_income` retire *ticks* were
never git-mv'd and belong to Topic 1). **13 / 793 parked.** Reported, never steered (#12).
