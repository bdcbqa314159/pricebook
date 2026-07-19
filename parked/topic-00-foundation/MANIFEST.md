# Topic 0 — Foundation (L0, cross-asset complete) · MANIFEST

Living tracker (#13 §3.3) for the **foundation** topic. Status: `target` → `covered` | `dead` |
`reassigned→X` → `parked`. Fan-in = production consumers in `python/pricebook/` (excl. own tests) —
**irrelevant to parkability** (#13 §2: ng never imports the quarry; the quarry's tests park with it).

Topic 0 is **part migration, part new construction** — `Money`, `Quantity`, `RollRule`, `Accrual`,
`CouponPeriod`, `ScheduleTerms`, IMM/CDS rolls, `Delivery`, `PricingResult`/`PricingFailure` have **no
quarry counterpart** (handoff §note). So the covered rows below are the migrated fraction; the rest of
Topic 0 is new L0 that stands on nothing to retire.

Spec: `redesign/16_topic0_foundation.md`. Close ruling: `rulings_topic0_gate.md` §Gate.
Gate audit (F1–F4 + S1–S16, all landed): `AUDIT_topic0_foundation.md`.

---

## THE SET (handoff `handoff_topic0_foundation.md` §"Topic 0 close")

| quarry file | LOC | fan-in | domain role | status | covered-by (ng) | evidence / shed | slice |
|---|---|---|---|---|---|---|---|
| `core/day_count.py` | 274 | 228 | 7 day-count conventions + `year_fraction` | **covered** | `foundation/day_count.py` | ng is a **superset** (10 conventions: 7 + ACT/365L · 30E/360-ISDA · NL/365); calendar passed in (BUS/252 raises, no São Paulo default); ICMA via `CouponPeriod`; degenerate periods raise (S14). **shed:** `date_from_year_fraction` `deferred→topic-01` (t→date curve helper, no ng consumer). | daycounts |
| `core/calendar.py` | 943 | 54 | 37 calendars + holiday-rule DSL | **covered** | `foundation/calendar.py` + `market_calendars.py` | all 37 declared; DSL (`fixed`/`easter`/`orthodox`/`nth`/`monday`, `since`/`until`); `JointCalendar`; three `Observance` regimes incl. Japanese *furikae* + Colombian Ley Emiliani; `DayType` + half-days (S5). Identity-keyed; currency→calendar a lookup (C1). | calendars |
| `core/schedule.py` | 143 | 81 | schedules, stubs, EOM | **covered** | `foundation/schedule.py` + `tenor.py` | ng **supersets**: `Frequency`(Tenor) · `StubType` · `RollRule` · `ScheduleTerms`→**adjusted AND unadjusted** (C2) · EOM anchored once from start (ISDA §4.10) · **IMM + CDS rolls** (new). **shed:** long-stub merge heuristic (replaced by explicit stub spec, handoff §S3). | schedules |
| `core/rate_index.py` | 329 | 2 | RFR + IBOR indices | **covered** | `foundation/rate_index.py` + `underlying.py` | declarative index identity; F2/F3 decomposed (`IndexId`/`AccrualConvention`/`FixingRule`/`RfrConvention`); full RFR set (obs-shift/lookback/lockout/payment-delay); generic `FixingHistory`; `spread_adjustment`; **carries own `RollRule`** (F2 fix — no currency→calendar inference). **shed:** `overnight_indices`/`indices_for_currency` query helpers `deferred→topic-01` (curve-build queries, no ng consumer). | index-identity |
| `core/currency.py` | 135 | 4 | Currency enum + CurrencyPair | **covered** | `foundation/money.py` | `Currency` now an **open registry** (code + `minor_units`; S1/S4) · `CurrencyPair`(+spot lag) · `Money` (mixing = TypeError). **shed:** `CurrencyPair.forward_rate`/`forward_points` + `all_g10_pairs` `deferred→FX` (FX math on a convention object — reassigned in topic-01 manifest §REASSIGNED). | money-quantity |
| `core/settlement.py` | 398 | 0 | settlement types + per-product settle | **covered** | `foundation/settlement.py` | L0 **vocabulary** covered: `SettlementType` · `Delivery(date, Quantity)` · `SettlementTerms` · `settlement_date` (T+2); cash≠physical, settlement-ccy≠contract-ccy (4b). **shed:** per-product settlement *computation* (`cash_settlement`, `cds_settlement_*`, `option_settlement_*`, `futures_settlement_*`, `settlement_risk`) `deferred→products` (L4/L6 logic, not L0 — travel with each product). | settlement |
| `core/numerical_config.py` | 123 | 1 | numerical knob bag | **covered** | `foundation/numerical_config.py` | decomposed into `MonteCarloConfig`/`LatticeConfig`/`IntegrationConfig`/`SolverConfig` (gate §1 — no `fields-exempt`, passes on merit); tree folded into `LatticeConfig`. Full knob set pre-empts the retire-#1 retrofit. | numerics-config |
| `core/interpolation.py` | 298 | 56 | 5 interpolation schemes | **covered** | `foundation/interpolation.py` | L0 **mechanism** covered (`Interpolation` enum + `interpolate`: linear/log-linear/cubic). **shed:** curve-specific schemes (Monotone/Akima/**Hyman-filtered**) `deferred→topic-01` — L1 curve *policy* (#16 §2.5, C4), no ng consumer; **mine this parked file** for Hagan-West when topic-01 builds curve interpolation. | numerics-config |
| `core/solvers.py` | 261 | 41 | root-find + optimise | **covered** | `foundation/solvers.py` | L0 mechanism covered (`bisect_root` bracketing root-find + `nelder_mead`). **shed:** `newton`/`secant`/`halley`/`itp`/`brentq` variants `deferred→topic-01` (curve/calibration solvers; `bisect_root` is the robust degenerate — mine parked file for Brent when convergence speed bites). | numerics-config |
| `core/serialisable.py` | 649 | 95 | serialisation **framework** | **covered** (pattern) | per-class `to_dict`/`from_dict` | framework **not carried by design** (#16 §2.6, handoff §S6 — "the 831 lines are not carried"). ng uses the per-class **pattern** (`schema_version` + `to_dict`/`from_dict`, demonstrated on `NumericalConfig`); per-class impl `deferred→persistence` (CP-3 serialisation ruling — never blocks a tick). | numerics-config |
| `core/serialization.py` | 182 | 4 | serialisation helpers | **covered** (pattern) | per-class `to_dict`/`from_dict` | same as `serialisable.py` — helper layer of the shed framework; superseded by the per-class pattern. | numerics-config |

## REASSIGNED OUT OF TOPIC 0 (handoff listed them, but retire-read → no ng L0 counterpart)

| quarry file | LOC | → topic | why not L0 |
|---|---|---|---|
| `core/data_registry.py` | 156 | **topic-01** | JSON ↔ **curve-convention** dataclasses (G10 table) — Topic 1 curve-convention data, not a foundation type. Already on topic-01's list. |
| `core/notional.py` | 56 | **topic-01** | `normalize_notional` (scalar↔list expansion for amortising) — an L2 product/schedule concern (swap/bond notional profiles), not L0. Already on topic-01's list. |

---

## Coverage

**11 covered · 2 reassigned→topic-01 · 0 dead · 0 blocking.** Every `target` resolved. No omission is a
deferred *product* (would block, §3.2.4); all are deferred *capabilities*, forward-linked to their
crossing topic. → **Topic 0 COVERED.**

**Physical parking (this manifest's git mv):** the 11 covered files → `parked/topic-00-foundation/`.
`data_registry` + `notional` stay in the quarry on topic-01's list (reassigned, not parked here).

**Drawdown:** first physical parking of the build (the pre-replan `fixed_income` retire *ticks* were
never git-mv'd and belong to Topic 1). **11 / 793 parked.** Reported, never steered (#12).
