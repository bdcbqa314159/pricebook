# Checkpoint — Topic 0 GATE (the foundation complete)  ·  stop, review, then park the set

The cross-asset foundation (L0) is built. **This is the gate**: on the ruling, the Topic-0 quarry
`core/` set is parked to `parked/topic-00-foundation/` and Topic 1 (yield curves) may begin — not
before. One checkpoint covers S4b/S5/S6 per the settlement-ruling cadence.

**Versions:** v0.55.0 (parking) → **v0.63.0**. **Foundation:** 13 modules, 1743 LOC. **Tests:** 81 L0
oracles green. **Gates:** `acyclic · layers · fields · debt · version · provenance` all green.

---

## 1. What Topic 0 delivered (the slices)

| slice | v | delivered |
|---|---|---|
| 0 `ng-parking` | 0.55 | parked all prior ng → `ng_parked/`; clean seed; **`verify.py layers`** gate |
| 1 `calendars` | 0.56 | DSL + 3 observance regimes + furikae + JointCalendar + **37 markets**, identity-keyed |
| 2 `daycounts` | 0.57 | 10 conventions (7 + ACT/365L, 30E/360-ISDA, NL/365); `CouponPeriod`; strict ICMA; calendar-required BUS/252 |
| 3 `schedules` | 0.58 | `RollRule`/`ScheduleTerms`/`Schedule` (adjusted **and** unadjusted); EOM-from-start; **IMM + CDS rolls** |
| — S3 corrections | 0.59 | ANZAC per-rule `observed`; ACT/365L frequency-dependent; `Coverage` marker (8 EM secular-only) |
| 4 `money-quantity` | 0.60 | `Money` (mixing = TypeError) · `Currency`×37 · **`Quantity`/`Unit`** · `CurrencyPair` · `Cashflow`/`Leg`/`Accrual` (+`Accrual.year_fraction`) |
| 4b `settlement` | 0.61 | `SettlementType` (cash/physical/**auction**-marker) · **`Delivery(Quantity)`** · settlement ccy ≠ contract · `settlement_date` |
| 5 `index-identity` | 0.62 | widened `RateIndex` (all rate kinds: `observation_style`, `spread_adjustment`) · generic `FixingHistory` · `accrued_rate` (lookback≠obs-shift, **EXPONENTIAL BUS/252**, `exponential_growth`) · explicit registry |
| 6 `numerics-config` | 0.63 | full `NumericalConfig` (+serialisation pattern) · `distributions` · `solvers` · `interpolation` (mechanism) · `PricingResult`(+`DiscountBasis`)/`PricingFailure` |

---

## 2. Oracle-quality audit

**Every Topic-0 oracle is a published external reference** — the strongest tier for conventions
(there is no closed form for a holiday calendar; the authority *is* the statute/ISDA rule). No
self-consistency-only oracle in the entire topic.

- **calendars:** national statutes (5 U.S.C. §6103, UK BFDA 1971, NZ Holidays Act), observance
  divergences, Juneteenth/Store-Bededag year-gates, furikae, ANZAC no-shift regression.
- **daycounts:** ISDA 2006 §4.16 worked examples; ICMA Rule 251 (**UST semi-annual coupon = exactly
  2.0000** — a regression oracle: the deleted `strict_icma` priced it 1.9836).
- **schedules:** ISDA §4.10 EOM; four stubs; published IMM/CDS tables.
- **index-identity:** hand-computed compounded RFR; lookback ≠ observation-shift; forward-term ≠
  backward-compounded; fallback = base + spread; **Brazilian exponential flat rate reprices to itself
  exactly** (0.10 → 0.10, vs money-market 0.05 → 0.05001).
- **numerics:** published norm_cdf/ppf values; bracketed root; Nelder-Mead minimum; ISDA A2
  clean = pv − accrued.

---

## 3. Quarry-drawdown reconciliation — the Topic-0 set (park on the ruling)

Topic 0 **supersedes** these quarry `core/` modules; on the gate ruling they `git mv` to
`parked/topic-00-foundation/`. Classification (Cowork spot-checks the `dead`/`covered` calls):

| quarry `core/` | status | covered-by (ng) / evidence |
|---|---|---|
| `calendar.py` | covered | `foundation/calendar.py` + `market_calendars.py` |
| `day_count.py` | covered | `foundation/day_count.py` |
| `schedule.py` | covered | `foundation/schedule.py` |
| `rate_index.py` | covered | `foundation/rate_index.py` |
| `fixings.py` | covered | `rate_index.FixingHistory` (generic over index) |
| `currency.py` | covered | `money.Currency` |
| `interpolation.py` | covered | `foundation/interpolation.py` (mechanism) |
| `solvers.py` | covered | `foundation/solvers.py` |
| `numerical_config.py` | covered | `foundation/numerical_config.py` (rebuilt complete) |
| `settlement.py` | covered | `foundation/settlement.py` |
| `notional.py` | **dead** | ng carries amounts as `Money`; no `normalize_notional` shim needed |
| `data_registry.py` | **dead** | registries are **explicit construction** — the import-time JSON reload (one bad row dropped 27) is the bug we refused to inherit |
| `serialisable.py` · `serialization.py` | **dead** | the 831-line framework is shed for the **per-class `to_dict`/`from_dict` + `schema_version`** pattern |
| `forward_interpolation.py` | **reassigned→Topic-1** | curve forward-rate interpolation is an L1 curve policy, not an L0 mechanism |

**Topic-0 tally:** 10 covered · 4 dead · 1 reassigned = **14 core/ modules retired, 1 moved on.** Global
roll-up refreshed at parking (reported, never chased).

---

## 4. Design choices to challenge

Already ruled this topic (carried for the record): observe-lift → per-rule `observed` override;
ACT/365L frequency-dependent; `Coverage`/lunar deferred-but-marked; `CouponPeriod` serving three
conventions; numeraire = L3 (not `Money`). **Open for this gate:**

1. **`CompoundingMethod.EXPONENTIAL` + basis-from-day-count + `exponential_growth`** (Brazilian BUS/252)
   were added mid-slice on a user challenge. Confirm the shape — the basis is generic (`_annual_basis`,
   `BUS/252 → 252`), and the fixed-`r` (LTN/NTN-F) vs floating-`rᵢ` (CDI) split is `exponential_growth`
   vs `accrued_rate`. Is the four-method category (COMPOUNDED/EXPONENTIAL/AVERAGED/FLAT) complete?
2. **`DiscountBasis` on `PricingResult`** records only `collateral: Currency | None`. Enough to make a
   PV unambiguous (currency in `pv` + collateral here), or does the basis need the discounting
   *curve/index* identity too (that lands at L1/L3)?
3. **Serialisation pattern established on `NumericalConfig` only.** The other L0 value types
   (`Money`/`Cashflow`/`RateIndex`/…) don't serialise yet — deferred until a persistence consumer
   (`deferred→persistence`, the CP-3 §4.5 rule). Confirm that's the right call at the gate.
4. **`interpolation` raises on extrapolation** (extrapolation is an L1 curve policy). Confirm L0 owes
   only the in-range mechanism.
5. **`Delivery` delivers a `Quantity` only** — a defaulted **security** (CDS physical / repo / bond
   future) needs a security identity (higher layer); forward-linked, marked not silent. Confirm the
   boundary.
6. **`FixingHistory` is `Mapping[str, Mapping[date, float]]`** — a single float per (index, date). Fine
   for rates and inflation levels; FX/equity observations that need a *source/time* stamp will extend it
   when their sibling index type lands. Accept the float-now shape?

---

## 5. Smell + debt scan

- **`verify.py debt` green — zero suppressions** across all 13 modules (no `# type: ignore`, `# noqa`,
  `# pragma`, skips, or load-bearing TODOs). Two `# pragma: no cover` were removed rather than ledgered.
- **No third-party runtime deps** — stdlib only (no `dateutil`, no `numpy`); `hypothesis` is test-only.
- `fields` green: the two legitimately-wide aggregates (`NumericalConfig`, `RateIndex`) carry explicit
  `# fields-exempt` markers; every other value type is ≤5 fields. `PLR0913` green (`year_fraction` at the
  5-ceiling, resolved via `Accrual.year_fraction` + `CouponPeriod`).
- The judgment calls that arose mid-topic (observe-lift, exponential gap, deliver-a-security) were
  surfaced — to the user or in-code forward-links — not taken silently.

---

## 6. Spine-conformance audit (5th input)

Every module placed **by what it is**; `verify.py layers` (L0 finance-free) is CI-wired and green.

- **L0 finance-free confirmed** for all 13 — calendars/day-counts/schedules/money/quantity/settlement/
  rate-index/config/distributions/solvers/interpolation/results. No strikes, vols, payoffs, discounting-
  *factors*, or option vocabulary. The one place finance vocabulary appears (settlement's `AUCTION`,
  results' "collateral/discounting") is **naming a convention**, not computing value — and is a
  documented marker, with the mechanics (recovery, numeraire choice) explicitly pushed to L1/L3/credit.
- **Dependencies point down / stay in-layer:** `market_calendars→calendar`, `schedule→calendar`,
  `day_count→calendar` (TYPE_CHECKING), `cashflow→{money,day_count,calendar}`, `rate_index→{cashflow,
  day_count,market_calendars,money}`, `settlement→{calendar,money}`, `results→money` — all equal-rank L0.
  `acyclic` green. Nothing reaches up; no `Money`-into-pricing leak.
- **`black.py` cannot recur:** the semantic gate that the drift slipped past now exists and runs per
  commit.

---

## 7. Ready-for-gate / named next

**Recommended gate action (on the ruling):** `git mv` the 14 covered/dead `core/` modules to
`parked/topic-00-foundation/`; forward-link `forward_interpolation` onto Topic 1's row; refresh the
reconciliation roll-up to a thin per-topic summary.

**Named next checkpoint — the Topic 1 gate.** Topic 1 (yield curves, `redesign/14`): `CurveSet`,
`CurveHandle`, the `RateIndex` capstone, bootstrap, and curve risk — built on this foundation, checkpointed
at *its* gate. Topic 1 does not begin until Topic 0's gate is green here.

**Ask for Cowork:** rule §4 (the six challenge items — especially #1 EXPONENTIAL, #2 `DiscountBasis`
completeness, #3 serialisation-deferred), spot-check the §3 `dead`/`covered` classifications, and green
the gate to authorise the parking + Topic 1.
