# Artifact #15 — Foundational layer: ours vs the quarry's L0

---

## 1. The definitional difference (this causes everything else)

**Quarry L0 was defined by IMPORT TOPOLOGY:** "imports nothing from `pricebook`". It is a *mechanical*
classification — `tools/layer_deps.py` computed it. Nothing checked whether a module *meant*
anything foundational. That is how `pe/` (LBO & DCF — portfolio analytics), `viz/`, `ts/` and `db/`
came to sit in L0: they happen to import nothing internal.

**Our L0 is defined SEMANTICALLY:** time & market conventions · value types · finance-free numerics.
**No pricing, no payoffs, no dynamics, no market state.** The sharp line:
> *Conventions describe how time and money are counted; pricing computes value.*
A day count is a convention. Black-76 is pricing. That is why `black.py` failed the gate and
`day_count.py` does not — the test is **no valuation**, not "no finance words."

**Consequence:** the acyclic check can prove direction but never membership. Semantic conformance
needs `verify.py layers` + the 5th checkpoint review input.

---

## 2. Scale

| | quarry L0 | ours |
|---|---|---|
| definition | imports-nothing-internal (mechanical) | conventions + value types + finance-free numerics (semantic) |
| modules | **111** across 8 sub-packages | ~15 objects |
| sub-packages | core 30 · numerical 31 · statistics 18 · viz 14 · ts 8 · pe 5 · db 3 · market_data 2 | one `foundation/` |
| numerics | 49 modules (numerical + statistics) carried wholesale | demand-migrated: interpolator · solvers · distributions |

---

## 3. `core/` module by module — where each actually belongs

Of the quarry's **29 `core/` modules, only ~11 are genuinely foundational.** The rest belong higher
or elsewhere — i.e. **~62% of "core" was not core.**

| quarry `core/` | LOC | our home |
|---|---|---|
| `calendar.py` | 943 | **L0 `Calendar`** — but keyed by **identity**, not currency (C1) |
| `day_count.py` | 274 | **L0 `DayCountConvention` + `year_fraction`** — calendar passed in; `strict_icma` deleted |
| `schedule.py` | 143 | **L0 `ScheduleTerms`/`Schedule`** — returns **both** adjusted and unadjusted (C2) |
| `rate_index.py` | 329 | **L0 `RateIndex`** — + `RollRule` (calendar!) + full RFR set |
| `currency.py` | 135 | **L0 `Currency`** (the `CurrencyPair` pricing half → FX topic) |
| `fixings.py` | 253 | **L0 `FixingHistory`** |
| `interpolation.py` | 298 | **L0 `Interpolator`** (pure math) + **L1** curve policy (C4) |
| `numerical_config.py` | 123 | **L0 `NumericalConfig`** — *already retired, 3 fields vs 15* |
| `solvers.py` | 261 | **L0** finance-free numerics |
| `data_registry.py` | 156 | **L0** convention loading — fix the all-or-nothing JSON replace |
| `notional.py` | 56 | folded into **`Money` / `Leg`** |
| `discount_curve.py` | 300 | **L1 market** (`YieldCurve`) — it is market *state*, not a convention |
| `survival_curve.py` | 271 | **L1 market** → credit topic |
| `pricing_context.py` | 375 | **L1 `MarketSnapshot`** (the model carries it, A1) |
| `forward_interpolation.py` | 256 | **L1** curve interpolation — *and its "monotone convex" is not Hagan-West* |
| `greeks.py` | 39 | **L5 risk** |
| `trade.py` · `book.py` · `daily_pnl.py` | 705 | **L6 shell** |
| `settlement.py` · `mandate.py` | 768 | **L6 shell** |
| `market_conventions.py` | 210 | **reassigned** → equity/commodity/inflation (misleading name; nothing rates) |
| `serialisable.py` · `serialization.py` | 831 | **not carried as a framework** — per-class `to_dict`/`from_dict` |
| `approximation.py` · `numerical_safety.py` · `convergence_framework.py` | 1175 | numerics — demand-migrate when a consumer lands |
| `caching.py` · `dependency_graph.py` | 394 | infrastructure; production-orphans in the quarry |

---

## 4. Objects we have that the quarry has NOT

These are the design's additions — mostly **un-bundled primitives becoming value objects**:

| ours | why | quarry equivalent |
|---|---|---|
| **`Money(amount, currency)`** | currency-mixing becomes a type error | *none* — amounts are bare floats |
| **`Accrual(start, end, day_count)`** | one accrual period, one object | 3 loose args |
| **`CouponPeriod`** (ICMA anchors) | ICMA needs anchors; not 3 more args | `ref_start`/`ref_end`/`frequency` loose |
| **`RollRule(calendar, convention, eom)`** | the adjustment rule is one thing | 3 loose args on a 6-arg `generate_schedule` |
| **`ScheduleTerms`** + dual date sets | accrual needs unadjusted, payment adjusted | one flat adjusted list — accrual dates **lost** |
| **`RateIndex` carrying `RollRule`** | an index knows its own calendar | no calendar field — inferred from currency |
| **full RFR set** (shift/lookback/lockout/delay) | metadata must be declarative | `observation_shift` only; **lookback inexpressible**, lockout absent |
| **`PricingResult` / `PricingFailure`** | decomposition + failure-as-value | ad-hoc returns; exceptions/NaN |
| **Calendar by identity** | joint calendars, NY vs SIFMA | keyed by currency (their own admitted flaw) |

---

## 4b. The 18 non-foundational `core/` modules — analysis DEFERRED (ruled)

Their retire-reads happen **when their topic is picked up**, not now (#13 §3: the delta only exists
once the ng counterpart does; analysing earlier means planning from a feature-diff — the
phantom-residual trap). Recorded here only so nothing is lost:

**Still Topic 1, later clusters (4):**
| module | cluster |
|---|---|
| `discount_curve` · `pricing_context` · `forward_interpolation` | curves (L1) |
| `greeks` | curve risk (L5) |

**Other topics — defer entirely (14):**
| modules | topic |
|---|---|
| `survival_curve` | credit |
| `trade` · `book` · `daily_pnl` · `settlement` · `mandate` | L6 shell / lifecycle |
| `market_conventions` | equity / commodity / inflation |
| `approximation` · `numerical_safety` · `convergence_framework` | numerics |
| `caching` · `dependency_graph` | infrastructure |
| `serialisable` · `serialization` | **cross-cutting — needs its own decision** (fan-in 95; we are not carrying the framework, only per-class `to_dict`/`from_dict`) |

## 5. The shape of the change

The quarry's L0 was a **bucket of things that happened not to import upward** — 111 modules including
portfolio analytics, plotting, time series and a database.

Ours is a **deliberate vocabulary**: ~15 objects that more than one layer must speak, with the
composites built from the leaves, and `RateIndex` as the capstone that makes multi-curve work.

The recurring transformation is the same one the signature rule enforces: **loose primitives become
value objects** (`Money`, `Accrual`, `CouponPeriod`, `RollRule`), and **state moves up** out of the
foundation (curves → L1, greeks → L5, trade/book/P&L → L6).
