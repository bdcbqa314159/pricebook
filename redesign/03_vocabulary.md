# Artifact #3 — Core Vocabulary (DRAFT)

**Status:** Draft for reaction. These are the stable value types every layer speaks
through. Coupling lives or dies here: if all layers agree on these nouns, the layers stay
decoupled; if a shared noun lives high in the stack, coupling leaks back down.

**Grounded in the real tree.** Most of this already exists — the job is to *pin* it,
promote the few that sit too high, and freeze the two that are still mutable.

---

## Rule for what belongs in the vocabulary

A type is core vocabulary iff **more than one layer must speak it**. Such types live at
L0 (or L1 for market nouns), are **immutable value objects**, are **serialisable with a
version**, and carry **no behaviour that reaches upward**. Everything else stays local to
its layer.

---

## The nouns (grouped), with current home → target

### Time & conventions  (L0 — exists, keep)
| Noun | Now | Note |
|---|---|---|
| `Date` | python `date` | keep std lib; no custom date |
| `DayCountConvention` + `year_fraction()` | `core/day_count.py` | ACT/ACT ICMA, 252, 30/360 — keep |
| `Calendar`, `BusinessDayConvention` | `core/calendar.py` | keep |
| `Frequency`, `StubType`, `Schedule` | `core/schedule.py` | keep; `Schedule` as a value type |
| `Tenor` | string (`"3M"`) + `tenor_to_years/date` | **decision: promote to a real `Tenor` type?** |
| `CompoundingMethod` | `core/rate_index.py` | keep |
| `RateIndex` | `core/rate_index.py` (frozen, serialisable) | keep — the model to copy |

### Money & rates  (L0 — partly missing)
| Noun | Now | Note |
|---|---|---|
| `Currency`, `CurrencyPair` | `core/currency.py` | keep |
| **`Money` (amount + currency)** | ✗ — amounts are raw `float` | **biggest decision, below** |
| `Rate` | raw `float` | decision: keep float, or a thin typed `Rate`? |

### The instrument atom  (needs promotion)
| Noun | Now | Target |
|---|---|---|
| **`Cashflow`** | `fixed_income/fixed_leg.py` (**L3!**) | **promote to L0.** The shared atom of every instrument must not live inside one asset class. |
| `Leg` | scattered per asset class | define a common `Leg` = ordered `Cashflow`s + convention |

### Market nouns  (L1 — mostly exists, good)
| Noun | Now | Note |
|---|---|---|
| `Quote`, `QuoteId`, `QuoteKind` | `market_data/_types.py` (frozen) | keep — already right |
| `MarketSnapshot` | `market_data/_types.py` (frozen) | keep — the L1 anchor already exists |
| `FixingHistory` / `FixingsStore` | `market_data/_types.py`, `core/fixings.py` | reconcile the two into one |
| **`CurveHandle`** | ✗ — `DiscountCurve`/`SurvivalCurve` are concrete | **decision: introduce a handle/protocol** so higher layers depend on a stable reference, not a concrete curve class |

### Engine I/O  (L0/L4 — partly exists)
| Noun | Now | Note |
|---|---|---|
| `NumericalConfig` | `core/numerical_config.py` (frozen) | keep — the explicit-config type the spine requires |
| **`PricingResult`** | ad-hoc returns | **promote to one value type**: pv, cashflows, sensitivities, diagnostics |
| **`PricingFailure`** | exceptions / NaN | **new**: failure-as-value (spine invariant #4) |

### Identity & lifecycle  (L0/L6)
| Noun | Now | Target |
|---|---|---|
| `Trade` | `core/trade.py` (**not frozen**) | **freeze**: an immutable description |
| `Portfolio` | `core/trade.py` | keep |
| `PricingContext` | `core/pricing_context.py` (**not frozen**) | **freeze** or dissolve into `(MarketSnapshot, NumericalConfig)` per the stateless-engine contract |
| `BookedTrade` | ✗ | **new L6**: description + lifecycle events (the shell's unit of state) |

---

## The three vocabulary decisions that matter

### 1. `Money` type vs. raw floats  ← biggest
Today every amount is a bare `float`; currency is tracked separately (or implicitly). A
`Money(amount, Currency)` value type makes currency-mixing a **type error, not a silent
bug** — you cannot add USD to EUR by accident. For a correctness-and-education library
this is high value. Cost: it touches every cashflow and PV in the tree during migration.
*Options:* full `Money` everywhere / `Money` only at boundaries (PV, cashflow) with
floats inside hot loops / keep floats.

### 2. `CurveHandle` (stable reference) vs. concrete curves
Higher layers currently import `DiscountCurve`/`SurvivalCurve` directly. A `CurveHandle`
(a protocol: `df(date) -> float`, `survival(date) -> float`) lets models/engines depend
on the *capability*, not the concrete class — so a curve implementation can change
without rippling upward, and curves stop being mutable behind higher layers' backs
(fixes the `curve_bumper` in-place-mutation anti-pattern). *Recommend: yes.*

### 3. Freeze `PricingContext` and `Trade`; dissolve context toward the snapshot
Both are mutable dataclasses today. The stateless engine takes an **immutable**
`MarketSnapshot` + `NumericalConfig`. So either freeze `PricingContext` into an immutable
bundle, or dissolve it — the engine takes `(instrument, model, snapshot, numerics)`
directly and `PricingContext` becomes just a convenience grouping. *Recommend: freeze
now, dissolve opportunistically during migration.*

---

## Why this is the anti-coupling artifact

Every cross-layer edge in the current tree is a place where two layers had to agree on a
type. Pin the agreement here, once, and those edges become safe. The two smells this
already surfaces — `Cashflow` living at L3 and curves being concrete+mutable — are
exactly the kind of "shared noun in the wrong place" that let coupling grow back. Fixing
them in the vocabulary is cheaper than fixing them in every consumer.

---

## Ratified decisions (2026-07, Bernardo)

1. **`Money` at boundaries.** ✅ `Money(amount, Currency)` at cashflows and PVs; plain
   floats inside hot numerical loops (MC/PDE). Currency-mixing becomes a type error where
   it matters, without a performance cost in the kernels.
2. **`PricingContext`: freeze and keep** (do **not** dissolve). ✅ Evidence: it is
   referenced in 76 files / 188 times / ~127 call sites, and its fields already ARE the
   built market state (curves, vols, fx, `numerical_config`) — its docstring even calls
   it an "immutable snapshot of market data." Dissolving = quarry-wide mechanical churn
   for zero correctness gain; the real defect is that it's a mutable `@dataclass`. Fix:
   **freeze it**, reframe as the engine's built-market-state input, and record its link
   to the quote-level `MarketSnapshot` it was built from (quotes → build → context).
3. **`Tenor`** — keep strings + `tenor_to_years/date` helpers *(default adopted; no real
   type)*. Revisit only if string tenors cause a concrete bug.
4. **`Rate`** — keep raw floats *(default; `Money` already covers the currency-safety
   need)*.
5. **`CurveHandle`** — introduce the protocol *(default adopted)*: models/engines depend
   on `df(date)`/`survival(date)` capability, not concrete curve classes; kills the
   in-place `curve_bumper` mutation.

Two promotions carried into the slice plan: **`Cashflow` L3 → L0**, and **freeze
`Trade`**.
