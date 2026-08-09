# Artifact #20 — The Foundation Contracts (cross-asset, settled before multicurve)

**Status:** Ratified. **The single place the vertical's contracts live.** Part A
consolidates what was already ratified but scattered across Amendments A1–A6 and a dozen rulings;
Part B designs the five genuine gaps.

**Why before multicurve:** every contract here is consumed by *every* asset class. Discovering them
rates-first is the retrofit trap we have already paid for twice (single-curve under an XVA stack;
a multicurve-scoped L0). Same rule that produced L0 and F1: **design complete, populate
incrementally.**

---

# PART A — already ratified (consolidated, not re-opened)

| contract | rule | source |
|---|---|---|
| **Product** (L2) | frozen, pure data — legs, cashflows, payoffs. **No `pv()`.** Pricing lives in L4. | A1 · spine §2 |
| **Model** (L3) | carries the `MarketSnapshot` it was calibrated to (`model.market`). Market is *upstream* of the model, never a peer input. | A1 |
| **Engine** (L4) | `price(product, model, numerics) → PricingResult`. Stateless: referentially transparent · no ambient state · frozen inputs · failure-as-value · explicit `NumericalConfig` · valuation-date-aware. | A1 · A2 |
| **Trade → Book** (L6) | `Trade` = collection of products + start date + lifecycle. `Book` = trades. Dates live at the trade, not the product. | D4 · A3 |
| **Realized vs mark** (L6/L4) | **realized P&L** = cashflows already paid (benefit table, *undiscounted*, remembered by the shell). **mark** = future PV + accrued (computed by the engine). `dirty = clean + accrued`. | A3 |
| **Risk** (L5) | **perturb market data → rebuild the model → reprice.** Nothing bumps a model directly. `Priceable` protocol; no `isinstance` ladders. | A1 · A6 |
| **Market data** (L1) | `QuoteSet` → calibrate → `MarketSnapshot`; closed shapes × open keys; dual-role bump rule. | #19 |

## Part A addendum — two pre-Topic-1 rulings (post-closure seam pass, 2026-07-20)

Both are **contracts settled now, code lands with the first consumer** — the same treatment F1/F2
get (settle the shape, build at first use), and each is L0's ~30 lines rather than Topic 1's, so no
curve invents its own. Recorded here; **not built** (30 unconsumed lines is what Phase 4 deleted).

- **A1 — `TimeMeasure` is the only sanctioned `date → t` mapping.** A frozen
  `TimeMeasure(anchor: date, day_count: DayCountConvention)` with `t(d) -> float` is the single
  authority for turning a date into a year-fraction from an anchor. **No curve or model pairs an
  anchor and a day-count ad hoc** — the drift that produces (each curve measuring time differently)
  is the failure class this design exists to prevent. Build it as an **L0 module** in Topic 1's
  first slice, with its first curve consumer. *(Promotes audit AC-T4.5 from "someday" to "before
  Topic 1 opens".)*

- **A2 — `Frequency ↔ ICMA frequency: int` bridge, ruled.** A `Frequency.per_year() -> int`
  **raises** for any tenor with no integer periods-per-year (28D, daily, bullet). And **BUS-period
  products (TIIE, CDI) do not enter ICMA contexts at all** — that is the recorded answer to the
  "undecidable 365/28" mapping, not a silent rounding. Implement with the **fixed-leg builder**
  that first needs the conversion. *(Resolves audit AC-T4.15 / the `Frequency`-bridge item.)*

---

# PART B — the five gaps (designed here)

## B1. Calibrator contract
```python
calibrate(quotes: QuoteSet, spec: CalibrationSpec) -> tuple[CalibratedModel, CalibrationResult]
```
- **`CalibrationSpec`** declares *what is being solved*: the target(s), the instruments/quotes that
  pin them, the method (sequential / simultaneous), and its `NumericalConfig` slice. It is **data**,
  not a callable — so a calibration is reproducible and serialisable.
- **`CalibrationResult`** (sibling of the model, never an attribute on it — #19 §5): quotes used ·
  method · residuals · iterations · convergence · **the parameter→quote Jacobian**.
- **The model does not orchestrate its own calibration.** `calibrate` is a free function at L3;
  the model is its *output*. (A model that calibrates itself would hold state and reach for market
  data — both forbidden by A1.)
- **Universal oracle:** every calibrating instrument reprices to its quote. For curves that is
  reprice-to-par; for a vol model it is match-the-quoted-vol. Same shape for every asset class.

## B2. Product protocol — what every product exposes
Today products are frozen dataclasses with **no common interface**, and engines dispatch by type.
The moment the engine registry lands, that becomes an `isinstance` ladder — the exact thing removed
from risk. So define the minimum now:

```python
class Product(Protocol):
    @property
    def currency(self) -> Currency: ...          # the natural currency of the payoff
    def underlyings(self) -> tuple[Underlying, ...]: ...   # what market data it needs
```
- **Deliberately minimal.** Not `cashflows()` — a swaption has none; not `pv()` — that is L4's job.
- `underlyings()` is what lets a trade declare its market-data requirements **without the engine
  knowing the concrete type** — it is how the registry selects an engine and how risk knows what to
  bump.
- Rule of two is met: rates, credit, FX, equity all need it.

## B3. Lifecycle event vocabulary
A3 says `BookedTrade` carries "lifecycle events" but never defines them. Closed set of *shapes*,
open set of *payloads* (the same meta-rule as market data):

| event | carries | effect |
|---|---|---|
| `Fixing` | index · date · value | resolves a float period |
| `Payment` | date · amount | moves a flow from mark → **realized** (benefit table) |
| `Exercise` | date · which right | changes the product set of the trade |
| `NotionalChange` | date · new notional | amortisation / accretion |
| `Settlement` | date · type · amount/delivery | cash · physical · auction |
| `CreditEvent` | date · type | credit (payload defined by the credit topic) |
| `Novation` / `Termination` | date · counterparty | trade-level |

**Rule: events are append-only and dated.** A `BookedTrade`'s state at any date is a *fold* of its
events up to that date — so history is replayable and a valuation is reproducible. Events never
mutate the trade's description.

## B4. P&L attribution
A3 gives realized vs mark, but not *why* P&L moved. Attribution is a **decomposition of the change**
between two valuation dates:

```
ΔP&L = carry/theta        (time passing, same market)
     + market move        (market data changed → the risk explains it)
     + lifecycle          (payments, fixings, exercises)
     + new/removed trades
     + unexplained        ← must be reported, never silently absorbed
```
- **Sequential (waterfall) attribution:** apply each effect one at a time, revaluing between steps —
  order is declared, because attribution is order-dependent.
- **`unexplained` is a first-class output.** A silent plug is how attribution lies; a visible residual
  is how it earns trust.
- **Oracle:** the components sum to the actual P&L exactly; on a static market with no lifecycle
  events, all components except carry are zero.

## B5. Risk aggregation and book-level scenarios
```python
class Scenario(Protocol):
    name: str
    def apply(self, quotes: QuoteSet) -> QuoteSet: ...      # or snapshot, per the dual-role rule
```
- **Scenarios transform market data, never trades** — consistent with A1/A6, and it means one
  scenario applies to a book of any composition.
- **Composable:** scenarios compose (a rates shock ∘ an FX shock); composition order is explicit.
- **Aggregation is a sum over trades in a *stated* currency**, with the FX conversion basis recorded
  (`DiscountBasis` already does this for a PV — the same discipline for aggregates).
- Sensitivities aggregate **only when the bump is identical** — same key, same shift, same rebuild
  policy. Aggregating a par delta with a zero delta is meaningless; the contract must make that a
  type-level impossibility rather than a convention.

---

## What this unlocks
With A + B settled, the five build stages have contracts to build *against* rather than discover:

```
S1 market-data foundation   (#19)
S2 model + calibrator       (B1)
S3 multicurve construction  ← the first consumer; proves S1+S2
S4 products + engine        (B2)
S5 trade/portfolio/risk/P&L (B3 · B4 · B5)
```
Each asset class after rates adds **keys, products and event payloads — never a new contract.**
That is the test: if credit or equity forces a change to Part A or Part B, the design was wrong here.
