# Artifact #2 — The Spine (DRAFT)

**Status:** Draft for reaction. This is the constitution: the layers and the permitted
direction of dependencies. Once ratified, every migrating entry aligns to this shape.

---

## Organizing principle: functional core, imperative shell

The system splits in two, and the split is the whole design:

- **Functional core (stateless).** Pricing is a pure function.
  `price(trade, market, model, numerics) → result`. No hidden state, no thread-locals,
  no cached calibration on the engine, no eager side effects. Identical inputs always
  produce identical outputs. This is what "stateless engine" means, precisely.
- **Imperative shell (stateful).** Everything that persists — a booked trade, its
  lifecycle, valuations over time, the database — lives *around* the core and only ever
  *calls into* it. The shell remembers; it never computes prices itself.

Why this delivers all five requirements at once:

| Requirement | Where it lives |
|---|---|
| Stateless engine that takes market data → builds models → prices | **Core**, L1–L4 |
| Create a trade and monitor it over its whole life | **Shell**, L6 lifecycle |
| Risk: greeks, XVA, RWA | **L5** — the engine re-run under bumped/simulated markets |
| Small but meaningful database | **Data spine** + shell persistence |
| Clean, easy API | **Facade** over core + shell |

---

## The layers (dependencies point DOWN only, never up)

```
        ┌───────────────────────────────────────────────┐
 SHELL  │ L6  LIFECYCLE / BOOKING / PORTFOLIO            │  stateful
 (state)│     trade booking, monitoring, P&L, desks, viz │
        └───────────────────────────────────────────────┘
                          │ calls into
        ┌───────────────────────────────────────────────┐
        │ L5  RISK & CAPITAL                             │
        │     greeks · XVA · RWA/regulatory              │  depends only on
        │     (depends on engine + Priceable protocol)    │  the ENGINE interface,
        └───────────────────────────────────────────────┘  never on concrete trades
                          │
        ┌───────────────────────────────────────────────┐
 CORE   │ L4  ENGINES  ── the stateless heart            │  PURE FUNCTION
(pure)  │     bind instrument + model + market → PV/risk │
        └───────────────────────────────────────────────┘
                 │                    │
        ┌────────────────┐   ┌─────────────────────────┐
        │ L3  MODELS     │   │ L2  INSTRUMENTS          │
        │ dynamics,      │   │ declarative trade descr. │  DATA, not behaviour;
        │ calibration    │   │ legs · cashflows · payoff│  knows nothing above it
        └────────────────┘   └─────────────────────────┘
                 │                    │
        ┌───────────────────────────────────────────────┐
        │ L1  MARKET DATA                                │
        │     snapshot · quotes · fixings · curves · vol │  immutable, built FROM
        └───────────────────────────────────────────────┘  a snapshot
                          │
        ┌───────────────────────────────────────────────┐
        │ L0  FOUNDATION                                 │
        │     time & conventions · value types · numeric │  finance-agnostic core
        └───────────────────────────────────────────────┘

   ┌──────────────┐
   │ DATA SPINE   │  DuckDB/ECB ingestion → feeds L1.  Infrastructure at the side,
   │ (side)       │  wired into the shell, never imported by the core.
   └──────────────┘
```

### L0 — Foundation
Time & conventions (dates, day counts incl. ACT/ACT ICMA + 252/exponential, calendars,
schedules, compounding), core value types (`Money`, `Rate`, `Tenor`, cashflow),
serialisation, and the finance-free **numerical toolkit** (MC, PDE, Fourier/COS,
optimize, AAD, distributions). Depends on nothing in pricebook.
*Maps from:* `core`, `numerical`, `statistics`.

### L1 — Market data
The **immutable `MarketSnapshot`**: quotes, fixings, and everything built from them —
discount / projection / hazard curves, vol surfaces. A snapshot is a value object: a
frozen picture of the market at an instant. Curves are built *from* a snapshot with the
link recorded (fixes the "curves conflated with quotes" defect). Depends on L0 only.
*Maps from:* `market_data`, `curves`, `data`.

### L2 — Instruments
The **declarative trade description** — legs, cashflows, payoffs. Pure data, no
behaviour. An instrument does **not** know about market data or models; it cannot price
itself. This is what lets one trade be priced by many engines and lets risk depend on a
protocol, not a class. Depends on L0 only.
*Maps from:* instrument definitions in `fixed_income`, `credit`, `options`, `fx`,
`equity`, `commodity`, `crypto`, `structured`.

### L3 — Models
Dynamics: Black-76, Hull-White, G2++, LGM, LMM, SABR, Heston, jump/rough, stochastic
intensity. A model is **calibrated to a snapshot** and is otherwise stateless. Includes
the **unified calibration front** (fixes scattered calibration). Depends on L0 + L1.
*Maps from:* `models`, `calibration`.

### L4 — Engines (the stateless heart)
The only layer that binds everything below: `engine.price(instrument, model, snapshot,
numerics) → PricingResult` (PV, cashflows, sensitivities, diagnostics). **Holds no
state.** All reproducibility knobs — seeds, MC paths, PDE grids, tolerances — arrive as
an explicit `NumericalConfig` input, never as scattered defaults (fixes the implicit-
config debt). The engine registry lives here (fixes leaky top-level `registry.py`).
*Maps from:* `pricing`, `registry`.

### L5 — Risk & Capital
Greeks (bump the snapshot, re-run the engine), XVA (simulate many snapshots, run the
engine per path, aggregate exposures), RWA / regulatory capital (SA-CCR, FRTB, SIMM).
**Depends only on the engine interface + a `Priceable` protocol — never on concrete
instrument classes.** This is the big structural fix: risk moves from its current L3
placement (where it switches on `isinstance`) to above the engine. Statelessness is what
makes all three tractable and parallel: no state to reset between bumps or paths.
*Maps from:* `risk`, `regulatory`.

### L6 — Lifecycle / Booking / Portfolio (the stateful shell)
Where a trade is **created, persisted, and monitored over its whole life.** A booked
trade carries its immutable description + a lifecycle position (coupons paid, options
exercised, fixings observed, settlements). Monitoring = for each valuation date, pull
that date's snapshot, run the stateless engine, store the result. Portfolios/books
aggregate; P&L attributes; desks organise; viz reports. This layer *orchestrates* the
core; it never re-implements pricing.
*Maps from:* `desks`, `pe`, `ts`, `viz` + a new `lifecycle`/`booking` home.

### Data spine (side infrastructure)
DuckDB (target; currently SQLite) + ECB/FRED/BOE ingestion feeding L1 snapshots.
Persists the **small, meaningful** set: entities, trades, market snapshots, pricing
results, P&L history. Wired into the shell; the core never imports it.

---

## The stateless engine contract (the load-bearing decision)

```python
result = engine.price(
    instrument,          # L2 — pure description, no behaviour
    model,               # L3 — calibrated to `market`, otherwise stateless
    market,              # L1 — immutable MarketSnapshot
    numerics,            # L0 — explicit NumericalConfig (seeds, paths, grids, tol)
)   # -> PricingResult(pv, cashflows, sensitivities, diagnostics)  — a value object
```

Invariants that make "stateless" real and testable:
1. **Referential transparency** — same four inputs ⇒ identical result, always.
2. **No ambient state** — no thread-locals, no globals, no clock reads; "today" comes
   from the snapshot.
3. **No mutation of inputs** — snapshot, instrument, model are frozen.
4. **Failures are values** — a `PricingFailure` result, not a raised exception or a
   silent NaN (fixes the exception-driven failure debt).
5. **Config is explicit** — reproducibility never depends on a hidden default.

Everything else in the system is defined by how it *uses* this function.

---

## The stateful lifecycle (how a trade is monitored for life)

A booked trade is the shell's unit of state:

```
book(instrument) -> BookedTrade(id, instrument, lifecycle_events=[])
   value(date)   -> pull snapshot(date) → engine.price(...) → store PricingResult
   observe(fix)  -> append fixing/exercise/settlement to lifecycle_events
   monitor()     -> value() each date over the trade's life; persist the series
```

The trade's *state* is its lifecycle position — never its price. Price is always
recomputed statelessly from the snapshot of the day. This is what lets you replay,
audit, and reprice history deterministically.

---

## Clean API (thin lid over core + shell)

```python
import pricebook as pb

pv       = pb.price(trade, market)                 # stateless one-shot (core)
book     = pb.book(trade)                           # persist, begin lifecycle (shell)
val      = book.value(date)                         # monitor as of a date
greeks   = pb.risk(trade, market, ["delta","vega","xva","rwa"])
```

One obvious entry point per intent; layering hidden; hard to misuse.

---

## The three structural fixes this spine encodes

1. **Risk above instruments** — `risk` L3 → L5, protocol-based, no `isinstance` ladders.
2. **Unified calibration** — a single `calibration` front at L3 over per-family solvers.
3. **Registry relocated** — top-level `registry.py` → the L4 engine layer.
Plus two carried from the scope contract: **schema versioning** on serialisation, and
**`pe` re-homed** to the L6 shell rather than its accidental L0 placement.

---

## Ratified decisions (2026-07, Bernardo)

1. **Instruments are pure data.** ✅ Frozen dataclasses that describe legs/cashflows/
   payoffs only — no `pv()`/`pv_ctx()` behaviour. All pricing lives in the L4 engine.
   During migration, each instrument's valuation logic is lifted out into an engine.
   This deletes the `isinstance` risk ladders and makes statelessness structural.
2. **Persistence: SQLite now, behind a swappable interface.** ✅ Keep the working SQLite
   spine; define a clean persistence interface at the data spine. DuckDB is a scheduled
   *low-cost swap* (DuckDB reads SQLite files directly), taken when migration reaches the
   data-spine layer — not a redesign dependency.

## Open questions (non-blocking; resolve at the relevant layer)

3. **Engine registry, one facade.** *(default adopted)* A registry of engines
   (discounting, Black, MC, PDE, tree) selected per instrument+model, unified behind a
   single `price()` facade — the QuantLib-style pattern. Revisit only if a case breaks it.
4. **Snapshot granularity** — global `MarketSnapshot` per valuation date, with per-trade
   *views* for XVA performance. *(default; confirm when we reach the risk/XVA layer.)*

---

## Amendment A1 (2026-07) — engine depends on model, not market

Supersedes the earlier `price(instrument, model, market, numerics)` contract.

- **New signature:** `price(instrument, model, numerics) → PricingResult`.
- **Market is upstream of the model, not a peer input:** `market → calibrate → model →
  price`. A model is a **`CalibratedModel` that carries the `MarketSnapshot`** it was
  calibrated to (`model.market`); the engine reads curves/vols through the model.
- **Linear products** (bonds, swaps, the Slice-0 cashflow) use a thin **`DiscountingModel`**
  that wraps the discount curve — a real type with many consumers (rule of two), not
  ceremony. Its "calibration" trivially adopts the curve.
- **Why:** removes the model/market mismatch class of bug (a model calibrated to one
  snapshot can't be priced against another), and makes "pricing depends on the model"
  literally true.
- **Consequences:** refactors the Slice-0 `DiscountingEngine` to take a `DiscountingModel`;
  risk (a market bump) now flows through re-deriving/rebinding the model — more correct,
  and a real change to how greeks are computed. `MarketSnapshot` is unchanged as a
  vocabulary type; it is simply reached through the model.

## Amendment A2 (2026-07) — valuation is temporality-aware

The engine reasons about three dates — **start** (accrual/issue), **valuation** (as-of,
`model.market.valuation_date`), **payoff** (payment):

- Cashflows with `payoff ≤ valuation` are **historical** → excluded from PV (the L6 shell
  handles them via fixings/settlement); never discounted with a non-positive `t`.
- Cashflows with `payoff > valuation` discount from `valuation`.
- `start > valuation` (forward-starting) and `start < valuation < payoff` (seasoned:
  accrued interest, clean vs dirty price) are handled explicitly.
- Reset/fixing dates `≤ valuation` use realized `FixingHistory`; `> valuation` use
  forward-implied.

Gets its own slice with closed-form temporal oracles (seasoned bond excludes the paid
coupon; forward-starting; accrued/clean/dirty).

## Amendment A3 (2026-07) — Product/Trade/Book + realized-vs-mark decomposition

Consolidates the domain-hierarchy and temporality thread (Cowork + build session).

**Hierarchy (renames the L2 atom):**
- **Product** (L2, frozen, pure data) — the priceable atom; needs a model. (Was "instrument".)
- **Trade** (L6, frozen description) — holds a *collection of products* + a **start date** +
  lifecycle. Priced as the sum of its products' marks.
- **Book** (L6) — a collection of trades.
- Dates (start, valuation) live at the **trade**, not the product.

**The snapshot is curves + fixings.** The economy a model is built on = discount/projection/
hazard curves **and** `FixingHistory`. `FixingHistory` is a first-class part of the
immutable `MarketSnapshot` (L1) — the core needs fixings to resolve current-period amounts.
`build(snapshot) → CalibratedModel` (A1) consumes both.

**Valuation of a live trade = remembered past + deterministic accrual + core-priced forward:**
- **Realized P&L (benefit table)** — cashflows that already paid: actual cash, recorded in the
  **L6 shell** (`BookedTrade` remembers it; ties to the quarry `pnl_history`). Never discounted.
- **Accrued** — earned-but-unpaid slice of the current period: part of the *mark*, computed by
  the engine (dirty = clean + accrued).
- **Future PV** — remaining flows discounted from the valuation date.
- **The engine computes the mark** (future PV + accrued); **the shell remembers realized P&L.**

**Engine I/O grows a decomposition.** `PricingResult` is not a scalar: it carries **dirty PV**
plus the cashflow/accrual breakdown needed to derive **clean** and **accrued** (already specced
to hold cashflows/sensitivities/diagnostics).

**"Fail on past cashflow" is retired.** The Slice-0 guard that raised on a cashflow ≤ valuation
is replaced by **segment-and-settle**: partition each product's cashflows into
past (settle → benefit table), current (accrue), future (price). No raise.

**Risk** perturbs the snapshot and rebuilds the model (A1), then reprices — consistent across
PV / greeks / XVA / RWA.

## Amendment A4 (2026-07) — rulings from the v0.2–v0.8 build reports

Ratified from `redesign/handoffs/amend_A1_A2_A3_report.md` §5 and
`forward_v0.3-v0.8_report.md` §5.

**A4.1 — Migration is demand-driven (vertical).** Confirmed. Slices pull quarry entries as
they need them (`ng-migration-mode`); the quarry empties by demand, not layer-by-layer. Progress
is tracked via each report's *ledger-deltas* table, not a pre-ruled per-layer ledger. The
L0/L1 ledger cockpit is retired as the primary tracker (kept only as reference).

**A4.2 — Market data curves are ALL first-class in `MarketSnapshot`.** The
survival/hazard curve **is** market data → it moves out of `CreditModel` into the
`MarketSnapshot` (alongside `discount_curve` + `FixingHistory`). `CreditModel` is *built
from* it (A1). Consequence: credit greeks (`credit01`) flow through the **same `Priceable`
/ bump-the-snapshot** path as `dv01` — one greek machinery, no per-family bypass. Rule of
thumb: *if risk bumps it, it lives in the snapshot.*

**A4.3 — Building-block math may live at the layer that owns the curve/dynamics; the engine
composes.** Ratified boundary: a **curve** (L1) may expose closed-form building blocks
(`df`, `RPV01`, `cds_par_spread`), and a **model** (L3) may expose analytic blocks (`B(t,T)`,
zero-bond-option) — reused upward. The **L4 engine composes these to price the product**;
the **product stays pure data** and never prices itself. "Pricing lives in L4" governs
*product* pricing (the binding of product+model), not every scalar of math. Mirrors
QuantLib; avoids duplicating or inverting the bootstrap's dependency direction.

**A4.4 — Confirmed defaults (simplicity-aligned).**
- **Engine/model registry** lands with the **first genuinely mixed trade / multi-method
  product**, not before (open question #3 stays open until then).
- **`PricingResult`** grows its per-cashflow breakdown / sensitivities / diagnostics
  **on demand** (a consumer arrives), not speculatively.
- **Demand-migrate the minimum**: pulling `random.Random` instead of the quarry MC toolkit
  is correct; migrate the smallest thing that satisfies the oracle.
- **Clean/dirty semantics**: `accrued` is **nominal (undiscounted)**; `clean = dirty −
  accrued` (market convention). Ratified.

## Amendment A5 (2026-07) — `MarketSnapshot` internal shape: keyed registry

Ratified from `redesign/handoffs/decision_market_data_shape.md`. A4.2 ruled *what* lives in
the snapshot; this rules *how it's shaped*. **Option C, adopted now (before commodity).**

**Why now (not speculative):** §6b introduces an abstraction *when the second real consumer
arrives*. FX was the first "spot/vol/curve keyed by an underlying"; **equity is the second**,
so the rule-of-two is satisfied — C now is the rule firing on schedule, not early. The real
prize is **greek de-duplication** (one `bump_spot`/`bump_vol`/`spot_delta`/`vol_vega` instead
of per-asset copies — killing the copy-paste bug class), plus namespacing that removes the
latent `Currency "EUR"` vs ticker `"EUR"` collision.

**The shape:**
```python
class AssetClass(Enum): FX; EQUITY; CREDIT; CMDTY; INFLATION; ...   # closed dimension, typed
@dataclass(frozen=True)
class MarketKey: asset: AssetClass; id: str      # id open: ccy / ticker / issuer

@dataclass(frozen=True)
class MarketSnapshot:
    valuation_date: date
    discount_curve: CurveHandle                  # HOME NUMERAIRE — stays special (structural)
    fixings: FixingHistory = ...
    curves: dict[MarketKey, CurveHandle] = {}    # survival, dividend, foreign-discount, projection
    spots:  dict[MarketKey, float] = {}          # FX + equity + commodity spots
    vols:   dict[MarketKey, float] = {}          # flat vols
```

**Rulings:**
- **A5.1** — `discount_curve` (home numeraire) stays a **named field** (always present, what
  valuation-date discounting uses); it is not per-asset risk data.
- **A5.2** — `survival_curve`, `fx_*`, `equity_*`, and foreign-discount curves **fold into the
  keyed maps**. Folding survival *adds* capability (multi-issuer, vs today's single `Optional`).
- **A5.3** — `MarketKey(asset: AssetClass, id: str)` — enum namespace (typed/exhaustive) + open
  string id.
- **A5.4** — greeks collapse to one generic each (`bump_spot`/`bump_vol`/`bump_curve`,
  `spot_delta`/`vol_vega`) on the `Priceable` path; the per-asset `fx_*`/`equity_*` variants are
  deleted.
- **Timing/oracle** — its own **behaviour-preserving** slice guarded by the existing FX/equity/
  credit oracles (PVs + greeks byte-identical); commodity then lands with *no* snapshot edits.
```
