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
        │     (depends on engine + Pricable protocol)    │  the ENGINE interface,
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
**Depends only on the engine interface + a `Pricable` protocol — never on concrete
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
```
