# Checkpoint CP-2b — general-curve build + first spine vanilla

Date: 2026-07-17   ·   Version: `0.42.0`   ·   Tests: 243 green (`verify.py all`)

Second checkpoint under the cadence (`redesign/11`), first **parity-depth** cluster. Covers the
three slices since the CP-1 rulings (`rulings_CP1.md`) that executed the ruled priority order —
general curve (#1) → general-curve HW (#2) → first fixed_income spine vanilla (#3). Reconciliation
map refreshed alongside (`quarry_reconciliation.md`). Carries the four mandatory review inputs.

---

## 1. Slices landed (CP-2b cluster)

| Slice | Version | Layer | What |
|---|---|---|---|
| general-curve-rates | 0.40.0 | L1 | `zero_rate` + `instantaneous_forward` on flat & bootstrapped curves (piecewise-constant forward) |
| general-curve-hw | 0.41.0 | L3 | HW reads `f(0,t)`/`df` from ANY curve; `forward_short_rate` + path sim date-based; byte-identical on flat |
| fra-spine | 0.42.0 | L2/L4 | `ForwardRateAgreement` + `FRAEngine` — single-curve forward-vs-fixed, prices on any curve |

**Ledger deltas:** `market/discount_curve.py` + `market/snapshot.py` gained the rate accessors
(shared `_bracket_slope`). `models/hull_white.py` de-flattened (r0 → `f(0,t)`, time axis from the
snapshot). `engine/swaption_mc.py` + `risk/exposure.py` moved to date-based rate simulation. New
`products/fra.py` + `engine/fra.py`. ng module count 49 → 51.

## 2. Oracle-quality audit

| Slice | Oracle | Class |
|---|---|---|
| general-curve-rates | flat → constant; forward integrates to `-ln df`; `zero_rate = -ln df/t` | **closed-form** |
| general-curve-hw | curve refit `zero_bond(0,S,f(0,0)) == P^M(0,S)`; ZCB put-call parity; flat-pillars == flat curve (exact); **analytic swaption == MC on a bootstrapped curve** | **closed-form + cross-check** |
| fra-spine | par (K=L) → 0; closed-form off-par PV; sign flip; par → 0 on a bootstrapped curve | **closed-form** |

All strong; none self-consistency-only. The general-curve-hw MC cross-check (`rel=2%`) is backed by
the analytic swaption, not standalone.

## 3. Quarry-drawdown reconciliation (refreshed)

- **768 quarry · 51 ng · deletable: 0. Drawdown still 0.0/768** — but three gaps narrowed:
  - `core/discount_curve`: rate accessors added (2 of 3); **`forward_rate` + pluggable interpolation
    + `roll_down`** remain → not deletable.
  - `models/hull_white`: the **flat-curve gap (the biggest) is CLOSED** — HW is now general-curve.
    Remaining vs the module: constant vol (no term-structure), no per-currency/tree. Closest to parity.
  - `fixed_income/fra`: **new** (was untouched backlog) → partial (forward-starting only; seasoned
    needs a fixing).
- **Honest caveat (process gap):** no *rigorous per-module parity audit* has been done — I have not
  read each quarry counterpart end-to-end to confirm nothing is missing. So "deletable" is asserted
  conservatively as 0. **Proposed:** every parity slice from here ends by reading its quarry
  counterpart and listing the residual, so a module can be *confirmed* deletable (and drawdown ticks).

## 4. Design choices to challenge

1. **FRA forward computed in the engine (composition), not a `curve.forward_rate` method.** Follows
   "pricing composes building blocks" (CLAUDE.md). The quarry curve exposes `forward_rate`; ng does
   not. Right call — or add `curve.forward_rate` for interface parity? (rule-of-two trigger at OIS,
   the 2nd forward consumer.)
2. **HW `curve` property is typed `CurveHandle` but calls `instantaneous_forward`** (duck-typed —
   the protocol only declares `df`; survival curves have no forward). Deferred a new `RateCurve`
   protocol to avoid a cross-cutting abstraction mid-cluster. Introduce it now, or at the 2nd
   forward-curve consumer?
3. **`forward_short_rate` + the path simulator went date-based.** Byte-identical, cleaner (they need
   `f(0,t)`). Confirm the direction (vs year-fraction + a passed forward).
4. **Deletable-bar rigor (§3):** endorse the "read the counterpart, confirm coverage" step per parity
   slice so drawdown becomes real?

## 5. Smell + debt scan

- **HW `curve` duck-types `instantaneous_forward`** (see §4.2). — *proposed: accept; `RateCurve`
  protocol at the 2nd consumer.*
- **`_bracket_slope` recomputes pillar `ts`/`lns` on every `df`/`forward` query** (O(n) per call) —
  inherited from the original `df`; fine for small curves, a cache is the upgrade if a hot loop bites.
  — *accept, ceiling noted.*
- **Process (not code): a branch mix-up** — general-curve-hw was committed onto the previous slice's
  branch before I noticed; recovered cleanly (rebased the 3 commits onto main, nothing lost). Reminder
  logged: create the slice branch *first*. — *note only.*
- No suppressions; `verify debt` green.

## 6. Ready-for-next / named next checkpoint

Parity order continues at fixed_income (#3). **Named next checkpoint (CP-2c): fixings + spine.**
- **Fixings / seasoned float (high leverage):** consuming `FixingHistory` unblocks the seasoned FRA
  and swap AND the **L6 float-leg realized P&L** (the benefit table's deferred gap) — one cross-cutting
  piece serving both the spine and the shell.
- Plus **deposit** and **OIS / RFR compounding** as the remaining vanilla spine, each ending with the
  §3 parity confirmation against its quarry module.

**Requesting Cowork:** ratify §4 (esp. the deletable-bar step §4.4 and the `RateCurve` protocol
timing §4.2), and confirm CP-2c = fixings-first.
