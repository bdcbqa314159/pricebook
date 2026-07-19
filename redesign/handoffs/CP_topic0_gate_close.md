# Checkpoint — Topic 0 GATE CLOSE (foundation parked)

**When:** Topic 0 close — all 8 slices of `HANDOFF_topic0_gate.md` landed + parking done.
**Version:** `0.73.0`. **Branch:** `slice/topic0-gate-park`.
**Claim:** the L0 foundation is complete, oracle-gated, spine-conformant, scipy-backed, and its quarry
set is parked. **Ask:** ratify the gate → open Topic 1 (multicurve + linear rates).

---

## The gate checklist (`HANDOFF_topic0_gate.md` §"Topic 0 gate")

| # | item | state |
|---|---|---|
| 1 | all `[BLOCK]` + `[solid]` items landed (S1–S17, F1–F4) | ✅ Slices 1–8 |
| 2 | `verify.py fields` green **on merit** — zero `fields-exempt` except `PricingResult` | ✅ |
| 3 | `layers` · `acyclic` · `debt` · `version` · `provenance` · ruff green | ✅ |
| 4 | two regression oracles: SOFR-on-SIFMA ≠ generic-USD ; BRL end-to-end | ✅ both pass |
| 5 | park the Topic 0 set → `parked/topic-00-foundation/` | ✅ **13 parked** |
| 6 | refresh `quarry_reconciliation.md` roll-up; report per redesign/11 | ✅ this report |

**125 L0 tests green.** All slices merged to `main` through v0.73.0.

## Slice ledger

| slice | delivered | built vs pre-existing |
|---|---|---|
| 1 `l0-tenor-frequency-roll` | S7 `date+Tenor` · S8 `RollConvention.IMM/CDS` anchor · S3 | S7/S8 **built**; S3 prior |
| 2 `l0-rate-basis` | S12 `Compounding`/`convert_rate`, `AccrualMethod` rename | prior (verified) |
| 3 `l0-index` | F2 index-carries-RollRule · F3 decompose · F4+S10 `Underlying`+siblings | prior (verified) |
| 4 `l0-flows` | S13 signed · S2 `Leg` of `Cashflow\|Delivery` · S14 raise | prior (verified) |
| 5 `l0-numerics` | scipy adapters (dist/solvers/interp) · per-end extrapolation · F1 · S15 (S17) | **built** |
| 6 `l0-open-domains` | S1 open `Currency` · S4 open `Unit` · S11 daycount completeness | prior (verified) |
| 7 `l0-results-and-invariants` | S6 `PricingResult` · S5 `day_type` · S9/S16 invariants | prior (verified) |
| 8 `l0-serialisation` | per-class `to_dict`/`from_dict` + `schema_version` on `Leg` hard case | **built** |

---

## Review input 1 — Oracle-quality audit

Every landed item carries a published-reference or closed-form oracle; **none rests on quarry
cross-check** (the weakest tier). Highlights: ISDA §4.16 / ICMA 251 (UST coupon = exactly 2.0000);
published holiday/observance + IMM/CDS roll tables; `date+3M` month-clamp; lookback≠observation-shift;
SOFR-on-SIFMA ≠ generic-USD over Columbus Day; scipy adapters asserted against published `Φ`/`√2`/linear-
fit values; PCHIP no-overshoot; `Leg` serialisation round-trip (nested + enum + collection + union) +
`schema_version` rejection; BRL end-to-end (Currency·BUS/252·São Paulo). **Audit: clean.**

## Review input 2 — Quarry-drawdown reconciliation

**13 / 793 parked** (`parked/topic-00-foundation/MANIFEST.md`) — the first physical drawdown.
11 covered + `data_registry` (**dead** — import-time JSON registry ruled away, S5) + `notional`
(**absorbed** into `Money`/`Leg`; scalar→list expansion `deferred→L2`). `core/fixings` is
**reassigned→market-data** (immutable `FixingHistory` read model is the L0 type; mutable store + file I/O
is not L0) — **not** parked here. 0 blocking; every shed item is a forward-linked deferred *capability*.
`quarry_reconciliation.md` is the thin topic roll-up; the pre-replan CP record stays frozen history.

## Review input 3 — "Challenge me"

1. **`data_registry` parked `dead` (not reassigned).** The spotcheck ruled its capability away
   (no import-time I/O, S5); I took that literally — no forward trigger, nobody rebuilds it. Contest if
   a curve-convention *loader* is wanted later (it would be new code, not this file).
2. **`notional` parked `covered`/absorbed**, with the amortising scalar→list expansion `deferred→L2`.
   The L0 concept is just "an amount" (`Money`); the *profile* is an L2 product schedule. Agree it's not
   an L0 residual?
3. **`fixings` split L0 (read model) / market-data (store).** My earlier split was ratified; I routed
   the store to *market-data*, not Topic 1. Confirm the target topic name.
4. **scipy exact-pins** (`numpy==2.4.3`/`scipy==1.17.1`) in CI — reproducibility vs upgrade friction.
5. **Serialisation ticked on one demonstrated hard case** (`Leg`), the rest `deferred→persistence`.
   Sufficient to retire the framework, per the CP-3 "serialisation never blocks a tick" rule?

## Review input 4 — Smell + debt scan

- **Debt:** `verify.py debt` green — **0 suppressions**; no Topic-0 `OPEN.md` entry. The two PLR0913s
  that appeared this gate (`_on`, `_extrapolate`) were fixed by bundling / deriving, never silenced.
- **Fields:** on merit; sole `fields-exempt` is `PricingResult` (output record, §3b-legal).
- **Layers:** `verify.py layers` green — scipy/numpy are numerics; no finance vocabulary at L0 (the
  `black.py` drift that motivated the gate cannot recur).
- **Spine conformance (5th input):** foundation stays L0 finance-free; scipy wrapped behind our API +
  provenance as the single C++-port swap point; products are pure data; failure is a value.

---

## Named next checkpoint

**Topic 1 — first internal checkpoint** at the first of: ≤6 Topic-1 slices, or the **curve capability +
`CurveSet`** boundary (#13 §4). Before Topic-1 Slice 1: run the scoping pass and refresh
`parked/topic-01-yield-curve/MANIFEST.md` (its conventions rows now point at the parked topic-00 copies;
mine `parked/topic-00-foundation/interpolation.py` for Hagan-West, `solvers.py` for Brent/Newton if
curve-solve needs them).

**Topic 1 does not begin until this gate is ratified by Cowork** (`HANDOFF_topic0_gate.md` §gate).
