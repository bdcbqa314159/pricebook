# Checkpoint — Topic 0 GATE (foundation close)

**When:** Topic 0 close (cadence #11 — a layer/topic boundary). **Version:** `0.71.0`.
**Branch:** `slice/topic0-park`. **Preceding:** gate audit `AUDIT_topic0_foundation.md` (F1–F4 +
S1–S16) — all landed and merged; `rulings_topic0_gate.md` (NumericalConfig decomposition + §3c).

**Claim:** Topic 0's foundation is complete, oracle-gated, spine-conformant, and its 11 covered quarry
`core/` files are parked. **Ask:** ratify the gate → open Topic 1 (yield curves).

---

## What landed since the S3 checkpoint (`CP_topic0_s3_checkpoint.md`)

| slice | version | delivers |
|---|---|---|
| money-quantity | 0.60 | `Currency`/`CurrencyPair`(+lag) · `Money` · `Quantity` · `Cashflow` · `Leg` · `Accrual` |
| settlement (4b) | 0.61 | `SettlementType` · `Delivery(date, Quantity)` · `SettlementTerms` · T+2 `settlement_date`; cash≠physical; settlement-ccy≠contract-ccy |
| index-identity | 0.62–0.63 | declarative index; `RateIndex` (RFR + IBOR); generic `FixingHistory`; `spread_adjustment`; sibling underlyings |
| numerics-config | 0.65 | `NumericalConfig` decomposed (MC/Lattice/Integration/Solver); `PricingResult`/`PricingFailure`; serialisation pattern |
| **gate rework** | 0.64–0.70 | **F1–F4 + S1–S16** — tenor value type · F2 RateIndex-carries-RollRule · F3/S12 index decomposition + `AccrualMethod` · S5 Calendar `day_type`/`HolidaySet` · S11 day-count gaps · S14 degenerate-period raises · S1/S4 open `Currency`/`Unit` registries |
| **park** | 0.71 | 11 covered files → `parked/topic-00-foundation/`; manifests + roll-up |

Full L0 tier: **106 tests green.** Gates green: `layers` · `fields` (on merit, no config exemption) ·
`provenance` · `acyclic` · `debt` (0 suppressions) · `version`.

---

## Review input 1 — Oracle-quality audit

| module | oracle | tier |
|---|---|---|
| day_count | ISDA 2006 §4.16 worked examples · ICMA Rule 251 · **UST coupon = exactly 2.0000** (regression) · 30U/360 Feb edges | closed-form / published |
| calendar | published holiday & observance dates per market; Juneteenth `since=2021`; Store Bededag `until=2023`; furikae/Emiliani; joint-calendar union | published reference |
| schedule | ISDA §4.10 EOM anchoring; all four stubs; adjusted≠unadjusted under a holiday; published IMM + CDS roll tables | published reference |
| rate_index | compounded RFR vs hand-computed fixing series; **lookback≠observation-shift**; SONIA 0/0 vs SOFR 2/2; **F2 regression: SOFR-on-SIFMA ≠ same-ccy-TARGET over Columbus Day 2024** | closed-form / self-consistency |
| money-quantity | currency-mixing = TypeError; unit arithmetic closed same-unit only; accrual yf vs day-count primitive; **BRL end-to-end** (Currency·BUS/252·São Paulo) | type-level / closed-form |
| settlement | cash≠physical; settlement-ccy≠contract-ccy; T+2 lag over a holiday | closed-form |
| numerical_config | round-trip `to_dict`/`from_dict` nested; decomposition ≤5 fields each | self-consistency |

**No oracle rests on quarry cross-check** (the weakest tier, and demonstrably unreliable here — the ICMA
fallback priced a UST coupon at 1.9836). All are published-reference or closed-form. **Audit: clean.**

## Review input 2 — Quarry-drawdown reconciliation

**11 / 793 parked** (`parked/topic-00-foundation/MANIFEST.md`). First physical drawdown of the build.
Every file in the handoff's Topic 0 set is classified `covered` (11) or `reassigned→topic-01` (2:
`data_registry`, `notional`) with consumer-analysis evidence. **0 dead, 0 blocking** — every shed
feature is a deferred *capability* (forward-linked to its crossing topic), never a deferred *product*.
Pre-replan CP-1…CP-4 `fixed_income` retire *ticks* are frozen history (never git-mv'd; belong to
Topic 1). `quarry_reconciliation.md` is now a thin topic-method roll-up.

## Review input 3 — "Challenge me" (design choices to contest)

1. **`data_registry` + `notional` reassigned, not parked here.** The handoff listed both in Topic 0's
   park set, but the retire-read found **no ng L0 counterpart** — `notional` is L2 (amortising profile),
   `data_registry` is Topic-1 curve-convention data. I moved them onto topic-01's list rather than
   building L0 machinery with no consumer (§6b). *Contest if you intended them built at L0.*
2. **`interpolation`/`solvers` parked as `covered` on the L0 mechanism only.** ng has linear/log-linear/
   cubic + bisect + Nelder-Mead. The curve schemes (Hyman/Monotone/Akima) and Brent/Newton are
   `deferred→topic-01`, to be **mined from the parked copies**. A file parks once; Topic 1 mines it. Is
   parking-with-forward-mine the right call, or should these two hold as partial crosses in the quarry?
3. **serialisation "covered by pattern."** The 831-line framework (`serialisable`+`serialization`) is
   shed by design (#16 §2.6); ng has the per-class `to_dict`/`from_dict` pattern demonstrated only on
   `NumericalConfig`, rest `deferred→persistence`. Is a demonstrated *pattern* (not a populated
   surface) enough to tick the framework `covered`?
4. **`fixings` split L0/Topic-1.** ng's read-model `FixingHistory` (lookup + lag) covers the L0 need;
   the mutable store + CSV/JSON persistence is reassigned→topic-1/market-data. `core/fixings.py` was
   **not** in the git-mv set (it stays in the quarry under topic-01) — but the handoff listed it in
   Topic 0's park set. *Flagging the deviation explicitly.*
5. **`fields-exempt` on `PricingResult`.** One marker remains (output record, §3b-legal). The gate
   ruling said "pass on merit"; I read §3b as permitting the marker for genuine output records. Confirm
   this is not the config-smuggling §3b bans.

## Review input 4 — Smell + debt scan

- **Debt ledger:** `verify.py debt` green — **0 suppressions**, `OPEN.md` carries no new Topic-0 entry.
- **Fields:** every `foundation/` dataclass ≤5 on merit; the sole `fields-exempt` is `PricingResult`
  (output record) — challenge-me #5.
- **Layers:** `verify.py layers` green — no finance at L0 (the `black.py` drift that motivated the gate
  cannot recur).
- **Signatures:** no `PLR0913` suppressions; the two that appeared (`_on`, `_unadjusted`) were fixed by
  bundling into `RfrConvention` / `ScheduleTerms`, not silenced.
- **Quarry half-state:** parking leaves the working-tree quarry with broken imports (`core/day_count`
  etc. gone). **Expected and accepted** (#13 §1 — the quarry is read-only reference, not a running
  system; runnability lives in git). Quarry-regression CI already dropped at ng-parking.

---

## Named next checkpoint

**Topic 1 — first internal checkpoint** at the first of: (a) ≤6 Topic-1 slices, or (b) the capability
boundary **curve capability + `CurveSet` built** (slice 4 of #13 §4). Topic 1 slice order is #13 §4
(conventions-completion → RateIndex → curve capability → CurveSet → pillar quotes → …). Scoping pass
(the `parked/topic-01-yield-curve/MANIFEST.md` target set) is already drafted; refresh it before slice 1.

**Do not begin Topic 1 until this gate is ruled** (`rulings_topic0_gate.md` §Gate + #16 §"Checkpoint").
