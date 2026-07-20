# redesign/ — where we are, where we're going, what to read

---

## Where we are

**L0 (the foundation) is closed** at ng `v0.74.1`. 16 modules, ~2,400 LOC, zero debt, one legitimate
`fields-exempt` (`PricingResult`). **13 quarry modules parked** → `parked/topic-00-foundation/`.
Cross-asset by design: `Currency`/`Unit` open registries · `Tenor` · `Frequency` as a tenor-step
(bullet/28-day/daily) · `RollRule` · IMM & CDS rolls · `Money`/`Quantity` · `Flow = Cashflow |
Delivery` · `RateIndex` (full RFR set, carrying its own calendar) · `rate_basis` · settlement ·
`Underlying` protocol · scipy-backed numerics adapters.

**Drawdown: 13 / 768.**

---

## Forward scope — the sequence

**Two cross-asset foundations come before multicurve.** Everything above them — rates, credit,
equity, commodity — consumes them, so they are settled once rather than discovered per asset class.
This is the same rule that produced L0.

| # | phase | design | build | why it comes first |
|---|---|---|---|---|
| **F1** | **Market-data foundation** | ✅ `19_market_data_design.md` | ⬜ | every asset prices off a snapshot; the shape must be cross-asset |
| **F2** | **Model + calibrator foundation** | ⬜ **next session** | ⬜ | what a model *is*, what it expects, how it orchestrates a calibrator |
| **T1** | **Multicurve + linear rates** | ✅ `18_topic1_yield_curve.md` | ⬜ | the **first consumer** of F1 + F2 — and the proof they work |
| T2+ | credit · equity · commodity · inflation · FX · AAD · ML curves · XVA | — | — | each adds *keys*, never new foundation shape |

**F1/F2 are design-complete now, built with their first consumer.** They are not built speculatively
in isolation — that would be infrastructure ahead of a need (§6b). Their *contracts* are settled now;
the code lands as T1's opening work, written **generically** rather than rates-shaped, and T1 proves
them by pricing a real swap end to end.

**F1 settled (19):** `QuoteSet` → calibrate → `MarketSnapshot` · the dual-role bump rule · snapshot as
**closed shapes × open keys** (term structure · surface · scalar · series · schedule) · `CurveSet` with
typed accessors · adapter/resolver at the data spine · `CalibrationResult` beside the curve ·
**resolution safety** (reprice-to-par is blind to mis-resolution).

**F1 also completes the FX-spot algorithm (was foundation ledger `AC-3.6b`, now scoped work here, not
deferred).** L0 ships a *joint-count* `fx_spot_date`; F1 owns the rest, shaped by the market-data layer:
the **FX pair-conventions registry** (quote order, cross triangulation) and the **asymmetric ACI
intermediate-day rule** (a USD holiday on an intermediate day does not pause a USD pair's count) —
which must land **with a citable ACI source pinned to a verifiable worked example** (green-oracle gate;
until then the joint-count behaviour stands, documented in the `fx_spot_date` docstring).

**F2 open questions for the next session:** what a model carries and expects · the calibrator contract
(`quotes + spec → calibrated model + CalibrationResult`) · how the model orchestrates calibration ·
where the engine boundary sits · how this generalises past rates to credit/equity dynamics.

**T1 shape (18):** a **full vertical L1→L6** in three clusters — C1 market data + construction ·
C2 model + engine + products · C3 trade/portfolio + risk/stress. Most of C2/C3 is a **re-base from
`ng_parked/`**, not greenfield. Drawdown target **13 → ~50**.

---

## ► START HERE

| # | doc | what it is |
|---|---|---|
| 0 | **`independent_audits/foundation_audit/README.md`** | **foundation audit CLOSED (v0.75.0–v0.84.0)** — start here. The **15 deferrals grouped by the topic that will surface them**, what the three passes did, and the file index. **Read this rather than `OPEN.md`** — `OPEN.md` is the machine-side ledger. Full record + the re-verification and control documents sit beside it in the same folder. |
| 1 | **`../CLAUDE.md`** | the guardrails — law, not suggestion |
| 2 | **`19_market_data_design.md`** | **F1** — the market-data foundation |
| 3 | **`18_topic1_yield_curve.md`** | **T1** — multicurve scope + the L1→L6 vertical (§9) |

## ► THE METHOD
| doc | status |
|---|---|
| `13_topic_migration_and_parking.md` | **current** — topics · tick mechanism (§3) · target/use/**apply the policy** (§5) |
| `12_domain_build_order.md` | **current** — block order; drawdown reports, never steers |
| `11_checkpoint_and_review_cadence.md` | **current** — ≤6 slices or cluster; the five review inputs |

## ► REFERENCE WHILE BUILDING
| doc | status |
|---|---|
| `02_spine.md` | **current** — layers + **Amendments A1–A6** |
| `17_quarry_L0_classification.md` | **current** — where every remaining quarry-L0 file goes; T1's set |
| `15_foundation_comparison.md` | ours vs the quarry's L0 |
| `09_verification_and_audit.md` · `10_ci_and_cross_platform.md` | `verify.py`, CI matrix, tolerance oracles |
| `06_versioning…` · `07_branching…` | 0.x→1.0 · branch per slice · red-before-green |
| `handoffs/quarry_reconciliation.md` | living drawdown tracker |
| `parked/topic-01-yield-curve/MANIFEST.md` | T1 file set + findings from the quarry read |

## ► SUPERSEDED (kept for history — do not act on)
`03_vocabulary` (absorbed into 16/19) · `04_slice_plan` (topic method replaced it) ·
`14_topic1_object_model` (superseded by 18/19; C4's L0/L1 interpolation split was **withdrawn**) ·
`16_topic0_foundation` (Topic 0 closed) · `kickoff_slice0` · `L0_ledger.xlsx` (per-topic manifests
replaced it) · `01_scope_contract` — the scope decision stands, its drawdown framing does not.

## ► handoffs/
`quarry_reconciliation.md` is live. **Everything else is the reasoning record** — checkpoint reports
and `rulings_*`. Useful for *why*, never for *what next*. The ones that shaped the design most:
`rulings_deletable_definition` · `rulings_spotcheck_retire_1` (evidence protocol) ·
`rulings_CP3_correction` (phantom-residual rule) · `rulings_spine_conformance` (the `black.py`
precedent) · `AUDIT_topic0_foundation` (F1–F4, S1–S17).
