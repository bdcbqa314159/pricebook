# Checkpoint — the inflation vertical (T1–T6)

**Date:** 2026-10-31 · **By:** Cowork · **Baseline:** v0.6.0, PR #7 · **Scope:** the cluster from
the minimal spine through JY calibration. First checkpoint of the reset repo.

## 0. Status — the arc Bernardo chose is built and oracled
The full inflation arc, on the T1 stateless spine:

| tranche | capability | anchor oracle |
|---|---|---|
| T1 | one discounted cashflow, end to end | `N·e^{−rt}`, closed form |
| T2 | ZC inflation swap (model-free) | `N[Pr−Pn]`; par identity; pointwise 1.2.16 |
| T3 | inflation-linked bond | 1.2.5 closed form; strip reconciliation to T2 |
| T4 | YoY swap + Jarrow–Yildirim | `σr→0` reduction + non-vacuity companion |
| T5 | inflation cap/floor | Black reduction; `cap−floor = YoY` parity |
| T6 | JY vol calibration to the floor strip | parameter recovery + identifiability discriminators |

**v0.6.0 · 103 tests · eight gates green on ubuntu + windows · ruff + pyright clean · zero
suppressions.**

> **Verified by Code, 2026-11-01**, since Cowork cannot execute: the test count, the gate results,
> ruff/pyright, and zero suppressions all hold. Three factual claims in this record did **not**
> hold and are corrected in place below — `OPEN.md`, the `# crosscheck:` markers (§4), and the
> carried-field attribution (§5). The corrections are marked; nothing else was rewritten.

## 1. Oracle-quality audit — accepted
Every tranche is anchored by a closed form, an exact reduction, or a structural requirement — **no
tranche rests on a self-consistency (par-reprice) alone**; where one appears it is explicitly kept
as a consistency guard with a non-telescoping anchor beside it. The cross-product oracles (T3's
strip, T5's parity, T6's cross-stage round trip) tie the products into one object and twice caught
a defect no within-product check could. T6's identifiability pair (multi-start uniqueness + Jacobian
conditioning) is the strongest single oracle the reset has produced: it excludes the degenerate
valley that a parameter-recovery round-trip passes on — demonstrated, not argued (a single-quote
strip fits "perfectly" at `σI` wrong by 3×).

## 2. The throughline — four defects, one shape → a charter clause
The vertical shipped or caught four defects, and they are one shape, now ruled into §4:
1. **T4 anchor** `ρ→0` — taken from the stock's approximation, not the source.
2. **v0.4.0 convexity** mis-nested — transcribed from one `-layout` rendering, twice.
3. **T6 calibration target** — the stock fits vols to a ZCIIS that carries none.
4. **T6's A4 premise** — checked on the quote maturities, where the curves agree by construction.

Each was **an oracle whose domain excluded the failure.** "The stock is not authority" and
"transcription independence" named three of the *sources*; the §4 addition names the *shape* and
subsumes them under the floor rule: widen the oracle's domain until it includes where the failure
can live, or the green is empty. Three of four came from treating the stock as authority, the
fourth from treating one rendering of the source as authority — the discipline held because
pre-registration and the cross-product oracles were in place.

## 3. Spine clarification → §2 amended
"L3 MODELS + CALIBRATION" misread as putting all calibration at L3. Ruled and spelled out in §2:
model-free calibration (a curve bootstrap) is L3; a calibrator that **inverts an engine** (JY to a
floor strip, calling `price()`) is an engine consumer and sits at L4 (`engine/jy_calibration.py`).
`acyclic` enforces it.

## 4. Challenge me — the honest scope of what is built
This is **JY inflation, at inception, ATM** — a coherent complete slice, with these *named* limits,
none hidden:
- **Seasoned** YoY / ZCIIS / linker (valuation after inception) is deferred — it needs an index
  **series** in the snapshot and a lag-resolution convention (the indexation lag is carried but not
  resolved). The vertical prices at `t=0`.
- **Market Model II** (the thesis's second model, ch 3–4) is not built — only Jarrow–Yildirim.
- **`σn`** is a named input, not fitted (no nominal-cap engine); the `(σr,σI)` fit is conditional on
  it.
- **No smile** (ATM only) — faithful to the thesis's own finding.
- **QuantLib** cross-check deferred — but **the `# crosscheck:` markers charter §8 requires do not
  exist**: `grep -rn '# crosscheck:' src/` returns nothing across all six tranches. *(Corrected by
  Code, 2026-11-01; the record claimed they were "placed at every site that wants it".)* So the
  mechanism meant to make the QuantLib tier complete **by construction** is empty, and the tier
  would have to be assembled from memory — the one thing the clause forbids. Logged as `OPEN.md`
  **OI-1** rather than silently fixed, because the choice between enforcing the clause (markers
  owed across five tranches, and a gate, or it drifts again) and narrowing it is a ruling, not a
  cleanup. Blocks nothing: the tier is outside the reproducibility contract and every merge gate.

## 5. Debt + smell scan
Zero suppressions in `src/` or `tests/`, confirmed by execution. No file over the field/arg limits
(`verify.py fields`).

**`OPEN.md` did not exist** at this checkpoint — the record said "empty". *(Corrected by Code,
2026-11-01.)* The `debt` gate tolerates its absence while suppressions are zero (0 − 0 = 0), so no
gate was violated, but the ledger charter §8 mandates was simply absent. It now exists, carrying
OI-1 above as its first entry.

The recurring smell — a carried-but-unread field — is **not** confined to stock code as this record
first stated. `InflationLinkedBond.indexation_lag` (`products/inflation.py:99`) is carried and
unread in our own tree. It is **permitted**, not debt: charter §6's test is
constitutive-of-identity, and two linkers on the same index with different lags are different
instruments, with the serialisation round-trip as its present consumer. Recorded accurately rather
than attributed elsewhere, because "every instance was in stock code" would have made the next
reader treat a legitimate carried field as a violation.

## 6. Next unit — Bernardo's call
The vertical is complete enough to stop or extend by interest. Natural continuations: the seasoned
case (index series + lag resolution), Market Model II, or a new asset class entirely. Interest
picks; the charter is the constant.
