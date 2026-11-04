# Checkpoint — the inflation thesis complete through calibration (T7–T11)

**Date:** 2026-11-03 · **By:** Cowork · **Baseline:** v0.13.0 · **Scope:** the cluster from seasoned
pricing through the full JY calibration. Second checkpoint of the reset repo (first: the T1–T6
`CP_inflation_vertical`).

## 0. Status — the thesis's own arc is built and oracled
BERHAN-2012, by chapter: **ch 1 complete** (all four instruments), **ch 2 §2.5/§2.6 complete** (JY
YoY + cap/floor), **ch 5 complete bar §5.4** (nominal + real curves §5.1, the simplified-JY/Black
reduction §5.2 + its surface inversion, the nominal-cap fit §5.3.1, and the full §5.3.2). Remaining
of the book: **ch 3 and ch 4 (the two market models) + §5.4**, and the non-thesis risk layer.

| tranche | capability |
|---|---|
| T7 | seasoned pricing (`t>0`) — the index series + lag resolution |
| T8 | the nominal par-swap bootstrap (§5.1) |
| T9 | the implied-vol surface inversion (§5.2); Black extracted to L3 |
| T10 | the Hull-White nominal cap + the `(κn, σn)` fit (§5.3.1) |
| T11 | the full six-parameter JY calibration (§5.3.2) |

**v0.13.0 · 366 tests · eight gates green on both OSes · ruff + pyright clean · zero suppressions.**

## 1. The two honest negatives — the library went one level below the source
§5.3.2 was built as a *negative-result* tranche, and the result is **stronger than the thesis's**:

- **JY cannot smile** — a Gaussian model, flat implied vol across strikes. The thesis names this.
- **Three of the six §5.3.2 parameters are structurally unidentifiable from YoY floors — at any
  strike, ever** — `(σI, ρrI, ρnI)`, because the index correlations enter a floor price only pooled
  with `σI` and `ρnI` is absent from the convexity. Proven from the formula and shown by a
  wrong-sign `ρnI` recovered at rms 8e-15 on flawless data. **The thesis does not name this**; it
  blames the skew alone. Reproducing a source *and finding where its own diagnosis stops short* is
  the honest capstone, and the best teaching result of the arc.

## 2. Oracle-quality tally — the discipline earned itself
The project's one recurring defect is **an oracle whose domain excludes the failure**, and it
appeared ~six times across this cluster (the stock as authority, one PDF rendering, a source
misprint, degenerate fixtures ×3, the identifiability valley, a noise-dominated condition number).
**Every one was caught** — by pre-registration, a cross-product oracle, a requirement oracle,
mutation, or a multi-start spread — never shipped to a user. §4 of the charter grew directly from
them and now carries: domain-contains-the-failure, degenerate-fixtures + mutation, the
noise-floor/behavioural-invariant rule, and multi-start-is-necessary-not-sufficient.

## 3. The two-surface split, measured
Cowork's review caught Code's weak discriminators (T6's identifiability companion, T9's strip count);
Code's execution caught ~five unverified Cowork claims (T11's three false empirical claims, the
stock-crash misattribution, the parity-settles-the-radical over-claim). It works in **both**
directions, which is the whole argument for it (§10).

## 4. Sources
`BERHAN-2012` (`worked-from`); `MCINERNEY-ZASTAWNIAK` (`in-hand`, unblocked the HW cap §5.3.1);
`CAPIŃSKI-ZASTAWNIAK` credit text (`in-hand`, for a future credit vertical). The references
discipline held: Code hard-stopped T10 rather than infer an un-held HW formula.

## 5. Debt + next
Zero suppressions; `OPEN.md` clean (OI-1 closed). **Next (Bernardo): finish the two market models.**
MM-I (ch 3) first — BERHAN ch 3 publishes its formulas in full (verified) — then MM-II (ch 4) + its
calibration §5.4, enabling the JY-vs-market-model comparison that is the thesis's actual purpose.
