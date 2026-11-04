# Tranche 14 — Market Model II calibration (§5.4), the unit that closes the book

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.17.0`.
**Source:** `BERHAN-2012` §5.4 (`worked-from`) — read from the held PDF: §5.4.1 (σn from the quoted
Black ATM cap vol, automatic) and §5.4.2 (the iterative per-expiry fit of the index-vol and
correlation parameters to YoY-floor quotes). **Chosen by Bernardo:** finish the two market models and
their calibration. T14 completes BERHAN ch 1–5; only §1.3's risk layer (L5, orthogonal) remains of
the book.

## 0. What this is, and the one thing it must not do

MM-II's parameters, fitted to market quotes the way §5.4 prescribes. Two stages:

- **§5.4.1 — σn is not optimised, it is read.** MM-II models `Fn(·,Ti)` lognormally, so the quoted
  Black ATM cap volatility `σᵢᴬᵀᴹ` *is* `σn,i`. This is an identity, not a fit — the same shape as
  T9's Black-vol inversion, and its oracle is an exact round-trip (a Black quote in gives that σn
  back), not a least-squares residual.
- **§5.4.2 — the remaining per-expiry set `{σI,i−1, σI,i, ρI,i, ρI,n,i}` fitted iteratively to
  YoY-floor quotes.** `σI,1` comes directly from the 1-year floor (it depends on no other unknown);
  then least-squares fits the rest expiry by expiry, the expiries linked by the σ_I recursion T13
  built (`index_vol_step`).

**The one thing it must not do: assert an identifiability verdict I have not measured.** T11 is the
scar here — I wrote three false empirical claims into its first spec. So this spec names the
structural facts readable from the formula and the *oracles to run*; it does **not** claim how many
of the four per-expiry parameters are identifiable, which way a correlation lands, or whether MM-II
calibrates better than JY. Those are findings for Code's execution to report, and for me to rule on
after.

## 1. The structural facts (readable now) and the open questions (for execution)

**Readable from the source and the T13 code:**
- σn,i is the Black ATM cap vol by construction (§5.4.1) — an identity.
- The expiries are **linked** by `index_vol_step` (σI,i−1 ← σI,i), and T13 established that this
  recursion **drives σ_I negative in ~2 steps at realistic rate vols** and `index_vol_step` refuses a
  non-positive result. So a §5.4.2 fit walking the recursion across maturities **can hit that
  refusal** — a concrete, source-grounded failure mode, not hypothetical.
- Per expiry there are **four** free parameters `{σI,i−1, σI,i, ρI,i, ρI,n,i}` against essentially
  one ATM floor price at that expiry (plus the recursion link). That is a strong *prior* toward
  per-expiry under-determination — but it is a prior, **asserted as a question, not a result**.

**Open, to be settled by execution (the T11 method, mandatory here):**
- Identifiability of the four per-expiry parameters on **perfect ATM synthetic data** — measured by a
  **multi-start spread with the synthetic truth in the start set**, and **distance-from-truth beside
  spread** (spread shows non-uniqueness, distance shows wrongness; T11 §8.1). No condition number.
- Whether the skew failure is the *same* as JY's §5.3.2 (the thesis says it is — ATM-only recovers,
  OTM underpriced / ITM overpriced) **and** whether, as in T11, the source's diagnosis stops short of
  a structural under-determination present already at ATM.

## 2. Pieces and layer

```
L4  engine/mmii_calibration.py — inverts the YoY-floor pricer (reprices through price()) to fit the
                                 §5.4.2 parameters; consumes MM-II + the engine → L4 (as JY's
                                 calibrators are). The free set is MM-II's own parameter shape
                                 (MMIIParams widened per expiry), ≤5 fields per object.
L3? σn-from-Black (§5.4.1) is a model-free analytic identity. If it is a thin reused step it may sit
    in models/ (L3) like T9's Black inversion; if it lives inside the inverting module it is L4 with
    it. Settle by what it READS (§2's rule), and raise it if the split is unobvious.
```

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **σn = the Black ATM cap vol, exact round-trip** (§5.4.1): feed a synthetic Black quote, recover that σn to tolerance | **exact / identity** | a mis-wired §5.4.1 that treats σn as a fitted rather than a read quantity |
| 2 | **σI,1 from the 1-Year floor alone** (§5.4.2's bootstrap start): on synthetic data where σI,1 is known, recover it with no other parameter in play | 2 — round trip | the bootstrap seam reading more than the 1Y floor |
| 3 | **identifiability of `{σI,i−1, σI,i, ρI,i, ρI,n,i}` — the T11 behavioural method, NOT a condition number:** multi-start spread with the truth among the starts, and distance-from-truth asserted beside spread | **1 — structural/behavioural** | whatever the real degeneracy is; this oracle REPORTS it rather than presupposing it |
| 4 | **the recursion-refusal is reachable:** a realistic rate-vol fixture drives `index_vol_step` to its non-positive refusal during the walk (T13's negative-σ_I, now inside the calibration) | structural | a calibration that silently produces a `nan`-bearing σ_I instead of failing as a value (invariant 4) |
| 5 | **the skew failure, asserted as its own axis** (as T11 oracle 4 did): ATM recovers; a skewed surface leaves irreducible OTM/ITM error; and whether an at-ATM under-determination exists *before* any skew is measured and stated | structural | conflating the two failures, or reproducing the source's framing without checking past it |
| 6 | **failure-is-a-value** (invariant 4): non-convergence / a non-positive σ_I → `PricingFailure` or a named raise the shell turns into one; never a silent `NaN` | structural | the solver seam's `converged` flag and `index_vol_step`'s refusal |
| 7 | **cross-model, the thesis's actual point:** on the SAME ATM floor quotes, calibrate JY (T11) and MM-II and report how the fits compare — identifiability and parameter honesty, not price (T13 found the price gap is 7.7e-9–3.7e-5). Stated as a measured comparison, no number fitted to Fig 5.10 | 2 — comparison | — (this is the deliverable, not a pass/fail; it is where the book's purpose lands) |

Oracles 3 and 5 **report** the identifiability and skew outcomes; the spec does not pre-judge them.
Oracle 7 is the comparison the whole inflation arc was for. **No number fitted to a figure** — Fig
5.10 is a plot; the comparison comes from the library's own two calibrators on shared synthetic data.

## 4. Design decisions (ruled)

- **σn is read, not fitted** (§5.4.1) — oracle 1 is an identity round-trip, not a residual.
- **Per-expiry fit, iterative, σI,1 bootstrapped first** (§5.4.2's own procedure) — do not restrict
  the expiry dimension (the source is explicit each expiry has its own set); the strike dimension is
  where the skew failure lives.
- **The identifiability verdict is a finding, not a spec claim** (the T11 scar). Oracle 3 runs the
  T11 behavioural method; whatever it shows is Code's to report and mine to rule.
- **Inherit T13's recursion guard, don't re-derive it** — `index_vol_step` already refuses a
  non-positive σ_I; the calibration must surface that as a value (oracle 4/6), not swallow it.
- **Cross-check disposition:** `engine/mmii_calibration.py` → `crosscheck: quantlib none — QuantLib
  has no inflation forward-CPI market-model calibrator` (as T13's model carried; the independent side
  is the synthetic-truth round-trip).

## 5. Stock
Mine `_stock/.../calibrat*`/`jy_calibrate` for the surface-fit objective's *shape* only; read the
procedure (§5.4.1 identity, §5.4.2 iterative bootstrap) and every formula from **BERHAN §5.4**, not
from the stock's code.

## 6. Process
Branch `tranche/14-mmii-calibration`. `test` (RED **pre-registered** — the T10a miss) → `feat` →
`docs` → `chore(release): v0.17.0`. Mutation on the per-expiry objective and on the bootstrap seam;
oracle 3 uses a **skewed** surface fixture for the skew axis and a non-degenerate strip (the T11
speed-vs-degeneracy lesson: do not thin the strip to make `slow` pass — drop starts instead).
`source:` cites `BERHAN-2012` §5.4. **Open as a DRAFT PR** (§10 step 3). This closes the book — a
full-arc checkpoint (`CP_inflation_book_complete`, ch 1–5) is written at this tranche's review.
