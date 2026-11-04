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


---

## 7. Result — the book is closed, and MM-II's calibration has a flaw the thesis does not name

453 tests, eight gates green, ruff and pyright clean, zero suppressions. Four mutations, three
caught; the survivor is analysed in §9.3.

### §5.4.1 is an identity, and §5.4.2's bootstrap is structural
`σn,i` **is** the quoted Black ATM cap vol — oracle 1 is an exact round-trip, not a residual. And
the 1-Year-Floor bootstrap works for a reason this library can state: the first period's lead is
zero, so `Di` vanishes (it carries the lead as a factor) and `V²` collapses to `σ_I,1²·(T₁−T₀)`. One
unknown, one price, no solver — a plain Black inversion reusing tranche 9. The oracle asserts that
structure *before* relying on it, and separately proves the bootstrap blind to every later parameter.

### The thesis's expiry claim is ALMOST right, and the exception is the finding
*"Since each expiry has it's own set of parameters, there is no need to restrict the number of
contracts in the expiry dimension."* Measured the tranche 11 way — truth among the starts,
distance-from-truth beside spread — **every index vol recovers to ~1e-16 except the terminal one**,
which from a low start lands at **0.002141 against a truth of 0.020000: 89% wrong at a residual of
8.6e-17.**

**The mechanism, measured not asserted.** `σ_I,i` enters period `i`'s variance *quadratically* and
its drift *linearly with a negative sign*, so the caplet price is **non-monotone** in it — U-shaped,
vertex at `ρ_II·σ_I,i−1·lead/(lead+span)` (predicted ≈ 0.0115, located by sweep at 0.0115). Two
volatilities reproduce one price. Every vol except the terminal one appears in **two** periods — as
period `i`'s pay vol and period `i+1`'s start vol — and the second appearance picks the root; the
terminal date appears once, so nothing breaks the tie.

| `σ_I(T₅)` | caplet price |
|---|---|
| 0.0015 | 6.5186e-03 |
| 0.0115 (vertex) | **5.5951e-03** (minimum) |
| 0.0200 (truth) | 6.4106e-03 |
| 0.0021 | 6.4107e-03 ← the conjugate root |

**And the practical statement is sharper than a recommendation to quote further out.** Whether a fit
*lands* on the wrong root is **start-dependent**: a high start recovers the truth, a low one falls
into the conjugate, and **both report a residual of ~1e-17**. So a single-start calibration of MM-II
is unsafe at the terminal point and no residual will tell you. Multi-start is not a diagnostic
nicety here; it is the only thing between the fit and a silently wrong terminal volatility.

### The cross-model comparison the whole arc was for
Tranche 13 measured the *pricing* gap between the three models at 7.7e-09 to 3.7e-05 — negligible.
The difference that matters is what a calibration can tell you:

| | free parameters | unidentifiable | failure mode |
|---|---|---|---|
| **JY** (§5.3.2, T11) | six, **global** | **three** (`σ_I`, `ρ_rI`, `ρ_nI`) | structural, pervasive; `ρ_nI` recovers at the **wrong sign** at rms 8e-15 |
| **MM-II** (§5.4.2, here) | per-expiry | **one** (the terminal point) | local, and a two-root ambiguity rather than a flat valley |

So the thesis's verdict — *"of the models evaluated, this approach seems the most promising"* —
holds, and the library can now say **why** in a way the thesis does not: not better prices, but a
calibration whose parameters are mostly recoverable, with one named exception at the end of the strip
instead of three unidentifiable parameters throughout.

Both models remain unable to fit the smile, and MM-II reproduces the source's **signed** direction:
fitted at ATM, it **underprices OTM and overprices ITM** floors, exactly as §5.4.2 states — with the
companion that its ATM fit itself is essentially exact, so the two axes are distinct.

## 8. Two of my own errors, both caught by execution

### 8.1 I assumed tranche 13's recursion depth transferred
T13 measured the `index_vol_step` refusal at **two** steps on *annual* periods. This tranche's
fixture is semi-annual, so `τ` is halved, each forward is smaller and the per-step adjustment is
smaller: it survives **three**. The oracle now walks until refusal and bounds the count, rather than
hardcoding a number that belongs to a different fixture.

### 8.2 My "quote one period further" recommendation was wrong
I wrote an oracle asserting that truncating the strip hands the ambiguity to the new terminal point.
It does not — the ambiguity **disappears**:

```
strip of 5:  0.00%  0.00%  0.00%  0.00%  89.29%
strip of 4:  0.00%  0.00%  0.00%  0.00%
strip of 3:  0.00%  0.00%  0.00%
```

Whether the conjugate root is *reachable from a given start* depends on the period's own lead and its
neighbour's vol, so it is a property of the point and its parameters, not of being last. Replaced
with the start-dependence finding above, which is both true and more useful.

## 9. Findings for Cowork (§10 — raised, not decided)

### 9.1 The identifiability verdict, as you asked for it rather than presupposed
One parameter ambiguous (terminal, two-root, start-dependent), not the four-per-expiry
under-determination the spec's §1 flagged as a *prior*. The prior was reasonable and wrong: the
recursion link plus each vol's double appearance does most of the identifying work. Yours to rule.

### 9.2 `σn` stays a scalar while the index vols became a term structure
§5.4.1 indexes `σn,i` by maturity too, but it is **read** not fitted, so one value per model instance
is the natural granularity for a single-expiry fit. A per-maturity ATM cap curve arrives with a
consumer that fits across maturities. Flagged because it is an asymmetry a reader will notice.

### 9.3 A surviving mutant that is mathematically equivalent but NOT bit-identical — a §4 gap
Reversing the term structure's key assignment survived all 453 tests. It is **provably
unobservable**: `_params_from` applies the reversal *both* inside the residual and at the final
assembly, so the solver optimises a relabelled coordinate system and the output un-relabels it. No
test can distinguish them, which is the definition of equivalent.

**But it is not bit-identical** — the fitted vols differ by ~1e-15 (`0.011999999999999365` vs
`0.012000000000000018`), because relabelling the free variables perturbs the solver's iteration
path. §4's clause requires the equivalence exception to be *"proved by showing the outputs
bit-identical"*, and that bar is unavailable here.

So the clause has a gap worth closing: **bit-identity is the right proof for a mutation of a pure
function, but an iterative solver's output can differ in the last digits under a change that
provably cannot alter the answer.** For code behind a solver the proof should be the *structural*
argument (the mutation is a consistent relabelling, readable from the code) plus agreement to the
solver's own tolerance — not bit-identity. Raised rather than decided, and I have not used it to
wave anything through: the structural argument here is airtight and stated.

---

## 10. Cowork's ruling on T14 — PR #20 (§10 step 4, 2026-11-04)

Reviewed against the code, not only the narrative. Walked `_params_from` myself: the mutation is the
reversal `P`, `residuals_mutant(x) == residuals_orig(P(x))` holds bit-identically, so the fitted dict
is provably identical and the ~1e-15 is pure solver path. The checkpoint ties out against every
tranche. Accepted — v0.17.0, the book closed.

**9.3 (the §4 gap) — ruled, and kept as a bit-identity proof rather than loosened.** Your structural
argument is correct, but §4's bar exists precisely to refuse "looks equivalent" readings, so I did
not replace it with "structural argument + tolerance." Instead the proof **relocates to the
objective**: for solver-backed code, bit-identity is demanded of the residual function (a pure
function) under the mutation paired with its reparameterization — `residuals_mutant(x) ==
residuals_orig(P(x))` exactly — from which the fitted output is provably identical and its tolerance
agreement is corroboration, not proof. That is still a theorem-level bit-identity check, only moved to
where it is available. Charter §4's equivalent-mutant clause now carries this with your T14 precedent.
Your instinct was right and you did not wave anything through; the rule is now explicit so the next
one cannot be.

**9.1 — ruled: accept the measured verdict, and my prior was wrong, which is the point.** One terminal
two-root ambiguity, start-dependent, invisible to any residual — not the four-per-expiry
under-determination I flagged as a *prior*. The recursion link plus each vol's double appearance does
the identifying work I under-rated. The non-monotone mechanism (vertex at `ρ_II·σ_{I,i−1}·lead/(lead+
span)`, predicted ≈ 0.0115, sweep-located 0.0115) is a clean independent oracle. **The practical
finding is the keeper: a single-start MM-II calibration is unsafe at the terminal point and no
residual will say so — multi-start is load-bearing.** This is the fourth honest negative and belongs
in the Substack spine exactly as the checkpoint states it.

**9.2 — accepted, no change.** `σn` scalar while the index vols are a term structure is the right
granularity: §5.4.1 *reads* σn from the ATM cap quote, it is not fitted, so one value per model
instance is correct until a consumer fits across the maturity cap curve. The asymmetry is honest and
flagged; not debt.

**The checkpoint — reviewed and endorsed as the book's closing record.** Accurate chapter by chapter,
the four negatives and six source defects are exactly right, and §4's "honest scope of what is built"
is the part that keeps the whole arc truthful. This is the Substack spine. **Next unit is Bernardo's
call** (§1.3 risk layer / a smile model / the QuantLib tier / the credit text) — not mine to name.
