# Re-audit #2 — findings (whole-tree, v0.35.0, T1–T29)

Charter §9's second periodic re-audit, fired at the **XVA arc close**. Run independently of the
builder (Code built all 29 tranches) from a read-only context (`/Users/bernardocohen/work/pricebook`,
reviewed source + a throwaway `/tmp` mutation copy). This file is the handoff; each finding is ruled
afterward by the design authority (Cowork), per §10 — nothing here is a patch.

## Environment / method

- Baseline reproduced on **Python 3.10.12, numpy 2.2.6 / scipy 1.15.3** (the host lacks a pinned
  3.12 interpreter — same posture as re-audit #1, and mild evidence the tolerance oracles are not
  over-fit to the pinned last digits).
- **Baseline: 990 passed (~16s); 9/9 gates green** — `verify.py all` (acyclic, fields, layers, debt,
  provenance, references, version) `ok`, plus `slow` `ok` and `tests` floored at 990 collected = 990.
- Mutation testing on a full-repo copy under `/tmp/pbcopy` that pytest actually imports
  (`pythonpath=["src"]` overrides `PYTHONPATH`), with `__pycache__` cleared before every run (the
  re-audit #1 stale-`.pyc` hazard). All survivor/caught verdicts below are from that copy.

---

## Target 1 — ORACLE POWER (targeted mutation): CLEAN, no holes

**~48 mutations**, prioritised by novelty (the XVA spine T25–T29 + T26b) and refactor-risk (older
oracles a later refactor may have weakened). Every behaviour-changing mutation was caught by a named
oracle; the sole survivor is proven equivalent by bit-identity.

### (a) The new XVA spine — thorough, all CAUGHT

**`risk/credit_valuation_adjustment.py` (T25/T27)** — EPE `E[(m+sZ)+]`: drop the `s·φ` optionality
term, drop the `m·Φ` weighting, flip the `m/s` ratio sign, and the `s=0` plain-positive-part limit
(`max→mean`); the integrand `DF·EPE·λS`: drop the `surviving` first-to-default factor, swap
`default_density→survival`, drop `DF`; trapezoid drop-½; `unilateral_cva_one_period` drop PD;
`BilateralAdjustment.bcva` sign (`CVA−DVA`→`CVA+DVA`); `bilateral_credit_adjustment` own/cpty curve
swap in the CVA leg — **all CAUGHT.**

> Note the charter's T27 concern is satisfied: dropping the `surviving` first-to-default factor
> (setting it to 1.0) is CAUGHT, so the bilateral oracle asserts that term's **magnitude**, not just
> its presence — the generalisation from T25's `surviving=None` into `credit_adjustment` kept its
> power.

**`risk/funding_valuation_adjustment.py` (T28)** — coefficient `γ/(1+γ)`→`γ`; residual drop `^+`;
residual reflow misread (`max(net−fva)`→`max(net)`); iteration drop `^+` — **all CAUGHT.** One
survivor, proven equivalent (below).

**`risk/collateral.py` (T29)** — gap drop `^+`; gap sign reverse `(V(t)−V(t−δ))`→`(V(t−δ)−V(t))`
(the T29 distributional-symmetry trap — CAUGHT, confirming the fixture carries material drift);
`previous` offset error (`V(t)−V(t)`); `initial_mark`→0; `δ=0` returns input not zeros — **all
CAUGHT.**

**`market/survival_curve.py` (T25)** — cumulative-hazard drop the segment `start` offset, `min(t,end)`→`end`;
density `hazard→1`; `S=exp(−Λ)`→`exp(+Λ)` — **all CAUGHT** (piecewise fixture read beyond the first
segment, as the module claims).

**`risk/mc_exposure.py` (T26)** — CVA weight `default_density→survival`; `positive`→raw values;
bilateral CVA-leg `own.survival`→`counterparty.survival`; DVA-leg drop first-to-default
(`counterparty.survival`→1); `cva−dva`→`cva+dva`; DVA uses `+values`; swap value `bond_first−bond_end`→`+`,
drop `strike·annuity`; `numeraire_relative /bond_end`→`/bond_first`; `live_exposure_times` drop the
`is_historical` filter; trapezoid-weight drop-half; `estimate()` ENE `−values`→`+values`; swap
annuity range `span.first`→`j`; swap bond product `(1−fτ)` sign — **all CAUGHT.**

**`foundation/time_measure.is_historical` (the atom routed through 5 engines + mc_exposure)** —
`pay <= valuation_date`→`pay <`: **CAUGHT** (6+ tests, matching C10's claim).

### (b) The signature defect — older oracles weakened by a later refactor: NONE FOUND

- The **T25 unilateral-CVA oracle after its T27 generalisation into `credit_adjustment`** retains its
  power: every unilateral-path mutation above is caught, and the `surviving=None` special-casing is
  the bit-identical structural reduction the code docstring claims.
- The **T26 exposure engine after T29's collateral extension**: all T26 mutations caught; the T29
  `agreement=None` path returns the input object (bit-identical), and the collateralized-gap mutations
  are independently caught.
- The **`is_historical` boundary across the five engines + mc_exposure** it now routes through: the
  single atom-level flip is caught; this is the C10 "divergence inexpressible" guarantee holding.

### (c) Oldest tranches (T2–T14 inflation, T15–T22 Jäckel) — light re-check, CLEAN

Spot mutations confirm no silent weakening since re-audit #1: LMM drop `/(1+fτ)` (T16), JY convexity
coeff `0.5→0.45` (T4), Black `d1` variance sign (T9), `norm_cdf` erfc sign (T20), linker drop
redemption (T3) — **all CAUGHT.** Also re-confirmed re-audit #1's **F-1.2 is closed**: dropping `ψ`
in the linker coupon (`inflation.py:136`) is now **CAUGHT** (T23's non-unit-`ψ` oracle bites).

### (d) The T26b `references` gate — its own power confirmed

Downgrading a cited `worked-from` key to `in-hand` → **gate FAILS**; pointing a `src/` `source:` at a
non-existent key → **gate FAILS**. The governance gate is powered, not decorative.

### The one survivor — proven equivalent, not a hole

`funding_valuation_adjustment.py:74` `if net <= 0.0:` → `if net < 0.0:` **SURVIVES** (203 L5/L1 tests
green). **Proven equivalent:** at `net == 0.0` the mutant falls through to
`γ/(1+γ) · net = γ/(1+γ) · 0.0 = 0.0` for every `γ ≥ 0` (verified bit-identical across
`γ ∈ {0, 0.3, 1, 2.5, 1e9}`), the identical value the `<=` early-return produces. Same equivalence
class as the `is_historical` expired-flow case (a boundary slip that is a real mispricing in one
engine and a proved-equivalent mutant where the quantity at the boundary is exactly zero). Not a hole.

**Verdict:** no surviving non-equivalent mutant. The XVA spine's documented traps (T29
distributional-symmetry, the FVA `^+` domain, the T27 first-to-default magnitude) are all defended.

---

## Target 2 — DUPLICATION & CONSISTENCY: CLEAN (1 pre-existing LOW stands, unchanged)

- **No duplicated value objects.** Every `class`/`@dataclass` name in `src/` is unique (no
  duplicates). The XVA arc added `ExposureProfile`, `SimulatedExposure`, `ExposureEstimate`,
  `CreditParties`, `BilateralAdjustment`, `CollateralAgreement`, `SwapValuesOnPaths` — each defined
  once.
- **The two trapezoid integrators are consistent AND guarded, not a defect.** The closed-form
  `credit_adjustment` (scalar reduction of a deterministic integrand) and the MC `mc_exposure`
  (per-path node weights, required for the pathwise SE — §3 invariant 7) are genuinely different
  shapes (§6: different shapes are *not* a shared shape, cf. re-audit #1 F-2.2). Drift between them is
  guarded by a cross-product oracle: `mc_exposure_cva`'s mean is asserted to reproduce T25's
  `unilateral_cva` (`tests/L5/test_mc_exposure.py` Leg A). No action.
- **F-2.1 (re-audit #1) still stands, unchanged.** The driftless-forward terminal
  `F·exp(−½V² + V·z)` is still inlined in `models/black_mc.py` and `risk/sensitivities.py._terminal`;
  the XVA arc added no third consumer and touched neither file, so the rule-of-two "extract at next
  touch" record in `to_go_deeper.md` (lines 87–89) is correct and current. No action.

---

## Target 3 — REGISTER ACCURACY: 1 LOW (rest clean & verified)

Clean and verified:
- **Suppressions = 0.** No `# type: ignore`, `# noqa`, `# pragma: no cover`, `@pytest.mark.skip/xfail`
  in `src/` or `tests/`. `verify.py debt` green; `OPEN.md`'s "Currently zero" holds.
- **`OPEN.md` ledger** — OI-1/OI-2/OI-3 all CLOSED and the code matches (norm_cdf is the `erfc` form;
  MC results carry `standard_error`; crosscheck markers present and gated). No open item misstated.
  *(Minor: the "verified at v0.6.0" parenthetical on the zero-suppressions line is a stale timestamp,
  but the `debt` gate enforces the invariant live — not worth a tranche.)*
- **`references.md` body is clean and the gate enforces it.** All six keys (`BERHAN-2012`,
  `JAECKEL-2002`, `MCINERNEY-ZASTAWNIAK`, `ACKLAM-2000`, `CREPEY-2026`, `CAPINSKI-ZASTAWNIAK`) are
  `worked-from` headers with real citers; no stale "in-hand until TX" promise survives in the body;
  `verify.py references` green and powered (Target 1d).
- **Deferrals genuinely absent, not silently built.** No KVA, Merton, CDS/bond survival calibration,
  initial-margin/MVA, threshold/MTA, or neural-net/regression XVA code anywhere in `src/` (the three
  "regression" grep hits are the English word in comments, not ML). Matches `to_go_deeper.md` /
  `out_of_method.md` / the arc plan.
- **Checkpoint `CP_xva_spine.md` accurate** against the code (T25–T29 + T26b, versions, layers, the
  six registered pricers, the method accretions). **`audit_log.md` accurate** (re-audit #2 DUE/running
  at v0.35.0 / T29, arc-close trigger, cadence re-arms ≈ T33).
- **C10 and C12 guard tests exist and bite.** `tests/L0/test_valuation_boundary.py` (C10, 5 tests);
  `test_two_curves_on_DIFFERENT_clocks_are_refused` in both CVA test files (C12). C11's claim of six
  snapshot-carrying models ↔ six registered pricers is accurate.

### F-3.1 [LOW] — `conventions_register.md` "Immediate reading" is pre-T26 stale and self-contradictory

**Location:** `docs/conventions_register.md`, the "## Immediate reading" section (lines 80–91).

**Evidence.** The summary prose was written by T24 (when C10 was still `per-module`) and was **not
updated** when T26 extracted the `is_historical` atom (upgrading C10 to `guarded — structurally`) and
added C12. Three concrete drifts, all in the summary while the rows themselves are correct and current:

1. Line 80: *"This tree has **five** live conventions (C7-C11)"* — there are now **six** (C7–C12); C12
   was added at T26 and sits in the live table (line 58).
2. Line 80–82: *"the one to watch is C10. It is the only row spanning five modules with **no shared
   atom** — each engine inlines `pay > valuation_date` — which is the exact standing the predecessor's
   discounting convention had the day before it broke."* — This directly **contradicts the C10 row it
   summarises** (line 56: *"guarded — structurally (tranche 26) — a divergence between engines is no
   longer expressible: there is one comparison"*). C10 is now among the best-guarded rows, not "the
   one to watch."
3. Lines 88–91: *"C10's upgrade path is exactly this: extract one `is_historical(pay,
   valuation_date)` and it joins them [C9/C11 as structurally guarded]."* — describes the extraction
   as still **pending**; it was done at T26. The whole paragraph reads as if the atom does not yet
   exist.

**Why it matters (and why only LOW).** §4's cross-module-convention clause leans on this register as
the enumeration of which conventions exist and how each is guarded. The *rows* (C7–C12) are accurate,
so a reader who reads the table is correctly informed; the harm is confined to the summary a reader
hits first, which paints C10 as the unguarded worry it stopped being at T26 and undercounts the live
rows. It is the re-audit's own purpose — a doc describing a practice the tree has moved past — but
docs-only, no code impact, rows correct. This is the T24-built register drifting under T26, the same
shape as re-audit #1's own F-3.2.

**Suggested disposition:** a small **docs fix-tranche** (or fold into the next docs touch): rewrite
"Immediate reading" to (a) say six live conventions C7–C12, (b) move C10 into the
structurally-guarded set beside C9/C11 (the extraction is done), and (c) name C12 (the CVA
shared-time-measure guard) and whichever live row is now genuinely least-guarded (C7/C8 remain OWED
but unviolable — one day count).

---

## Target 4 — CHARTER STALENESS: CLEAN

No stale, self-contradictory, or false-practice clause found. Falsifiable XVA-arc clauses
spot-checked against code and all hold:
- **§9 gate list** (`acyclic · layers · fields · debt · version · provenance · references · slow ·
  tests`) — all nine exist and are green; the `references` gate (§7/§9) is built and powered.
- **§3 invariant 7 "one engine-result shape"** — `mc_exposure_cva` and `mc_bilateral_adjustment`
  (T26) both return a `PricingResult` carrying `standard_error`; `MonteCarloEstimate` remains the L0
  primitive. The arc's two new MC engines did not fork the result shape.
- **§4 T25/T28/T29 additions** — the "reference is code too / convergence-rate" (T25), the FVA
  fixed-point-oracle shape (T28), and the distributional-symmetry degenerate fixture (T29) each match
  the code and their oracles (Target 1a confirms the T28/T29 traps are defended).
- **§1 three-gate / two-bin split** — `docs/to_go_deeper.md` and `docs/out_of_method.md` both exist
  and partition the Crépey deferrals as the clause describes; the deferrals are genuinely absent
  (Target 3).
- **§5 "our own docs are not an authority either"** and **§7 references gate + test-helper exemption**
  — descriptive of practice the code and registers follow.

*(The §4 prose cites dated per-tranche test counts — "986 tests", "all 404", etc. — these are
intentional historical anecdotes about when a mutant survived at that tranche, not claims about the
current 990-test suite; not staleness, as re-audit #1 already ruled.)*

---

## Summary table

| Target | Result | Count / severity |
|---|---|---|
| 1 Oracle power | CLEAN | 0 holes / ~48 mutations, 1 proven-equivalent survivor (FVA `net<=0` boundary) |
| 2 Duplication & consistency | CLEAN | F-2.1 (LOW) stands unchanged, no action; no new duplication |
| 3 Register accuracy | 1 finding | **F-3.1 [LOW]** — register "Immediate reading" pre-T26 stale/self-contradictory; rest clean & verified |
| 4 Charter staleness | CLEAN | 0 |

**Baseline:** 990 passed, 9/9 gates green (numpy 2.2.6 / scipy 1.15.3 — see Environment note).
**F-1.2 (re-audit #1) confirmed closed by T23.** The XVA spine is thoroughly oracled; the one new
finding is docs-only.

---

# Rulings (Cowork, §10) — 2026-11-10

Re-audit #2 is a **clean bill**: no oracle holes, and — the thing this re-audit most exists to catch —
**no silent weakening** from the XVA refactors (generalizing T25's unilateral CVA into
`credit_adjustment` at T27, extending T26's exposure engine with collateral at T29, routing five
engines through the `is_historical` atom) all held. Charter clean, F-1.2 confirmed closed by T23, and
the new `references` gate itself proven powered. The one survivor (FVA `net<=0`→`net<0`,
`γ/(1+γ)·0 = 0`) is a proven equivalent. A cleaner result than re-audit #1's two holes — the expected
shape: #1 swept the whole history for the first time and found the oldest instrument's blind spot; #2
found the new arc solid and the old oracles un-weakened.

## F-3.1 [LOW] — conventions-register "Immediate reading" summary was stale → fixed directly
The *rows* were correct; the T24-written summary prose was never updated under T26, so it still called
C10 the unguarded "one to watch (no shared atom)" and counted "five live conventions (C7-C11)".
Corrected in place (docs-only): six live conventions (C7–C12); C10 is now **guarded — structurally**
via the `is_historical` atom T26 extracted; C9/C10/C11/C12 all structural. **No fix-tranche** — a
one-paragraph prose correction, done as a Cowork docs edit this turn.

## Targets 1, 2, 4 — accepted clean.
Suppressions zero, `OPEN.md` all-closed, references/deferrals accurate, checkpoint and audit_log
correct. F-2.1 (the driftless-forward terminal, still one consumer short of rule-of-two) correctly
stands open in `to_go_deeper`.

**Remediation: COMPLETE** (the one LOW fixed in this turn). Re-audit #2 closed; the XVA arc is fully
closed.

---

## Single-point-of-failure oracles (ruled 2026-11-10, from Code's 0151 sample verification)

Code confirmed Target 1 by **executing** a 7-mutation sample weighted to the oldest tranches (where a
refactor-weakened oracle would hide); all seven die — confirming the pass by execution, not reading.
Two die to **exactly one test**: F-1.2's dropped-`ψ` (only T23's non-unit-`ψ` oracle) and T26's
numéraire-relative divisor (only Leg B). These are **single-point-of-failure oracles** — the
configuration most exposed to the signature defect one level down (a later *test* deletion reopens the
hole, suite still green).

**Ruling:** the periodic re-audit is the systemic backstop (a deleted killer resurfaces at the next
sweep), so recording is the floor and both are logged as the **highest-value re-run targets for
re-audit #3**. **F-1.2 — an already-shipped mistake, the thinnest margin — gets a second independent
killer** (test-only, Code to add). **T26's Leg B stays recorded without a forced second killer** — it
is the purpose-built independent anchor for that quantity, so a single killer there is by design.
§9 edited to make single-point-of-failure flagging a standing sweep output.
