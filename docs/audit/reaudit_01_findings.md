# Re-audit #1 — findings (whole-tree, v0.27.0, T1–T22)

Charter §9's first periodic re-audit, at the Jäckel arc close. Run independently of the builder
(Code built all 22 tranches) from a read-only context. This file is the handoff; each finding is
ruled afterward by the design authority (Cowork), per §10 — nothing here is a patch.

## Environment note (how the baseline was reproduced)

The repo pins `numpy==2.4.3` / `scipy==1.17.1` on Python 3.12 (reproducibility contract, §9). The
audit host had no network access to a 3.12 interpreter, so the suite was run on **Python 3.10 with
numpy 2.2.6 / scipy 1.15.3**. All 818 tests passed and all 8 gates were green on that stack, which
is itself mild evidence the tolerance-based oracles are not over-fit to the pinned last digits (§9's
"transcendental results differ 1–2 ULP across runtimes"). Mutation testing was done on a full-repo
copy under `/tmp` (never the repo), because pytest's `pythonpath = ["src"]` ini option overrides a
`PYTHONPATH` env var — a mutated file placed only on `PYTHONPATH` is silently ignored, which would
have produced false "survivor" results. All results below are from mutating the file inside the copy
that pytest actually imports.

## Baseline

- `pytest -q`: **818 passed** (~14s).
- Gates: `acyclic` · `layers` · `fields` · `debt` · `provenance` · `version` · `slow` — all `ok`
  (`tests` floored and green via the run above). **8/8 green.**

---

## Target 1 — ORACLE POWER (targeted mutation): CLEAN, no holes

26 concrete mutations across every load-bearing formula the §9 brief names, plus Hull-White. Every
behaviour-changing mutation was caught by a named oracle; the only survivors are provably equivalent
or explicitly documented as such.

Domain examined, mutation → result:

**LMM drift (`models/libor_market_model.py`, eq 12.8)**
- Drop `/(1+fτ)` denominator in `state_ratios` → **CAUGHT** (15 fail, incl. pointwise 12.8, self-term, non-uniform ρ).
- Sign flip in the up-branch → **CAUGHT** (10 fail).
- Span offset `range(i+1,N)`→`range(i,N)` → **CAUGHT** (9 fail).
- ρ transpose `matrix[i][k]`→`matrix[k][i]` → survives — **equivalent mutant** (matrix is enforced symmetric; the test even asserts symmetry). Not a hole.

**Rebonato swap-rate vol (`models/swap_rate_vol.py`, eq 5.40)**
- Drop span offset in ρ `[first+m][first+l]`→`[m][l]` → **CAUGHT** (the translation-invariance trap is explicitly guarded with a time-decaying ρ on non-uniform accruals).
- ρ transpose → survives — **equivalent** (symmetric).
- Drop span offset on forward in `Z` → **CAUGHT**.

**Acklam inverse-CDF + `norm_cdf` (`foundation/distributions.py`)**
- `norm_cdf` sign flip `erfc(-x/√2)`→`erfc(x/√2)` → **CAUGHT** (28 fail).
- Perturb an Acklam `_A` coefficient by ~1e-3 → **CAUGHT** (the published-accuracy oracle).
- Drop the Halley third-order denominator (→ Newton) in the production path → survives — **documented equivalent mutant**: `test_T20_HALLEYS_THIRD_ORDER_TERM_is_NOT_measurable_from_Acklams_start` asserts the two agree to machine precision from Acklam's start, and a dedicated companion exercises the denominator from a degraded start. Not a hole.

**Sobol (`foundation/sobol.py`)**
- Corrupt one direction integer in Table 8.3 (dim 2, `3`→`5`) → **CAUGHT** (property-A oracle).
- Drop `>> degree` in the eq-8.19 recurrence → **CAUGHT** (pointwise recurrence).
- Gray code `n ^ (n>>1)`→`n + (n>>1)` → **CAUGHT**.

**Brownian bridge (`foundation/brownian_bridge.py`)**
- Swap the bracket weights (`1-weight`↔`weight`) → **CAUGHT** (`AAᵀ=min(tᵢ,tⱼ)`).
- Drop the `(right-target)` factor in `conditional_variance` (eq 10.34) → **CAUGHT**.
- Drop `√` on the terminal (`√horizon`→`horizon`) → **CAUGHT**.

**Black + analytic Greeks (`models/black.py`)**
- `d₁` variance sign `+½V²`→`-½V²` → **CAUGHT** (42 fail across black/cap/caplet).
- `_d_plus` sign flip → **CAUGHT**.
- `black_vega` drop `forward` factor → **CAUGHT**.
- `black_gamma` drop `root` in denominator → **CAUGHT**.

**MC Greeks (`risk/sensitivities.py`, eqs 11.30–11.32)**
- LR gamma-weight sign (`z²-zσ√Δt-1`→`z²+zσ√Δt-1`) → **CAUGHT**.
- LR vega-weight sign (`(z²-1)/σ - z√T`→`+ z√T`) → **CAUGHT**.
- Pathwise vega sign (`√T·z - σT`→`+ σT`) → **CAUGHT**.
- LR delta-weight drop `root` → **CAUGHT**.

**Hull-White caplet (`models/hull_white.py`, eqs 3.19/3.20)**
- `bond_option_variance` drop `β²` → **CAUGHT** (38 fail).
- `hw_beta` sign `(maturity-t)`→`(t-maturity)` → **CAUGHT**.

**Verdict:** no surviving non-equivalent mutant found. The translation-invariance trap the charter
flags (§4, decaying ρ) is correctly defended in both the LMM and swap-rate-vol fixtures. The two
equivalent survivors (ρ transpose under a symmetric matrix; Halley→Newton at Acklam's start) are
genuinely equivalent and already documented.

*(Not individually re-mutated this pass, oracle read instead: the JY convexity, MM-I/MM-II drifts,
the nominal par-swap bootstrap, the exercise-boundary regression. Load-bearing but outside the §9
named set; their oracle structure matches the patterns verified above. A future pass could extend
the sweep there.)*

---

## Target 2 — DUPLICATION & CONSISTENCY: 2 low-severity items

No duplicated value objects (every `class` name in `src/` is unique; `MonteCarloConfig`,
`MonteCarloEstimate`, `Money`, `Term`, `PathGrid` each defined once). The two MC LMM engines
(`swaption_mc`, `bermudan_swaption_mc`) correctly **share** `engine/mc_path_engine` rather than
re-implementing path generation — the right outcome.

### F-2.1 [LOW] — the driftless-forward terminal is inlined in two modules
`models/black_mc.py:118` computes `forward * exp(drift + volatility·z)` (drift `= -½V²`) and
`risk/sensitivities.py:130` (`_terminal`) computes `forward * exp(-½V² + root·z)` — the same
"forward under its own measure" shape, inlined independently. `sensitivities` already imports
`MonteCarloConfig` from `black_mc`, so the dependency edge exists. Rule-of-two (§6) has a second
consumer sharing the *shape*; a shared `_terminal` helper would remove the risk of the convention
drifting between them. **Disposition:** `to_go_deeper` / a small consolidation when a third consumer
appears — not urgent (both are one-liners and currently identical).

### F-2.2 [LOW] — discount-factors-from-forwards recurrence in three places
`B_{k+1} = B_k/(1+fₖτₖ)` appears in `models/swap_rate_vol.py:84` (`tenor_discount_factors`, static
tuple), `engine/bermudan_swaption_mc.py:215` (scalar path loop) and `engine/swaption_mc.py:133`
(`np.cumprod`, per-path array). The **convention is consistent** across all three (no sign/ordering
divergence found), so this is repetition, not inconsistency, and the three operate on genuinely
different shapes (static vs scalar-path vs vectorised), which §6 says is *not* a shared shape.
Recorded for visibility only. **Disposition:** none required; note in `to_go_deeper` if a fourth
appears.

### Consistency note (not a defect)
Invariant-6 (historical flows → PV 0) is live across **5 real engines** (`cash`, `inflation`,
`nominal_cap`, `capfloor`, `yoy`) and is *not* represented by any current register row — see F-3.1,
because the register's invariant-6 row (C3) names engines that do not exist in this tree.

---

## Target 3 — REGISTER ACCURACY: 1 med + 1 low; rest clean

Clean and verified:
- **Suppressions = 0.** No `# type: ignore`, `# noqa`, `# pragma: no cover`, `@pytest.mark.skip/xfail`
  anywhere in `src/` or `tests/`. `verify.py debt` green. `OPEN.md`'s "Currently zero" is accurate.
- **`OPEN.md` ledger.** OI-1, OI-2, OI-3 all marked CLOSED and the code matches: OI-3's `norm_cdf`
  is the `erfc` form; OI-2's MC results carry a standard error; OI-1's crosscheck markers are present
  and gated (see below). No open items misstated.
- **Provenance `source:` keys all resolve.** The only real reference keys cited in `src/` are
  `BERHAN-2012`, `JAECKEL-2002`, `MCINERNEY-ZASTAWNIAK`, `ACKLAM-2000` — all four are
  `in-hand`/`worked-from` headers in `docs/references.md`. No citation-by-inference.
- **Crosscheck markers complete & gated.** Every module under `market/`/`models/`/`engine/` carries
  a `crosscheck:` line (real target or `crosscheck: none — <reason>`); `verify.py provenance` green.
- **QuantLib tier genuinely unbuilt.** No `import QuantLib`/`ql.` anywhere in `src/` or `tests/` —
  consistent with "deferred to the optional QuantLib tier" everywhere it is claimed.
- **`docs/audit_log.md`** correctly shows re-audit #1 DUE/running at v0.27.0 / T22.
- **`docs/jaeckel_completion_arc.md`** chapter-status table matches what is built (Sobol T20, Greeks
  T22, etc.).

### F-3.1 [MED] — `conventions_register.md` rows C1–C6 describe a tree that does not exist
`docs/conventions_register.md` presents C1–C9 as "the register" for this tree, and its **Immediate
reading** section calls C2 *"the widest-spread convention in the tree ... the one to worry about."*
But C1–C6 reference modules and tests that are **absent** from `src/pricebook/` and `tests/`:

- C1: `risk/exposure.py`, `risk/wwr.py`, `risk/xva_report.py`, `risk/xva.py`, `tests/L5/test_equity_mc_exposure.py` — none exist.
- C2: `models/black_model.py`, `models/sabr_model.py`, `models/process.py`, `calibration/sabr_calibrate.py`, `engine/vanilla_option.py`, `engine/equity.py`, `risk/mtm.py` — none exist (there is no `calibration/` dir at all).
- C3: `engine/linear.py`, `credit.py`, `xccy.py`, `vanilla_option.py`, `seasoned.py`, `tests/L4/test_current_period.py` — none exist.
- C4/C5: `market/building_blocks.py`, `models/process.py`, `engine/equity.py`, `tests/L4/test_equity_vanilla.py` — none exist.
- C6: `risk/xva.py` — does not exist.

Evidence: these were carried **wholesale from the predecessor at the scaffold commit** (`236c184
"chore: clean project scaffold (v0.0.0)"` already contains C1–C6); only C7, C8, C9 were added by this
tree's own tranches (commits for T3 and C9). The reset ruling (`handoffs/rulings_reset_to_fresh_repo.md`
§, "`conventions_register.md` is a keep-artifact") deliberately kept the file for its *lessons* — but
the rows now read as live inventory of a tree that was archived.

**Why it matters:** §4's cross-module-convention clause relies on this register to *enumerate which
conventions exist*. A reader auditing cross-module conventions against it today is checking a
non-existent (XVA/equity/SABR) tree, while this tree's live conventions are only partly covered: C7/C8
(inflation day-count, correctly OWED + "unrepresentable, one day count" — verified: only `ACT/365F`
exists) and C9 (index resolution, structurally guarded) are real and accurate, but invariant-6 across
the 5 real engines (F-2.2 note) and the `model.market`/snapshot discounting convention have no row.

**Disposition:** a docs fix-tranche: either partition the file into "predecessor lessons (archived
tree)" vs "this tree's live register", or prune C1–C6 to the lesson and re-derive the rows that apply
to the current modules. The "Immediate reading" section needs rewriting either way — it currently
asserts a worry about modules that aren't here.

### F-3.2 [LOW] — `to_go_deeper.md` BERHAN section has two stale lines
- Item 6: *"Six `crosscheck:` markers already name the sites"* — there are now **23** real
  (non-`none`) crosscheck targets across the tree (the BERHAN + Jäckel arcs), so "six" is stale.
- Item 1: *"L5 is empty — no deltas, no scenario engine."* L5 is **not** empty since T22
  (`risk/sensitivities.py` exists with three Greek estimators). The *specific* route described (an
  inflation risk layer over the existing engines via a `Priceable` protocol, bump-and-reprice) is
  genuinely still unbuilt — T22's L5 is Black/forward Greeks (Jäckel ch 11), not that layer — so the
  route survives, but the blanket "L5 is empty" sentence is now false.

**Disposition:** `to_go_deeper` edit — update the count and re-word item 1 to name what L5 holds vs.
what the route still wants.

---

## Target 4 — CHARTER STALENESS: CLEAN (claims spot-checked against code)

No stale, self-contradictory, or false-practice clause found this pass. The §4 prose cites dated
per-tranche test counts ("all 404 tests", "550", "603", "468", "312") — these are intentionally
historical anecdotes about when a mutant survived *at that tranche*, not claims about the current
suite, so they are not staleness.

Concrete falsifiable claims verified against code:
- §3 invariant 7: `MonteCarloEstimate(mean, standard_error, n_samples)` is the L0 primitive;
  `mc_caplet_price` (`engine/mc_path_engine.py:350`) returns it raw and is **not** `@register`ed
  (confirmed — no Caplet product); the Bermudan (`bermudan_swaption_mc.py:351`) and European swaption
  (`swaption_mc.py:108`) engines both return a `PricingResult` carrying `standard_error` — "one
  engine-result shape", as claimed.
- §2 calibration layer split: no `calibration/` dir; curve bootstrap lives in `market/`
  (`nominal_bootstrap.py`, L1), engine-inverting calibrations in `engine/` (L4,
  `jy_calibration.py`/`hw_cap_calibration.py`/…). `acyclic` + `layers` gates green.
- §8 crosscheck gating: present-with-exemption holds (every `market/`/`models/`/`engine/` module has
  a marker; `provenance` gate green).
- `@register` set is exactly {FixedCashflow, ZCIIS, InflationLinkedBond, NominalCap,
  InflationCapFloor, YoYInflationSwap} — the MC swaption/Bermudan engines are called directly (no
  registered product), consistent with §3's product-pull description.

*(Note: the one document that is out of step with the code is the conventions register, F-3.1 — but
it is a `docs/` living artifact, not a CLAUDE.md clause, so it is filed under Target 3.)*

---

## Summary table

| Target | Result | Count / severity |
|---|---|---|
| 1 Oracle power | CLEAN | 0 holes / 26 mutations, 2 documented-equivalent survivors |
| 2 Duplication & consistency | minor | 2 LOW |
| 3 Register accuracy | findings | 1 MED, 1 LOW (rest clean & verified) |
| 4 Charter staleness | CLEAN | 0 |

Baseline: 818 passed, 8/8 gates green (on numpy 2.2.6 / scipy 1.15.3 — see Environment note).

---

## Target 1 (cont.) — ORACLE POWER, inflation arc (T2–T14): 1 HOLE (MED)

The first pass (above) ran the targeted mutation sweep on the **Jäckel-arc** formulas (T15–T22) and
explicitly deferred the **inflation-arc** load-bearing formulas (T2–T14) — the parenthetical at the
end of Target 1 names them as "not individually re-mutated this pass, oracle read instead." This
section is that deferred half: an independent, second-reviewer mutation sweep of T2–T14, run on the
same `/tmp` full-repo copy that pytest actually imports (per the method note; `pythonpath = ["src"]`
overrides `PYTHONPATH`). Baseline re-confirmed **818 passed** on the same stack.

**One genuine hole found** (MED): the inflation-linked bond's coupon-accrual factor is untested, the
charter's own degenerate-fixture trap (§4, `τ = 1` on ACT/365F). Everything else — 50 mutations
across the four ch-1 instrument engines, both convexity/variance lineages, the three inflation models
and the four calibrations — was caught by a named oracle. The one non-caught survivor outside the
hole is a proven equivalent mutant.

**Scope note:** the exercise-boundary regression (`models/exercise_boundary.py`) is **tranche 18**,
the Bermudan swaption under the LMM — it belongs to the Jäckel arc, not the inflation arc, so it is
out of this section's scope (and was in the first pass's territory). Confirmed by its provenance
`slice:` header.

### F-1.2 [MED] — the inflation-linked-bond coupon accrual `ψᵢ` is dropped-able undetected

**Location:** `engine/inflation.py:135`, `price_inflation_linked_bond`:
`coupons = sum(bond.leg.coupon_rate * psi * real.df(pay) for pay, psi in future)`.

**Mutation (L1):** drop the `psi` factor → `bond.leg.coupon_rate * real.df(pay)`. **SURVIVES all 818
tests** (re-confirmed from a clean `.pyc` state, full suite and targeted). A **genuine hole**, not an
equivalent mutant: the engine does read `psi` in production (verified — the engine's PV matches a
hand `ψ`-weighted computation), so the term is load-bearing; only its oracles are blind to it.

**Why the oracles miss it — the `τ = 1` trap, §4.** Every linker coupon oracle runs on one fixture,
`DATES = (VAL + 365·k for k in range(6))` with `DayCountConvention.ACT_365_FIXED` (test file lines
41–44). Each period is exactly 365 days, so **every `ψᵢ = 365/365 = 1.0` exactly**, and `C·ψ = C`.
The three coupon oracles are all blind:
- `test_the_linker_matches_the_closed_form_with_accrual_fractions` (line 82) — despite the name, it
  computes `psi = (pay − start).days / 365.0`, which is `1.0` on these dates; "accrual fractions" is
  the name, not the fixture. This is precisely the self-deceiving oracle §4 warns of.
- `test_with_unit_accruals_...REDUCES_to_paper_eq_1_2_5` (line 104) — `ψ = 1` by explicit design.
- `test_the_linker_IS_a_strip_of_zero_coupon_inflation_bonds` (line 117) — reuses the same `(pay −
  start).days / 365.0 = 1.0` on the same `DATES`.

No linker test anywhere uses a non-annual / non-ACT365F schedule. (`test_the_schedule_day_count_is
_round_tripped_too`, line 229, only checks serialization, not the accrual's price effect.)

**Magnitude (proof of non-equivalence).** On a semi-annual 3% linker (182-day periods, `ψ ≈ 0.5`,
flat 1% real), the engine prices `1,058,729.13`; with `ψ` dropped the coupon leg roughly doubles and
PV becomes `1,147,417.60` — **+886.9 bp of notional, ~8.4% of price**. The defect scales with the
coupon and with how far `ψ` departs from 1 (any real linker that is not annual-ACT/365F: semi-annual,
30/360, ACT/360, ACT/ACT, or a stub period).

**Contrast — the same mutation is CAUGHT everywhere else it lives:** the YoY swap's inflation leg
(`yoy.py`, first-period and later-period `ψ`, and the fixed leg), the inflation cap/floor
(`capfloor.py` `accrual`), the nominal-bootstrap annuity (`swap_annuity`), the MM-II calibration
bootstrap (`black_value` accrual) and the HW-cap Black price (`black_cap_price` accrual) all have at
least one non-unit-accrual fixture and all 7 accrual mutations there failed. The linker is the lone
instrument whose accrual fixture is degenerate.

**Suggested fix (advisory):** add a linker oracle on a non-unit-accrual schedule (semi-annual, or an
ACT/360 annual so `ψ ≈ 1.0139`) asserting the `ψ`-weighted closed form, with the non-degeneracy
asserted as a premise (`assert any(ψ != 1.0)`), per §4's "assert the non-degeneracy as a premise"
and "prefer a mutation to a reading." Mirrors the T10 dropped-accrual remedy.

### Mutations that were CAUGHT (49), grouped

Each row is `mutation → CAUGHT (n tests failed)`. Tests run were the arc's L1/L3/L4 files.

**JY convexity `C`, eq 2.5.11 (`models/jarrow_yildirim.py`)** — all CAUGHT:
convexity-term sign `−0.5→+0.5` (2); drop the `− cross·β_n` term (1); the T4 mis-nesting (cross term
outside the `β_r(t,T₀)` factor) (3); drop the `σ_r` overall factor (8); coefficient `0.5→0.45` (2);
`cross` denominator `(κn+κr)→(κn−κr)` (1); drop `κr` in `1+κr·β_n` (1). The `σ_r→0` anchor and the
`C(t=Tᵢ₋₁)=0` requirement both bite.

**JY log-ratio variance `V²`, eq 2.6.4, eight terms (`jarrow_yildirim.py` + `jy_variance` test)** —
all 8 CAUGHT: drop `span` on the `σ_I²` term; sign flip on each of the three `ρ` cross terms
(`ρ_nr` β-term, `ρ_ni`, `ρ_ri`); drop factor-2 on the `ρ_nr` span term; drop `(span−β_n)` → `span`;
`1/(2κn)→1/κn` on the `σn²` term; `lead→span` in the `σn²` decay exponent. The independently
transcribed sweep catches the magnitude of every term — no §4 "property tolerates a dropped term"
survivor here.

**ZCIIS, eq 1.2.13 (`engine/inflation.py`)** — all CAUGHT: first-term `real→nominal` df (8); drop the
index ratio (2); base-index `ratio = I/I0 → I·I0` (8); fixed-leg exponent `term.years()→1` (3).

**Inflation-linked bond, eq 1.2.5 (`engine/inflation.py`)** — 3 of 4 CAUGHT: drop index ratio (1);
drop redemption (5); base-index `/→*` (5). **The 4th is F-1.2 (SURVIVES, the hole).**

**YoY swap, eqs 2.5.12/2.5.13 (`engine/yoy.py`)** — all 6 CAUGHT: first-term `real→nominal` (8);
`(m−1)→m` (6); drop first-period `ψ` (1); drop later-period `ψ` (1); fixed-leg drop `ψ` (1); seasoning
ratio `current/reset → reset/current` (2).

**Inflation cap/floor, eq 2.6.5 (`engine/capfloor.py`)** — all 3 CAUGHT: drop accrual (2); strike
`1+K→K` (29); drop `Pn(pay)` discount (10).

**Nominal par-swap bootstrap, eqs 5.1.1/5.1.2 (`market/nominal_bootstrap.py`)** — all 4 CAUGHT:
numerator sign `1−rS·A → 1+` (6); denominator sign `1+rS·ψ → 1−` (6); annuity `j=1..i−1 → j=1..i`
(include maturity, 5); `swap_annuity` drop `ψ` (1). The known-curve round-trip (mutual-inverse
requirement oracle) bites.

**MM-I freezing recursion + eq 3.1.11 (`models/market_model_i.py`)** — all 7 CAUGHT, including the
charter-documented T12(b) trap **drop `/(1+τF_r)` denominator** (1): freezing real-term sign (5);
drop `/(1+τF_r)` (1); drop `/(1+τF_n)` (1); integrand `mu_y→−mu_y` (2); drop the `−½σ_x²ρ²` convexity
in the numerator (2); variance `span→lead` (3); `mu_y` real-index term sign (2). The pointwise
transcription + non-uniform fixtures now catch the denominator the §4 anecdote records surviving at
T12.

**MM-II drift `Di` + variance, ch 4 (`models/market_model_ii.py`)** — all 9 CAUGHT, including both
documented ch-4 hazards: hazard-1 **drop `/(1+τF_n)`** (`1+Fn` vs `1+τFn`) (5); drift `σ_pay·ρ` sign
(5); drop `σ_start` outer factor (9); drop `lead` factor (6); variance lead `−2ρ→+2ρ` (9); hazard-2
**drop the augmented `σ_pay²·span` term** (9); `lead==0` branch `span→0` (8); drop variance `lead`
factor (6); drop `σ_start` in the drift bracket (8).

**Calibrations (`engine/jy_calibration.py`, `mmii_calibration.py`, `hw_cap_calibration.py`)** — all
load-bearing formula mutations CAUGHT: MM-II bootstrap forward real-ratio invert (1), drop `accrual`
in `black_value` (2), strike `1+K→K` (1); JY param assembly `σ_real↔σ_index` swap (3); HW
`black_cap_price` forward drop `/accrual` (2), drop `expiry` in `σ²·expiry` (2), forward `−1→+1` (13),
drop `accrual` in the sum (11). Synthetic-recovery (A1) + sensitivity (A2a) oracles bite.

### The one equivalent-mutant survivor (not a hole)

`jy_calibration.py:166` residual `pv − target → target − pv`: **SURVIVES, proven equivalent.**
`least_squares_fit` minimises `Σ residualᵢ²`, which is bit-identically invariant under negating every
residual. Proven, not assumed (§4): a driver run of `least_squares_fit` on a nonlinear vector
residual and its negation from the same start returns **bit-identical** `x`, `cost` and iteration
count (7). This is the documented sum-of-squares class (charter §4; cf. T14 §9.3). Not re-flagged
as a finding.

### Domain examined (so the verdict's scope is honest)

Mutated and tested: `models/jarrow_yildirim.py` (convexity + variance), `models/market_model_i.py`,
`models/market_model_ii.py`, `engine/inflation.py` (ZCIIS + linker), `engine/yoy.py`,
`engine/capfloor.py`, `market/nominal_bootstrap.py`, `market/fixing_series.py` (T7 lag + LINEAR
interp — 5 mutations, all CAUGHT), `engine/jy_calibration.py`, `engine/mmii_calibration.py`,
`engine/hw_cap_calibration.py`. **55 mutations total; 53 CAUGHT, 1 proven-equivalent survivor, 1
genuine hole (F-1.2).** Not exhaustively re-mutated: `engine/jy_full_calibration.py` (T11 — its
oracles are behavioural identifiability/spread checks, and it shares `jy_params_from` which was
mutated via `jy_calibration`); the pure-plumbing solver seams (`least_squares_fit` call-sites, the
`PricingFailure` wrapping), which are not formulas. The focus was the load-bearing numerical
formulas the §9 brief names.

### Updated summary (this section)

| Item | Result |
|---|---|
| Inflation-arc oracle power (T2–T14) | **1 HOLE (MED)** + 1 proven-equivalent survivor / 55 mutations |

F-1.2 is the project's signature defect (a degenerate fixture whose key parameter sits where it
cannot be seen) recurring a seventh time, now on the inflation-linked bond's coupon accrual. It is
handed to Cowork to rule, per §10 — advisory fix above, nothing patched here.

---

# Rulings (Cowork, §10 step 4) — 2026-11-07

**The re-audit did its job.** Oracle power over the Jäckel arc came back clean (26 mutations, 0
holes; the two survivors proven equivalent). The charter is clean. And the **extended sweep of the
inflation arc** — added precisely because the first pass's domain excluded the oldest oracles (T2–T14)
— found the one real defect (F-1.2). That is §4's "an oracle must contain the failure in its domain"
applied to the audit itself, and it is what the second pass bought.

## F-1.2 [MED] — linker coupon accrual degenerate-fixture hole → fix-tranche **T23** (oracle hardening)
Production code is **correct** (`ψ` is present and computed); the *oracle* is degenerate — every
linker coupon fixture uses exactly-365-day ACT/365F periods, so every `ψᵢ = 1` and a dropped-`ψ`
mutant survives all 818 tests, hiding a ~8.4% mispricing on a semi-annual linker. The charter's own
`τ=1` trap (§4), on the oldest code, caught only because the sweep was extended. **T23 is test-only**
(no `src` change expected): add an independent closed-form linker-PV oracle on a **non-unit-`ψ`**
fixture (`ψ≠1` asserted as a premise, §4) that pins `ψ` handling pointwise *and* kills the drop-`ψ`
mutant. RED/mutation proof is Code's by execution — new oracle green on production, red on the mutant.
If the non-degenerate fixture surfaces a real `ψ` bug (auditor says it won't), raise it, don't silently
fix. Spec: `docs/tranches/23_reaudit_linker_accrual_oracle.md`.

## F-3.1 [MED] — `conventions_register.md` C1–C6 describe the archived predecessor → fix-tranche **T24** (register rebuild, docs)
§4's cross-module-convention clause relies on the register being an accurate enumeration of the
conventions that **exist**; C1–C6 enumerate a tree (XVA/equity/SABR/vanilla) archived at the v0.0.0
scaffold, and "Immediate reading" asserts a worry about modules that aren't here. Rebuild: partition
into "predecessor lessons (archived)" vs "this tree's live register"; prune/re-derive C1–C6; add the
live rows that have no entry — invariant-6 across the 5 real engines (`cash`, `inflation`,
`nominal_cap`, `capfloor`, `yoy`) and the `model.market`/snapshot discounting convention — each with
its honest Guard state (guarded / per-module / owed); rewrite "Immediate reading." Adding a
cross-module guard for an "owed" convention is in scope if cheap; the spine is the honest
re-derivation. Spec: `docs/tranches/24_reaudit_register_rebuild.md`.

## F-3.2 [LOW] — `to_go_deeper.md` stale lines → folded into **T24**
"Six crosscheck markers" → 23; "L5 is empty" reworded to name what L5 now holds (Black/forward
Greeks, T22) vs what the inflation risk-layer route (bump-and-reprice over a `Priceable` protocol)
still wants — the route survives, the blanket sentence is false.

## F-2.1 [LOW] — driftless-forward terminal inlined in `black_mc` + `sensitivities` → `to_go_deeper` record (no tranche now)
A legitimate rule-of-two second consumer sharing the shape (forward under its own measure); both are
identical one-liners today, so the drift risk is real but small. Extract a shared `_terminal` at the
next touch of either file; recorded in `to_go_deeper` (added by T24) so it isn't lost.

## F-2.2 [LOW] — discount-from-forwards in three places → no action
Consistent convention, genuinely different shapes (static tuple / scalar loop / vectorised) — §6 says
that is *not* a shared shape. Recorded; revisit only if a fourth appears.

**Sequence:** T23 (correctness-adjacent, first) → T24 (docs). Both short; neither opens a new source
arc. The next-source direction is a separate decision, settled with Bernardo after remediation.
