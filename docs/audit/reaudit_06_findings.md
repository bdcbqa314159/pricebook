# Re-audit #6 — the collateralised-discounting arc close (v0.60.0, 2026-11-19)

Sixth whole-tree re-audit (§9), fired by the **arc-close trigger** (~nine tranches since re-audit #5 at the
Gatheral close — T50–T53 the Gatheral tail, T54/T55/T57/T58 this arc; arc-close fires first, as every time).
Run with independence: Cowork sets scope and rules; the mutation sweep is Code's execution on the **committed
instrument** (`tests/tooling/mutation_sweep.py`, T52); the register-accuracy and charter-staleness passes are
done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3) register
accuracy**, **(4) charter staleness**. (1) and (3) run together. Conduct rule (the house discipline): a
surface's report is content, not the artifact — Code re-derives each premise; the sweep reports
applied/killed/survived with the bytecode and ERRORED guards (the instrument encodes them).

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST — the proactive invariant-6 / historical-flow sweep** (the tenth-shape escalation's standing run).
   Drop the filter on every module that has one; the arc added the IRS, the CSA pricer and the collateralised
   exposure. The option-bypass sub-shape (F-4.1 moneyness) and the straddle premise both apply.
2. **The single-curve design invariant — a NEW named sweep this arc, because its failure is silent (T33b).**
   The arc's invariant is that single-curve and multi-curve are co-equal selectable modes. Sweep the
   discounting-touching modules (T55's pricer, T58's basis factor) for a mutation that makes a `CurveSet`
   **mandatory** (ignore a single-curve snapshot / force projections): the *differ* half of the invariant
   oracle must kill it. The invariant is structural in two places (the snapshot shape, the basis factor = 1)
   — confirm the structure actually holds, do not trust the report.
3. **MM-I's owed second killer** (`market_model_i.py:154/:155`) — two load-bearing holes behind the *same one*
   test, recorded since re-audit #5. **This re-audit's highest-value single-killer target**: add the second
   independent killer, or confirm it was added. Plus the standing single-killer re-run table
   (`docs/audit_log.md`).
4. **The new closed forms, first sweep** — the OIS/projection bootstrap (T54), the CSA pricer (T55), the LGM
   and its MC companion (T57), the threshold + the capstone wiring (T58). Probe the T39 reduction blind spot
   the arc paid twice (my eq-11.4 and threshold errors) and the T43 scale-invariance; confirm the LGM's HW
   reduction + MC companion both bite.
5. **The rest of the tree (T1–T53)** as the standing full-tree sweep.

Survivors ruled by Cowork, appended here. Any real hole → a test-only fix-tranche (the T23/T37/T45/T51
pattern).

## (2) Duplication — PARTIAL (Cowork read; Code sweep informs)

- **C2 (vol lives on one clock) — still OWED, the widest-unguarded.** Untouched this arc. Carried.
- **C14 (total-variance dimension, T46) and C15 (OIS-vs-projection, T55) — both guarded.** No new OWED
  convention opened this arc.
- **`dividends_before(T)` consolidation — still owed** (no equity touch this arc). Carried.
- **Value objects single-sourced — verified (Code).** `Tenor`, `ProjectionKey`, `SwapDirection`,
  `InterestRateSwap`, `CollateralAgreement`: one definition each.
- **But the era-selection logic is duplicated — see F-6.6.** §6's rule-of-two has fired at its second
  consumer with a shared *shape*; no hole today (item 2 swept both copies), the duplication is the risk.

## (3) Register-accuracy pass — DONE (Cowork)

- **`risk/collateral.py` (T29) — the oracle wording NARROWED (the T58 ruling, done here).** "Collateral
  never increasing the adjustment" overclaimed a theorem; it is a **plausibility check** (aggregate, a
  well-behaved mark; a gross breach signals implausible inputs — which is how it caught T58's reciprocal
  basis). The `oracle:` provenance line now says so — the §4 floor rule applied to an oracle's own
  description. No code/test change; the check stays, its description is honest.
- **`to_go_deeper.md` — the `LICHTERS-2015` header fixed** ("T54–, in progress" → "T54–T58 CLOSED v0.60.0");
  the T56 cross-currency deferral and the T29-fence routes are current.
- **`references.md` — clean.** `LICHTERS-2015` is `worked-from` with `src/` citers.
- **`OPEN.md` — OI-6 open and correct** (the T8 skipped-pillar `KeyError`, §3.4, low severity, re-open at
  the next `nominal_bootstrap.py` touch or this re-audit). It is the only open item. **Candidate to fix in
  the remediation tranche** if this re-audit produces one — a one-line `PricingFailure` guard with a test,
  a natural `nominal_bootstrap.py` touch.
- **`conventions_register.md` — clean and gated** (C15 added guarded, T55; `verify.py conventions` holds).
- **Arc plan (`modern_discounting_arc.md`) — current.** T56 struck, queue closed, single-curve invariant
  recorded, completion ratio matches the checkpoint.

## (4) Charter-staleness pass — DONE (Cowork): clean

- **No clause stale or in tension.** The two arc additions (§10 the step-1 reuse grep; §4 the
  relative-tolerance mirror) describe reality and each cite the earning tranche. The deliberate non-additions
  (the Cowork transcription errors caught by §5 + named reductions; the T29 narrowing; the frozen-basis
  choice; the loop-invariant lens) left the charter leaner-relative-to-coverage, as at the last three
  capstones.
- **Method-watch items carried (not charter):** the three-ceiling heavy-MC-file watch (a fourth earns the
  `slow`-budget conversation, §9, Bernardo's); the glyph-recovery sub-kinds (2 instances, from the Gatheral
  arc) and the first-shape-twice counter (Gatheral) — both still at two, carried.

---

## (1) Oracle-power mutation sweep — DONE (Code). **76 applied, 74 killed, 2 real holes**

Run on the committed instrument (`tests/tooling/mutation_sweep.py`, T52), oldest-first, in Cowork's order.
Every total below is **applied/killed/survived**, never a bare survivor count (§9).

| item | scope | applied | killed | survived |
|---|---|---|---|---|
| 1 | the proactive invariant-6 / historical-flow sweep | 24/24 | 24 | 0 |
| 2 | the single-curve design invariant's reachability (NEW) | 6/6 | 6 | 0 |
| 3 | MM-I's owed second killer + the single-killer re-run | 2/2 | 2 | 0 (**1 hole found in the re-run**) |
| 4 | the new closed forms — T54 / T55 / T57 / T58 | 20/20 | 19 | **1** |
| 5 | the rest of the tree, T1–T53, oldest-first | 24/24 | 24 | 0 |
| | **total** | **76/76** | **74** | **2** |

**Item 1 — clean (24/24).** The arc's three new filters (the IRS `_live`, the CSA pricer, the collateralised
exposure) all bite, and the F-4.1 option-bypass sub-shape found nothing new. The standing run stays armed.

**Item 2 — clean (6/6), the invariant confirmed structural rather than reported.** Mutations forcing a
`CurveSet` (ignore a single-curve snapshot, force a projection lookup, swap the era branches, invert the
basis factor) all die, in **both** directions: T55's `_curves` era selector and T58's `_basis_factor`
`projection is discount` shortcut. Re-derived, not taken from the report (the conduct rule).

**Item 3 — the second killer ADDED, and the re-run found a hole.** `index_vol_step`'s two denominators now
die to **two** tests each. The reason one existing companion did not already serve is worth recording:
`test_the_index_vol_step_DENOMINATORS_matter` compares against the form with **both** denominators dropped,
so a **single-leg** drop returns a third value that is neither variant and slips through. The new killer is a
**requirement**, not a second transcription — `τF/(1+τF)` is a weight, so it must **saturate at 1** — and it
asserts each leg separately, which is what makes a single-leg drop die (at `τF = 1000` the weight is 0.999
against an undivided 1000, three orders, versus the ~2% a realistic fixture shows).

Single-killer re-run (`docs/audit_log.md`'s table), measured by counting **every** failing test, not the first:

| recorded entry | killers now | verdict |
|---|---|---|
| MM-I's `τ/(1+τ)` denominators | 1 → **2** | hardened here |
| T3 linker's first `ψ` (F-1.2) | **2** | T37's hardening holds |
| T26's Leg B numéraire divisor | **1** | single **by design**, as recorded |
| T4 **YoY** leg's first `ψ` | **1** | → **F-6.2**: the table's note is about the *linker*, not this |

## Findings

### F-6.1 — the LGM numéraire's `½` is unpinned (item 4's survivor). Blind oracle, `main` since T57

`lgm_numeraire` is eq (11.1), `N(t) = exp(H_t z_t + ½H_t² ζ_t)/P(0,t)`. Dropping the `½` **survives all 2322
tests.** Measured on the oracle's own fixture and seed:

| oracle point | correct | `½` dropped | band |
|---|---|---|---|
| `t=1, T=5` | 1.52 SE | 1.18 SE | 4 SE |
| `t=3, T=10` | 1.38 SE | **1.60 SE** | 4 SE |

The relative shift is `exp(−½H_t²ζ_t) − 1`: **0.0072%** at `t=1` and **0.197%** at `t=3`. Two causes, and
both are chartered shapes:

1. **§4's per-defect band resolution (F-5.3, one arc on, and this is its sharper form).** The file carries
   `test_THE_RESOLUTION_this_band_sees_the_convexity_term`, which correctly measures the band against the
   eq-11.4 convexity defect at **7.5 SE** — a real, §4-compliant resolution test. It says nothing about a
   *different* defect in the *same* formula family, and the satisfied-looking clause is exactly what hid it.
   The defect here is **weakly observable** in the martingale ratio (T17's shape, not T34's — the
   sensitivity is nonzero, just 0.2%), so no path count is the fix.
2. **Eq (11.1) has no pointwise transcription.** `lgm_numeraire` appears in `tests/L3/test_lgm.py` only ever
   as a **divisor** inside the martingale ratio (lines 156, 184, 223) — never on its own against an
   independently computed value. §4: *"every load-bearing formula is transcribed pointwise at least once,
   and carries its requirement oracles; neither substitutes for the other."* Eq (11.1) has only the
   requirement.

**Not a pricing defect:** the shipped code is correct. The oracle set is what is thin.

### F-6.2 — T4's YoY module is `ψ`-blind: the FIXED leg's `ψ` survives the whole suite. Real hole, `main` since T4

Found by item 3's single-killer re-run, not by the sweep proper. `tests/L4/test_yoy_inflation_swap.py`'s one
schedule is `VAL + 365·k` on ACT/365F, so **every `ψ` is exactly `1.0`** — the `τ=1` degeneracy, in the
oldest corner of the tree (§9's "full-tree means the oldest tranches"). Consequences, all measured:

| mutation | killers | note |
|---|---|---|
| `yoy_inflation_leg_value`, the first `ψ` | 1 | and it is in **another product's file** |
| `yoy_inflation_leg_value`, the loop `ψ` (periods 2..M) | 1 | the same single test |
| **`yoy_fixed_leg_value`, `ψ`** | **0** | **a surviving mutant** |

The single killer of both inflation-leg sites is
`tests/L3/test_market_model_ii.py::test_cap_MINUS_floor_under_MMII_is_still_the_YOY_LEG` — T13's
cross-product oracle, in a **different layer's test file for a different product**. §4 credits exactly this
("a cross-product oracle catches a shared error no within-product check can"), and it is doing all the work
here: the YoY file's own transcription *does* write `psi1` and the loop `psi`, but multiplies by `1.0` on
both sides, so it cannot see them.

`yoy_fixed_leg_value` has no cross-product cover at all. Severity swept across the domain rather than taken
from one point (§8), fixed rate 2%, flat 3%:

| schedule | `ψ` | correct | `ψ` dropped | fixed leg overstated |
|---|---|---|---|---|
| annual ×5 (**the fixture**) | 1.0000 | 0.091475 | 0.091475 | **0.0%** |
| semi-annual ×10 | 0.4986 | 0.091934 | 0.184374 | **+100.5%** |
| quarterly ×20 | 0.2493 | 0.092280 | 0.370132 | **+301.1%** |
| monthly ×60 | 0.0822 | 0.091568 | 1.114079 | **+1116.7%** |

**This is F-1.2's sibling site, never hardened.** Re-audit #1 found the `τ=1` dropped `ψ` in T3's *linker*
and T37 gave it a second killer (confirmed **2** above). The *same* defect class in T4's YoY swap — two
sites in the inflation leg at one killer each, one site in the fixed leg at **zero** — was carried in
`audit_log.md` as "the YoY leg's first `ψ` … given a second killer at T37", conflating the two products. The
linker was hardened; the YoY swap was not.

### F-6.3 — Pyflakes never runs, so dead code accumulates unseen. Not a defect; a gate question for Cowork

`ruff.toml` has `select = ["PLR0913", "SLF001"]`, both added because a specific defect got through. `F`
(Pyflakes) is therefore **not** selected and never has been. Running it over `src/pricebook` finds **22**:

- **2 × F841, dead locals.** `risk/collateralised_capstone.py:128` — `states = simulate_lgm_states(params,
  times[-1], sampling)`, assigned and never read, so T58 **simulates a full path set per call and discards
  it** (`per_date` is what the loop uses); and `engine/yoy.py:99` — `measure`, T4-era. Both are provably
  inert (`simulate_lgm_states` seeds its own `UniformSource` per call, so deleting it cannot move a number),
  but the capstone one is a real wasted cost in the heaviest file this arc built.
- **10 × F401, genuinely unused imports**, across `engine/mc_path_engine.py`, `engine/mmii_calibration.py`,
  `engine/registry.py`, `engine/synthetic_cdo_tranche.py`, `engine/yoy.py`, `market/snapshot.py`,
  `products/credit_default_swaption.py`, `risk/collateralised_capstone.py`, `risk/mc_exposure.py`,
  `risk/sensitivities.py`.
- **10 × F401 that are correct as written** — the `@register` side-effect imports in the dispatch module,
  which is precisely Pyflakes' "consider adding to `__all__`" case.

Code raises this; the ruling is Cowork's, because it is a **gate** question and §7/§9's precedent cuts both
ways: the references gate earned its place off five drifts, while a gate needing an exemption list cannot
fail. The §9 test applies cleanly — an unused local is **statically decidable**, so it is gateable — but the
registry imports mean `F` cannot be selected wholesale without either `__all__` entries or a per-module
exemption, which is the shape §9 declines.

### F-6.4 — register: the single-killer table is two re-audits stale

`docs/audit_log.md`'s table is headed **"Accumulating for re-audit #4"** and its last row is T44b, though
re-audits #5 and #6 have since run. Re-audit #5's single-killer record went into
`docs/audit/reaudit_05_findings.md` instead, so the table §9 names as the standing re-run target no longer
holds the targets. (Cowork's register pass (3) read the rows; the heading and the gap are what this adds.)

### F-6.5 — register: the T29 narrowing is incomplete — the test still asserts what the docstring retracted

Pass (3) narrowed `risk/collateral.py`'s `oracle:` line from "collateral never increasing the adjustment" to
a **plausibility check**. The test it describes was not touched: `tests/L5/test_collateral.py:175` is still
`def test_collateral_can_never_INCREASE_the_adjustment`, inside
`class TestTheRequirementOraclesTheFinanceForces` — both the name and the class assert it as a requirement
**the finance forces**, which is exactly the claim the provenance line now retracts. The check itself is
right and stays; its name and its class are the stale half. One-line rename plus a docstring, in the
remediation.

### F-6.6 — duplication (pass 2): the era selection is written twice, and §6's rule-of-two has fired

Pass (2) asked Code to verify the arc's types are single-sourced. They are — `Tenor`, `ProjectionKey`,
`SwapDirection`, `InterestRateSwap` and `CollateralAgreement` each have exactly one definition. But the
**era-selection logic** — the one place the charter says the mode is chosen by reading the market rather
than a flag — exists in two:

| site | form |
|---|---|
| `engine/interest_rate_swap.py:102` (`_curves`) | `if ProjectionKey(currency, tenor) in market.curves.projections:` → `(projection, discount)`, else `(discount, discount)` |
| `risk/collateralised_capstone.py:102` | `key = ProjectionKey(currency, swap.tenor)`; `projection = … if key in market.curves.projections else discount` |

This is a **shared shape, not a shared word** (§6's trigger): the same lookup, the same fallback, the same
meaning. It is also the **T30 eighth-degeneracy shape** — one guard in N copies, where a fixture exercising
the branch in one copy leaves the other untested. Item 2's sweep covered **both** copies in both directions
and both bite today, so there is **no hole now**; the duplication is the standing risk, and §6's remedy is
to consolidate rather than to keep multiplying fixtures. Code raises it; where the single home belongs
(a public helper on `CurveSet` at L1, or a public `curves_for_era` in the engine that L5 imports) is
Cowork's design call.

## Status

- (1) **DONE (Code): 76 applied, 74 killed, 2 real holes** — both **blind oracles, no pricing defect.**
  Items 1, 2 and 5 clean; item 3's second killer added (1 → 2 per denominator) and its re-run turned up
  F-6.2; item 4's one survivor is F-6.1.
- (2) **DONE** — Cowork's read plus Code's verification. Types single-sourced; **F-6.6** opened (the era
  selection written twice). C2 still OWED (widest-unguarded) and the `dividends_before(T)` consolidation
  still owed, both carried untouched this arc.
- (3) and (4): **DONE (Cowork)** — with two register items this pass adds: **F-6.4** (the single-killer
  table is headed for re-audit #4 and stops at T44b) and **F-6.5** (the T29 narrowing left the test's name
  and class asserting what the provenance line retracted).

**Six findings for Cowork to rule**, none a shipped mispricing:

| # | what | kind | severity |
|---|---|---|---|
| F-6.1 | the LGM numéraire's `½` survives (1.60 SE inside a 4 SE band; eq 11.1 never transcribed pointwise) | blind oracle | the band cannot resolve it at any path count |
| F-6.2 | T4's YoY `ψ` — the **fixed leg's survives entirely**; both inflation-leg sites have one killer, in another product's file | blind oracle (`τ=1`) | +100.5% / +301% / +1117% off-fixture |
| F-6.3 | Pyflakes never selected — 2 dead locals (one a discarded LGM path set) + 10 unused imports | gate question | no numerical effect |
| F-6.4 | the single-killer table is two re-audits stale | register | — |
| F-6.5 | the T29 narrowing left the test name and class behind | register | — |
| F-6.6 | the era selection written twice (rule-of-two fired) | duplication | no hole today |

A remediation tranche is earned: **F-6.1** (a pointwise eq-11.1 transcription + a resolution statement naming
*which* defect), **F-6.2** (a non-unit-`ψ` YoY fixture, which closes all three sites at once and is the
`τ=1` premise §4 already demands), **F-6.5** (rename), and optionally **F-6.4**. F-6.3 and F-6.6 are
Cowork's rulings before any code moves. It can also close **OI-6** — still the only `OPEN.md` item, and a
one-line `PricingFailure` guard is a natural `nominal_bootstrap.py` touch.

**MM-I's second killer is written and measured, and lands in the remediation tranche** — not here. It was
item 3's named deliverable and it is done work (`test_each_freezing_WEIGHT_saturates_at_one_PER_LEG`,
verified 1 → 2 killers on each denominator), but re-audit #5's findings branch was **docs-only** and its
hardenings went to T51 with their own version bump, `CHANGELOG` entry and floor bump. Following that
precedent keeps one test-only tranche carrying every fix — MM-I's killer, F-6.1's pointwise eq-11.1
transcription, F-6.2's non-unit-`ψ` fixture, F-6.5's rename — under one stamp, instead of splitting the
floor across two branches.

Re-audit #6 **logged open**; the arc boundary is settled once the remediation is ruled.

---

## Cowork rulings (2026-11-19) — all six findings ruled; **T59** specced

Spec at `docs/tranches/59_reaudit6_remediation.md` — one test-centred tranche, reaudit/05 precedent,
→ **v0.61.0**, one floor bump, one `CHANGELOG`.

- **F-6.1 — fix both causes.** Add the missing **pointwise eq-(11.1) transcription** (`lgm_numeraire` is
  only ever a divisor, so §4's "every load-bearing formula transcribed pointwise once" is unmet) + make the
  resolution statement name *which* defect (the band correctly resolves the eq-(11.4) convexity defect and is
  silent about the `½` sibling — F-5.3 sharper). The `½` is T17-weak, so a power test states the band's reach.
- **F-6.2 — one non-unit-`ψ` fixture** closes all three sites; **correct the `audit_log` conflation** (the
  "second killer at T37" row is T3's *linker*, holds at 2 — not this YoY module).
- **F-6.3 — SPLIT: gate `F841`, not `F401`.** Remove 2 dead locals + 10 unused imports; keep the 10
  `@register` side-effect imports. `F841` is a clean gate (statically decidable, no exception). `F401` needs
  per-file-ignores for the registry pattern — the exemption-list shape §9 declines → `to_go_deeper` behind an
  `__all__` prerequisite.
- **F-6.4 — add T45–T58 rows.**
- **F-6.5 — finish the narrowing in the test name + class** (the check stays). *(Process note: when
  narrowing an oracle's description, grep its claim in test/class names too — pass (3) missed these.)*
- **F-6.6 — rule-of-two fired: build `CurveSet.curves_for_era` at L1**, both consumers call it (the swap's
  `_curves` "one place" docstring is false today). Single-sources the single-curve invariant into one
  structural home. Bit-identity-in-process proof (op order preserved).
- **OI-6 — closed in T59** (the `nominal_bootstrap.py` `PricingFailure` guard; natural touch).

Out of scope, carried: **C2** (widest OWED), **`dividends_before(T)`**. Re-audit #6 closes when T59 merges.
