# T59 — re-audit #6 remediation (one branch, one version bump → v0.61.0)

**Cowork spec, 2026-11-19.** Re-audit #6 (`docs/audit/reaudit_06_findings.md`) swept 76/76 applied,
74 killed, 2 survived — two blind-oracle holes, no shipped mispricing — plus six findings ruled below.
Per the reaudit/05 precedent this is **one test-centred remediation tranche**: every fix under one
`CHANGELOG` entry, one `__version__` bump, one floor bump, rather than splitting the floor across
branches. L0-order-first in the RED commit; the two refactors (F-6.6, F-6.3) are purity-proved.

The oracle work is the substance; the cleanups ride along because the branch is already open on these
files. Nothing here changes a shipped number — F-6.1 and F-6.2 are blind oracles (the code is right,
the test could not see a mutation), F-6.6 is swept-clean in both directions, the rest is dead code,
names, and rows.

## What the tranche carries

### 1. MM-I's second killer (item 3 — already written & verified, just land it)
The per-leg saturation requirement (`τF/(1+τF)` is a weight, must saturate at 1) that makes a
**single-leg** denominator drop die — the existing `_DENOMINATORS_matter` companion compares against
the *both-dropped* form, so a one-leg drop landed on a third value and slipped. Both denominators now
die to two tests. Commit the written killer.

### 2. F-6.1 — the LGM numéraire's `½` (eq 11.1) has no pointwise transcription
The survivor: dropping `½` in eq (11.1) sits at **1.60 SE** (t=3) inside the file's `4·SE` band. Two
chartered causes, and the fix addresses both:
- **Add the pointwise transcription of eq (11.1).** `lgm_numeraire` appears today *only* as a divisor
  inside the martingale ratio, so it is never transcribed against the source directly — the §4 rule
  that **every load-bearing formula is transcribed pointwise at least once** (two renderings, §4) is
  simply unmet here. Transcribe `N(t) = exp(H(t)·x(t) + ½·H(t)²·ζ(t))` (confirm the exact printed form
  from `LICHTERS-2015` eq (11.1) in `-raw` **and** `-layout`; do not copy from this spec — §5, our docs
  are not an authority) against an independent evaluation at a non-trivial `(x, t)`.
- **Make the resolution statement name *which* defect.** The file's existing resolution test is a
  *correct* §4 measurement against the eq-(11.4) **convexity** defect (7.5 SE) — it is not wrong, it is
  **silent about a different defect in the same formula family**. This is F-5.3's per-defect rule one
  arc on, in its sharper "satisfied clause hides the sibling" form. State, from measurement, that the
  band resolves the convexity defect and add the `½`-defect's own resolution (it is T17-weak — ~1.6 SE
  — not T34-invisible, so a higher-path power test where it falls outside the band is the companion, not
  a closed-form-only pin). The pointwise transcription is the primary anchor; the power test states the
  band's reach.

### 3. F-6.2 — T4's YoY `ψ` is τ=1-blind; the fixed leg has **zero** killers
`yoy.py`'s one fixture is `VAL + 365·k` on ACT/365F, so **every `ψ` is exactly 1.0** — the τ=1
degeneracy, in the oldest corner (§9 "full-tree means the oldest tranches", earning itself again).
Domain sweep: 0.0% at the fixture, +100.5% semi-annual, +301% quarterly, +1117% monthly.
- **Add a non-unit-`ψ` YoY fixture** (semi-annual or quarterly accruals) — **one fixture closes all
  three `ψ` sites** (the fixed leg's unguarded one and the two inflation-leg sites whose only killer is
  the cross-product `test_market_model_ii.py` oracle). This is the premise §4 already demands:
  the fixture must make the dropped-`ψ` branch produce a different number.
- **Correct the `audit_log.md` conflation.** The row reading "the YoY leg's first `ψ` … given a second
  killer at T37" describes T3's **linker** (re-measured at 2 killers — T37 holds), **not** this YoY
  module. Two products were conflated. Split the row: the linker stays at 2; the YoY module's fixed-leg
  `ψ` was at 0 and this tranche takes it to ≥1 (its own fixture) and the inflation-leg `ψ` from 1 to 2.

### 4. F-6.5 — finish the T29 narrowing (name + class, not only the provenance line)
`tests/L5/test_collateral.py:175` is still `test_collateral_can_never_INCREASE_the_adjustment` inside
`class TestTheRequirementOraclesTheFinanceForces` — both assert as a **finance-forced requirement**,
exactly the claim the narrowed `oracle:` provenance line now retracts. **The check is right and stays.**
Rename the test (e.g. `test_collateral_does_not_increase_the_adjustment_for_a_well_behaved_mark`) and
move it out of the finance-forces class into a plausibility-check group (or rename the class), with a
docstring stating it is a plausibility / implausible-input detector, not a theorem. *(Process note for
Cowork, not a clause: when narrowing an oracle's description, grep the claim's restatement in test
**names and class names** too — pass (3) narrowed the provenance and missed these.)*

### 5. F-6.4 — bring the single-killer table current
Heading already renamed by Code. Add the T45–T58 single-killer rows (two re-audits' worth) so the
standing re-run table is current.

### 6. F-6.3 — dead code cleanup + a *narrow* gate (ruled: `F841` yes, `F401` no)
- **Remove the 2 dead locals.** `collateralised_capstone.py:128` `states = simulate_lgm_states(...)` is
  assigned and never read — line 130's `per_date` does the real per-date draw, so T58 **simulates a full
  path set to the last date on every call and discards it** (dead *and* wasteful). Remove it.
  `yoy.py:99`'s `measure` likewise.
- **Remove the 10 genuinely-unused imports.** The other 10 are `@register` side-effect imports, correct
  as written — leave them.
- **Gate `F841` only.** Add `F841` to `ruff.toml`'s `select` (now `["PLR0913", "SLF001", "F841"]`). An
  unused local is statically decidable **and has no legitimate exception** — a clean gate that cannot
  false-flag, which §9 says is gateable. **Do *not* gate `F401`**: the registry side-effect imports are
  a legitimate exception, so `F401` wholesale needs per-file-ignores — an exemption list, the shape §9
  declines (the "7-of-14 exemption list cannot fail" rule). Route to `to_go_deeper`: *gate `F401` once
  the registry modules carry `__all__` entries so no exemption list is needed* — then the gate is honest.

### 7. F-6.6 — consolidate the era selection into ONE L1 home (rule-of-two has fired)
`engine/interest_rate_swap.py`'s `_curves` helper docstrings itself "**The one place the mode is
selected**" — but `risk/collateralised_capstone.py:102` re-implements the same
`ProjectionKey(...) in market.curves.projections` lookup + fallback inline, so that claim is **false
today**. Two consumers, one shared **shape** → §6 rule-of-two fires (and it is the T30 duplicated-guard
shape). **Swept clean in both directions this re-audit, so there is no hole — this is the abstraction
§6 says to build at the second consumer, not a bug fix.**
- **Build `curves_for_era(currency, tenor) -> (projection, discount)` as a public method on `CurveSet`
  (L1)** — where `projections` actually lives, and below both consumers (L4 swap, L5 capstone), so
  dependencies point down. Both the swap's `_curves` and the capstone call it; the swap's "one place"
  docstring becomes true.
- **Why L1 and why now, specifically:** the single-curve design invariant this arc established is
  *realised through this exact lookup*. Single-sourcing it means the invariant has **one structural
  home** instead of being re-derived per consumer — the consolidation strengthens the very invariant,
  which is the real argument over tidiness.
- **Purity proof:** the lookup is dict-membership + a branch over curve handles — integer/reference ops,
  **no reordering of floating-point work**, so op order is preserved and the proof is **bit-identity
  in-process** (§4), with the existing swap and capstone oracles green at their own tolerances as the
  committed regression (a tolerance, never a committed bit-pattern — the T49 rule). Confirm the
  capstone and swap PVs are bit-identical before/after in one interpreter.

### 8. OI-6 — close the only open `OPEN.md` item (natural touch here)
T8's skipped-pillar `KeyError` (§3.4, low severity): `nominal_bootstrap.py` raises a bare `KeyError`
on a pillar gap instead of returning a `PricingFailure` (invariant 4 — failure is a value). One-line
guard + a RED test that asks for a skipped pillar and asserts a `PricingFailure`. Its re-open trigger
is "the next `nominal_bootstrap.py` touch" — this is one. Close OI-6.

## Oracle / gate discipline for the tranche
- RED-first for every new oracle (MM-I killer already verified; F-6.1 transcription, F-6.2 non-unit-`ψ`
  fixture, OI-6 failure test all land RED before GREEN).
- F-6.1 and F-6.2 are **blind-oracle closures on previously-shipped code** → §9 wants each new killer to
  be a *second independent* killer where the site guards an already-made mistake; the YoY fixture is a
  fresh independent killer, the eq-(11.1) transcription is independent of the ratio's divisor use.
- Re-time the touched heavy-MC files (capstone, LGM) on **both CI runners** after adding oracles — §9,
  a local re-time is not the measurement; the three-ceiling watch stands.
- Floor bumps once for all new tests; `__version__` → `0.61.0`; one `CHANGELOG` entry naming the
  re-audit.

## Not in scope (ruled out, carried)
- **C2 (vol lives on one clock)** — still the widest OWED convention; no vol touch this tranche.
- **`dividends_before(T)` consolidation** — still owed; no equity touch.
- `F401` wholesale gate — deferred to `to_go_deeper` behind the `__all__` prerequisite (above).
