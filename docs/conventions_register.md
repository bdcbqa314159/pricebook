# Conventions register

**Started 2026-10-30** (ruling: `handoffs/rulings_CP_xva_reporting.md` §2). A **living doc**, like
the drawdown tracker — not law. The *obligation to maintain it* promotes to a CLAUDE.md §4 clause
on the second cross-module-convention bug.

## Why it exists

CLAUDE.md §4's cross-module-convention clause says *a convention shared across modules needs a
cross-module oracle*. It does not say **which conventions exist** — and that gap is what let one
ship: "exposure profiles are discounted to today" was folkloric, three builders honoured it, one
did not, and **equity CVA was 5.14% too large for two slices while every oracle passed**. Each
oracle was internally consistent; none compared the modules.

So this file enumerates what the clause has to check against. It is **documentation of facts that
already exist**, not an abstraction — §6b's rule of two governs software abstractions, never
enumeration.

## How to read the Guard column

- **guarded** — a named test compares the modules, not just each module against itself.
- **per-module** — each module is tested, but *nothing asserts they agree*. This is precisely the
  state the discounting bug was in. A convention here is a candidate for the next cross-module
  gate, not a satisfied one.
- **owed** — no oracle at all. Say so; an unguarded convention that reads as guarded is worse than
  one openly marked.

---

## The register

| # | Convention | Who must agree | Guard |
|---|---|---|---|
| C1 | **Exposure profiles are PV-dimensioned** — discounted to today before any XVA integral, because `cva` weights by `ΔQ` and discounts nothing | `risk/exposure.py` (4 builders), `risk/wwr.py`, `risk/xva_report.py`, consumed by `risk/xva.py` | **guarded** — `test_every_exposure_profile_is_PV_DIMENSIONED_the_convention_cva_requires` (`tests/L5/test_equity_mc_exposure.py`), rates builder as control |
| C2 | **Option vol lives on ONE clock** — `vol_time_measure`, ACT/365F from the valuation date — so a vol read at an expiry is consistent with the `t` it is priced at | **10 modules**: `models/black.py`, `black_model.py`, `sabr_model.py`, `process.py`, `calibration/sabr_calibrate.py`, `engine/vanilla_option.py`, `engine/equity.py`, `risk/exposure.py`, `risk/mtm.py`, `risk/wwr.py` | **OWED** — tests *use* `vol_time_measure` in fixtures; none asserts every vol-consuming module reads its `t` from it. Widest-spread convention in the tree and the least guarded |
| C3 | **Historical flows are excluded from PV** (invariant 6) — cashflows on or before the valuation date are the L6 shell's, never discounted with a non-positive `t` | `engine/linear.py`, `credit.py`, `xccy.py`, `vanilla_option.py`, `seasoned.py`, `risk/exposure.py` | **per-module** — `tests/L4/test_current_period.py` guards the swap path thoroughly (splice, accrued, boundary, missing fixing). Nothing asserts the *other* engines partition identically |
| C4 | **The equity forward is `F = S·e^{(r−q)T}`** | `market/building_blocks.equity_forward`, `engine/equity.py`, `models/process.py`, `risk/mtm.py` | **guarded** — `test_the_equity_forward_is_S_times_exp_r_minus_q_T` (`tests/L4/test_equity_vanilla.py`), plus put-call parity, which is model-free and checks the forward against the discount |
| C5 | **`r` is never a separate input — it comes from the discount factor** (`e^{r·t} = 1/df` by definition of the curve), so a simulated drift and a pricer's discount cannot disagree off a flat curve | `building_blocks.equity_forward`, `models/process.BlackScholesProcess._growth`, `risk/mtm._carry_and_df` | **per-module** — each composes `df`; no oracle asserts the absence of an `r` parameter, and a non-flat curve is what would expose a divergence |
| C6 | **`cva` weights by `ΔQ`; `fva` weights by `funding_spread·Δt`** — survival increments for a loss conditional on default, a spread over time for funding | `risk/xva.py` (the shared `exposure_integral` and its three callers) | **guarded** — `test_survival_weights_are_INCREMENTS_not_levels` (sums to `1−Q(tₙ)`), `test_fva_matches_the_flat_spread_closed_form`, and `cva`-via-the-atom == `cva` to the ULP |

---

## Immediate reading

**Three of six are not cross-module guarded, and C2 is the one to worry about.** It spans ten
modules — the widest of any convention here — and is guarded only by fixtures using the right
function, which is the same standing the discounting convention had the day before it broke. C3 and
C5 are better off (per-module tests exist, and C5's modules all compose `df` rather than taking
`r`), but neither has an oracle that would notice a disagreement.

**No gate is proposed here.** The ruling made this a living doc, and the clause promotes to law on a
second convention bug — so the honest action now is to *record the exposure*, not to pre-emptively
build three gates. If C2 is what breaks next, this file is the reason it will be found in one slice
rather than two.

## Maintenance

Add a row when a convention starts being shared by a second module — that is the moment it stops
being one module's business. Move a row to **guarded** only when a test *compares the modules*;
passing per-module tests do not qualify, which is the whole point.
