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

*(That bug happened in the **archived predecessor tree**, not this one. It is kept because it is the
reason this file exists — per the reset ruling, the predecessor is mined for its lessons. The modules
it names do not exist here; see the partition below, which re-audit #1's F-3.1 forced.)*

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

## The register is PARTITIONED, and re-audit #1 is why

Rows **C1–C6** were carried wholesale from the archived predecessor at the `v0.0.0` scaffold and
describe modules **that do not exist in this tree** — `risk/exposure.py`, `risk/xva.py`,
`engine/linear.py`, `engine/xccy.py`, `engine/vanilla_option.py`, `engine/equity.py`,
`models/sabr_model.py` and the rest. Re-audit #1 (F-3.1) found that §4's cross-module clause leans on
this register as *the enumeration of which conventions exist*, while the file was pointing a reader at
a phantom tree and covering this tree's real conventions only partly. The old "Immediate reading"
even named C2 *"the one to worry about"* — a worry about ten modules none of which are here.

The IDs are **kept stable** rather than renumbered: `docs/handoffs/`, `docs/audit/` and CLAUDE.md all
cite them, and renumbering would silently change what those citations mean. So C1–C6 stay, labelled
archived; the live rows keep C7–C9; new live rows continue at C10.

## This tree's LIVE register

| # | Convention | Who must agree | Guard |
|---|---|---|---|
| C7 | **The break-even-curve construction, the nominal curve's `TimeMeasure`, and a ZCIIS contract's `Term.day_count` share ONE day-count convention.** The par identity (1.2.16) holds only under it: `Pr` is built by converting quote dates on the *curve's* measure, while the fixed leg compounds on the *contract's* day count. | `market/inflation_curve.py` (L1 construction) ↔ `foundation/term.py` + `products/inflation.py` (L2 contract) ↔ `engine/inflation.py` | **OWED** — the **pointwise** 1.2.16 oracle guards the constructor's arithmetic, but the **par oracle telescopes on a matched convention** and has no power to detect a divergence between the two. **Currently unrepresentable**: the library has one day count (`ACT/365F`), so the convention cannot yet be violated. **Trigger:** the first second day count — a 30/360 market, or multi-currency inflation — at which point it needs a structural guard or a *powered* oracle (a fixture where the two conventions genuinely differ). |
| C8 | **A coupon schedule's accrual day count and its discount curve's `TimeMeasure` are one convention.** A linker pays `C·ψᵢ` with `ψ` on `CouponSchedule.day_count`, while `Pr.df` is looked up by date on the curve's own measure. | `foundation/schedule.py` (L0 accrual) ↔ `market/curve.py` (L1 measure) ↔ `engine/inflation.py` | **OWED** — a **C7 sibling**, registered because `day_count` is genuinely read (thread 0019's question). The closed-form oracle computes `ψ` itself, so it *does* guard the accrual arithmetic; what is unguarded is the two conventions *disagreeing*. **Currently unrepresentable** — one day count exists. **Trigger:** the first second day count, shared with C7. |
| C9 | **`I(t)` is resolved through ONE atom** — `market.fixing_series.reference_level(series, date, index)` applies the index's publication lag and interpolation. Three engines need a reference level and must not disagree about which monthly print it is. | `engine/inflation.py` (ZCIIS + linker, via `resolved_index_level`) ↔ `engine/yoy.py` (the seasoned reset, via `_seasoning_ratio`) ↔ `market/fixing_series.py` (the atom) | **guarded — structurally.** Not by a comparison test but by there being **one function**: both call sites compose `reference_level`, so a divergence is not expressible without first duplicating it. The atom's own conventions are oracled in `tests/L1/test_fixing_series.py` with a *powered* fixture (sloped series, mid-month date, FLAT vs LINEAR required to disagree). **Trigger to re-examine:** a fourth consumer that resolves its own level instead of composing this one, or an index whose convention is not (lag, FLAT\|LINEAR) — a quarterly print, or a seasonality adjustment. |

| C10 | **A flow on or before `valuation_date` is historical** (invariant 6) — excluded from the mark, PV 0, *"never discounted with non-positive `t`"*. The comparison is therefore `pay > valuation_date`, **strict**, and the boundary case is a payment landing exactly today. | **one atom** — `foundation/time_measure.is_historical(pay, valuation_date)`, consumed by all **eight** pay-boundary sites across the five engines (`cash`, `inflation`, `nominal_cap`, `capfloor`, `yoy`×4) and by `risk/mc_exposure.live_exposure_times` | **guarded — structurally** (tranche 26) — a divergence between engines is no longer *expressible*: there is one comparison. Pinned by `tests/L0/test_valuation_boundary.py` (the boundary case, totality, bool-ness) plus the engines' own boundary tests. Flipping the atom's `<=` to `<` fails **6** tests. **Why it was `per-module` until now:** five inline copies, and tranche 24 measured that it mattered — the same slip was a real **41% mispricing** in `capfloor` (5,929 → 8,364) while a proved equivalent mutant in `nominal_cap`/`yoy`, where an expired Hull-White caplet is exactly zero. Extracted when a real consumer pulled it (tranche 26's exposure grid), not speculatively (§6) |
| C11 | **The CDS protection leg is `(1−R)∫Z λ Q dt` integrated by ONE functional, breaking at every hazard pillar.** `λQ` jumps at a pillar, so a quadrature that does not break there loses an order — and the choice has to be the same on both sides or a bootstrapped curve does not reprice its own calibrating quote. | **one function** — `market/credit_legs.protection_leg`, with `segment_breaks` deciding where it breaks, consumed by `engine/credit_default_swap.py` (`protection_leg`, `riemann_bracket`, `par_spread`) and `market/cds_bootstrap.py` | **guarded — structurally and by comparison** (tranche 39). Structurally: there is one function, so a divergence is not expressible without duplicating it first. By comparison: `tests/L1/test_credit_legs.py::TestTheEngineAndTheBootstrapShareONEProtectionLeg` asserts the engine's leg is **bit-identical** to the shared one on a term-structured *and* a flat curve (with the interior-pillar premise asserted, since a flat curve cannot discriminate), and that a bootstrapped curve reprices through the **engine** to `-5.8e-18` of notional on the same grid, the residual on a coarser one falling at second order. **Why it was not guarded before:** until tranche 39 the engine integrated the leg itself, pointwise on a uniform grid, 1.846e-05 away from the bootstrap on a term-structured curve — `OPEN.md` OI-4, ruled closed by adopting the better one. **Trigger to re-examine:** a third consumer that integrates its own, or a hazard interpolation that is not piecewise-flat (the breaks stop being the pillars). |
| C11 | **A snapshot-carrying model reaches market data only through `model.market`** — never as a second argument to a registered pricer (§3), so a model/market mismatch cannot be expressed at the entry point. | the **six** models that carry a snapshot (`discounting_model`, `hull_white`, `jarrow_yildirim`, `implied_vol_surface`, `market_model_i`, `market_model_ii`) ↔ the **six** registered pricers (`price_fixed_cashflow`, `price_zc_inflation_swap`, `price_inflation_linked_bond`, `price_nominal_cap`, `price_inflation_capfloor`, `price_yoy_inflation_swap`) | **guarded — structurally, at the ENTRY POINT only**, and the qualifier is the honest part. Every registered pricer's signature is `(product, model)`, so there is no second place to pass a snapshot *in*; `verify.py acyclic`/`layers` keep the direction. But **inside** an engine, helpers do take a snapshot explicitly (`inflation_leg_value(swap, market)`, `resolved_index_level(index, market)`, the three calibration objectives) — legitimately, since they are called *from* a pricer that obtained it from `model.market`. So the guarantee is "a caller cannot supply a mismatched snapshot", not "no function takes one". **Scope, not exceptions:** the models with no `.market` (`black`, `black_mc`, `libor_market_model`, `swap_rate_vol`, `exercise_boundary`, `index_ratio`) are closed forms, parameter objects and numerical helpers — they do not read market data, so the convention does not reach them. **Trigger to re-examine:** a *registered* pricer taking a `MarketSnapshot` alongside a model, which would make a mismatch expressible for the first time. |
| C12 | **A CVA integral's survival and discount curves share ONE `TimeMeasure`** — the exposure profile's `t` is read on both (`survival.survival(t)` and `discount.df_at(t)`), so two anchors or two day counts would make one `t` denote two different dates and the integral would mix them | `risk/credit_valuation_adjustment.unilateral_cva` **and** `risk/mc_exposure.mc_exposure_cva` (tranche 26), consuming `market/survival_curve.py` + `market/curve.py` | **guarded** — `test_two_curves_on_DIFFERENT_clocks_are_refused` (`tests/L5/test_credit_valuation_adjustment.py`); the functional refuses the mismatch rather than recording the convention. Same family as C7/C8 |

## Archived — the predecessor tree's register (lessons only, NOT this tree's inventory)

**None of the modules below exist here.** These rows are the archived predecessor's inventory, kept
because the reset ruling (`handoffs/rulings_reset_to_fresh_repo.md`) mines that tree for *lessons* —
C1's PV-dimensioning bug is the reason this whole file exists. Read them as history. Nothing here
imposes an obligation on this tree, and no row below may be cited as a live convention.

| # | Convention (archived tree) | Who agreed (modules now gone) | Guard it had |
|---|---|---|---|
| C1 | **Exposure profiles are PV-dimensioned** — discounted to today before any XVA integral, because `cva` weights by `ΔQ` and discounts nothing | `risk/exposure.py` (4 builders), `risk/wwr.py`, `risk/xva_report.py`, consumed by `risk/xva.py` | **guarded** — `test_every_exposure_profile_is_PV_DIMENSIONED_the_convention_cva_requires` (`tests/L5/test_equity_mc_exposure.py`), rates builder as control |
| C2 | **Option vol lives on ONE clock** — `vol_time_measure`, ACT/365F from the valuation date — so a vol read at an expiry is consistent with the `t` it is priced at | **10 modules**: `models/black.py`, `black_model.py`, `sabr_model.py`, `process.py`, `calibration/sabr_calibrate.py`, `engine/vanilla_option.py`, `engine/equity.py`, `risk/exposure.py`, `risk/mtm.py`, `risk/wwr.py` | **OWED** — tests *use* `vol_time_measure` in fixtures; none asserts every vol-consuming module reads its `t` from it. Widest-spread convention in the tree and the least guarded |
| C3 | **Historical flows are excluded from PV** (invariant 6) — cashflows on or before the valuation date are the L6 shell's, never discounted with a non-positive `t` | `engine/linear.py`, `credit.py`, `xccy.py`, `vanilla_option.py`, `seasoned.py`, `risk/exposure.py` | **per-module** — `tests/L4/test_current_period.py` guards the swap path thoroughly (splice, accrued, boundary, missing fixing). Nothing asserts the *other* engines partition identically |
| C4 | **The equity forward is `F = S·e^{(r−q)T}`** | `market/building_blocks.equity_forward`, `engine/equity.py`, `models/process.py`, `risk/mtm.py` | **guarded** — `test_the_equity_forward_is_S_times_exp_r_minus_q_T` (`tests/L4/test_equity_vanilla.py`), plus put-call parity, which is model-free and checks the forward against the discount |
| C5 | **`r` is never a separate input — it comes from the discount factor** (`e^{r·t} = 1/df` by definition of the curve), so a simulated drift and a pricer's discount cannot disagree off a flat curve | `building_blocks.equity_forward`, `models/process.BlackScholesProcess._growth`, `risk/mtm._carry_and_df` | **per-module** — each composes `df`; no oracle asserts the absence of an `r` parameter, and a non-flat curve is what would expose a divergence |
| C6 | **`cva` weights by `ΔQ`; `fva` weights by `funding_spread·Δt`** — survival increments for a loss conditional on default, a spread over time for funding | `risk/xva.py` (the shared `exposure_integral` and its three callers) | **guarded** — `test_survival_weights_are_INCREMENTS_not_levels` (sums to `1−Q(tₙ)`), `test_fva_matches_the_flat_spread_closed_form`, and `cva`-via-the-atom == `cva` to the ULP |

---

## Immediate reading

**This tree has six live conventions (C7–C12), and they are in good standing.** The row that was once
"the one to watch", C10 (invariant-6, historical flows → PV 0), is now **guarded — structurally**. When
it spanned five engines each inlining `pay > valuation_date` with no shared atom, it had the exact
standing the predecessor's discounting convention had the day before it broke — and re-deriving it by
mutation rather than reading found a real **41% `capfloor` mispricing** a wrong boundary comparison would
have caused (closed at T24; of the five engines, three caught a boundary flip and two were proved
equivalent). T26 then extracted the `is_historical(pay, valuation_date)` atom when the exposure engine
pulled it, so all five engines plus the exposure grid route through **one** definition and a divergence
is no longer expressible — the upgrade earned by product-pull, not done for tidiness. C12 (a
survival/discount `TimeMeasure` guard, T25) is likewise structural.

**C9, C10, C11 and C12 are guarded *structurally*, which is the better outcome where it is reachable.**
None rests on a comparison test; each rests on there being exactly one way to express the thing — one
resolution function, one engine signature, one `is_historical` atom, one shared clock. A shared atom
beats a comparison test, because a test can only detect a divergence after it exists. **C10 joined them
at T26** by exactly this route — the atom extracted when the exposure engine first needed it, product-pull
rather than a tranche built for the guard.

**C7 and C8 remain OWED and remain *unviolable*.** The library still has one day count (`ACT/365F`),
so the convention cannot yet be broken, and building a guard for it would be §6's imagined problem.
The rows exist so that the day a second day count lands, this is found by reading rather than by a
wrong number. Tranche 23 is worth noting against them: it closed a *degenerate-fixture* hole in the
linker accrual (`ψ` invisible because every fixture was annual-ACT/365F) — which is adjacent to C8 but
not C8. C8 is about two conventions *disagreeing*; T23 was about one convention being *untested*. The
distinction is why C8 stays OWED rather than being quietly marked guarded by T23's work.

**No gate is proposed here.** The ruling made this a living doc, and the §4 clause promotes to law on
a second cross-module convention bug — in *this* tree, of which there has been none. The honest action
is to record the exposure, which is what the Guard column does.

## Maintenance

Add a row when a convention starts being shared by a second module — that is the moment it stops
being one module's business. Move a row to **guarded** only when a test *compares the modules*;
passing per-module tests do not qualify, which is the whole point.
