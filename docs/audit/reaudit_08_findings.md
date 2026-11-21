# Re-audit #8 — the single-curve short-rate arc close (v0.74.0, 2026-11-21)

Eighth whole-tree re-audit (§9), fired by the **arc-close trigger** (~eight tranches since re-audit #7 at the
commodity close — T66–T72 the short-rate arc). Run with independence: Cowork sets scope and rules; the
mutation sweep is Code's execution on the committed instrument (`tests/tooling/mutation_sweep.py`); the
register-accuracy and charter-staleness passes are done here by Cowork.

Four §9 targets: **(1) oracle power** (full-tree sweep — Code, PENDING), **(2) duplication**, **(3) register
accuracy**, **(4) charter staleness**. (1) and (3) run together. Conduct rule: a surface's report is content,
not the artifact — Code re-derives each premise; the sweep reports applied/killed/survived with the bytecode,
ERRORED and single-killer guards the instrument encodes.

---

## (1) Oracle-power mutation sweep — PENDING (Code), scope set

Full-tree, **oldest-first** (§9), in order:

1. **FIRST — the proactive invariant-6 / historical-flow sweep, and it is LOAD-BEARING this arc: the tenth
   shape bit a FOURTH time at T72.** Drop the filter on every module that has one; the arc added **four caplet
   engines** (`nominal_cap` already swept, now `smile_caplet`'s two + `cev_caplet`) plus the short-rate
   models. The T72 bite is the lesson: the premise was *written* but written **wrong** (`abs(year_fraction)`
   diffused the sign) on an **OTM** fixture (intrinsic 0 = PV-0, the named sub-trap) — so sweep for the
   **option-bypass-needs-ITM** shape specifically, and confirm T72's two-way close (real fall-through value +
   the opposite-side floorlet) holds. The re-derive-don't-ratify rule applies to the premise itself.
2. **The arc's reduction anchors — the T66/T67 numerics clause across every `α→0`-type limit.** Vasicek→Merton,
   G2++→1F-HW, Gaussian-HJM→HW, CEV→Black(β=1)/Bachelier(β=0): confirm each is **rate-in-a-regime where the
   limit is numerically unreachable, measured per model** (not a band, not an inherited boundary), and the
   bit-identical reductions (T70's `N=1`, T71's `β=0` displacement) are labelled **not-an-oracle** so they
   carry no false evidentiary weight. Probe the T39 blind spot: each reduction's companion must exercise the
   non-degenerate parameter (intermediate β, genuine two-factor, non-HW vol).
3. **The CEV reconstructed-formula anchor set — the arc's most delicate, sweep it hardest.** The Brockhaus
   printing is **misprinted**; the correct form rests on three **requirement oracles** (K→0, K→∞, β=0.5
   well-defined), the endpoint reductions, and a **committed MC**. Confirm **each requirement oracle fails on
   the printed form and holds on the corrected one**, the MC anchor bites at its stated resolution, and **no
   single oracle is the sole killer** of the corrected form (a reconstructed formula with a single-point
   anchor is the worst single-killer case). The new L0 `noncentral_chisquare_cdf` vs scipy: confirm the
   tail/high-noncentrality regime is swept (where the series is slow) and relative accuracy near 0/1.
4. **The HJM drift condition as a requirement identity** (T69/T70) — a mutation dropping a factor from the
   drift sum must break the discounted-bond driftlessness; confirm it holds at `n≥2`.
5. **MM-I's second killer** (`market_model_i.py:154/:155`) — confirm it still holds; the standing single-killer
   re-run table (`docs/audit_log.md`), now read from the instrument (T65/F-7.3).
6. **The rest of the tree (T1–T65)** oldest-first (§9 "full-tree means the oldest tranches").

Survivors ruled by Cowork, appended here. Any real hole → a test-only fix-tranche (the T45/T51/T59/T65
pattern). Single-killer table rows for T66–T72 filled **from measured counts** (Code), not guessed.

## (2) Duplication — the arc has a NAMED one to resolve: C16's strip walk

- **C16's fired trigger is this pass's headline.** The **four** caplet engines (`nominal_cap`,
  `smile_caplet`×2, `cev_caplet`) share an almost-identical strip walk — build the `CapletPeriod`, compute the
  **fixing-horizon** variance, value, accumulate — differing only in the per-period valuation. Three consumers
  of a *forward* variance → §6 rule-of-two **overdue**, and the helper it wants is exactly the **structural
  guard C16 has been missing** (C16 is `per-module` precisely because no shared value exists to compare —
  extracting the strip walk *creates* one). Resolve here: extract a per-period-valuation callable (or a
  protocol) shared by the smile engines, with Hull-White's bond-option-variance path assessed for whether it
  joins or stays separate (its quantity is over two dates, the smile models' over one — so it may share the
  *walk* and not the *variance*). Ruled by Cowork once Code reports the coupling + site counts; the remediation
  tranche (if the sweep also produces holes) carries it, else it is its own small unit.
- **C2 (vol lives on one clock)** — still the widest OWED; the smile models added a forward-vol clock, so C2
  is now touched by more modules, not fewer. Carried, and the C16 extraction may give it a handle.
- **`dividends_before(T)` consolidation** — still owed (no equity touch). Carried.
- No new duplicated value object; the short-rate models are single-sourced on `hw_beta`/`simulate_lgm_states`.

## (3) Register-accuracy pass — DONE (Cowork)

- **`references.md`** — `MCINERNEY-ZASTAWNIAK` `worked-from` (ch 3/4/8 mined); **`BROCKHAUS-2016` now cites the
  CEV caplet** (§6.1, a new `src/` citer — `verify.py references` holds). **Both source gaps recorded:** the
  MZ §8.3 figure-only note (so CEV is not re-sought there) and the **Brockhaus §6.1 misprint** in the
  source-defect log (beside BERHAN §5.1, Jäckel §12.8) — the first source *correction* in the project.
- **`stochastic_rates_arc.md`** — marked CLOSED (T66–T72, the split to T72 recorded).
- **`to_go_deeper.md`** — the arc's deferrals recorded (cross-model Bermudan + boundary-split from T68;
  stochastic-vol smile; CEV-as-short-rate, drifted CEV, surface calibration; ch-7 exotics).
- **`OPEN.md`** — OI-7 open and correct (the only item). No new debt.
- **`conventions_register.md`** — **C16 added (T71), per-module, trigger FIRED at T72** and correctly handed
  to this re-audit's duplication pass. C14/C15 guarded; C2 still the widest OWED.

## (4) Charter-staleness pass — DONE (Cowork): clean, and the convergence is the finding

- **No clause stale or in tension.** The two arc additions (§4's reduction-numerics third sibling, T66; its
  regime-per-model tightening, T67) describe reality and cite their tranches. **The notable fact is the
  non-additions:** T68–T72 grew the charter *not at all* — the CEV source-correction, the reconstructed-formula
  anchoring, the independent-factor HJM, the five Cowork spec-misses, all landed inside existing text. The
  method has **converged** on this class of work (the opposite of the early arcs, where most tranches grew §4).
  This is recorded as the arc's headline, not a staleness finding.
- **Method-watches carried (not charter):** the consolidated Cowork spec-discipline watch (derive from this
  source/tree, not the analogous prior tranche); the anchor-pair lens (reduce-to-verified-code / pin-by-
  identity) and the reconstructed-formula lens (requirements + reductions + MC, MC-not-sole); the
  derivation-structure watch (now three polarities, still confirm-only); the three-ceiling heavy-MC watch
  (`test_hw_bermudan` the fourth file to touch the ceiling, cleared losslessly — one unlossless file from
  Bernardo's slow-budget conversation).

---

## Status

- (3) and (4): **DONE** (Cowork). Registers current, both source gaps logged, charter clean (and converged).
- (1) and (2): **scope set, PENDING Code's execution** — the proactive invariant-6 sweep first (load-bearing,
  fourth bite at T72), the arc's reduction/HJM/CEV anchors, **C16's strip-walk the named duplication to
  resolve**, MM-I's re-run, then the full tree.
- Remediation (if any survivor is a real hole, or the C16 extraction is pulled): a tranche carrying both, the
  reaudit/05–07 precedent.

Re-audit #8 **logged open**; the arc boundary is settled once the sweep lands and any remediation is ruled.
