# Checkpoint CP-1 (complete) — measure oracle + first L6 vertical + consolidated XVA report

Date: 2026-07-16   ·   Version: `0.39.0`   ·   Tests: 230 green (`verify.py all`)

First checkpoint under the new cadence (`redesign/11`). Covers the three slices since the A6
rulings (`rulings_A6_measure_and_L6.md`, v0.37.0) — the **full CP-1 cluster** (A6.2). (An earlier
partial version of this doc was cut at 2/3; `xva_report` has since landed, so this supersedes it.)
No forward slice begins until this is ruled. Carries the four mandatory review inputs (§2–§5).

---

## 1. Slices landed (since A6, v0.37.0)

| Slice | Version | Layer | What |
|---|---|---|---|
| measure-consistency-oracle | 0.37.1 | L5 | A6.1 binding oracle: risk-neutral joint paths reproduce the forward-measure EE/PFE per date (change of numeraire). No behaviour change. |
| l6-trade-lifecycle | 0.38.0 | L6 | A6.2: `Trade.mark` + `Book.value`; realized (benefit table) + mark reconcile over a bond's life; dirty = clean + accrued; book linearity. |
| xva-report | 0.39.0 | L6 | A6.2: `xva_report` — netting set simulated ONCE → CVA/DVA/BCVA/FVA/KVA/MVA + EE/PFE/EAD; single-trade == standalone, mirror hedge nets to 0. |

**Ledger deltas:** `risk/exposure.py` — `_simulate_netting_set` (portfolio value on shared paths),
`_simulate_swap_values` now delegates to it (byte-identical), public `netting_set_exposure` (one
pass → pair + PFE), + the `test_measure_consistency` oracle & A6.1 doc. `shell/booking.py` extended
the A3 stub (`Trade.mark`, `Book.value`, `_combine`). New `shell/xva_report.py` (`XvaReport`,
`XvaReportConfig`, `xva_report`). Design docs committed: spine **A6**, cadence artifact **#11**,
`CLAUDE.md §6`.

## 2. Oracle-quality audit

| Slice | Oracle | Class |
|---|---|---|
| measure-consistency | two MC engines agree because both match the **analytic** change of numeraire (`P(0,t)·EE = swaption strip`) + OU moments | **cross-check** (anchored to closed form; MC-vs-MC `rel=4%`) |
| l6-trade-lifecycle | mark at issue = `Σ cf·DF`; mid-life = `Σ future cf·DF`; accrued = half-coupon by hand; realized = `Σ paid`; book linear | **closed-form** |
| xva-report | single-trade set reproduces each standalone L5 value **exactly** (same draws); mirror hedge nets portfolio exposure to 0 | **cross-check** (reuses the v0.25–v0.34 closed-form/analytic oracles) + exact-reproduction |

No slice rests on self-consistency alone. Watch item: the measure oracle's `4%` is a **statistical**
MC-vs-MC tolerance, not exact (see §4.2).

## 3. Quarry-drawdown reconciliation — THE honest gap (unchanged)

- Denominator: **768** quarry modules (`python/pricebook/**`, non-init). New tree: **49** modules.
- **Formally "crossed" (quarry module superseded + emptied): ~0 / 768.** The L5→L6 run is *forward
  construction adapting concepts*, and the ng versions are simplified (flat-curve HW, single-curve,
  one hedging set), so they do not yet fully supersede their quarry counterparts. This matches
  Cowork's own note that "a periodic quarry-crossing reconciliation is due."
- **Proposed action (early, before CP-2's systematic drawdown):** a one-pass **reconciliation map**
  — each ng module → the quarry module(s) it supersedes → the gap to "quarry-deletable" (e.g.
  ng-HW needs a general curve before it supersedes quarry `models/hull_white`). Turns drawdown into
  a real progress bar and exposes where ng is a *partial* adaptation vs a full cross.

## 4. Design choices to challenge (push back before they harden)

1. **Deferred SQLite persistence out of the L6 slice.** A6.2 bundled `pnl_history` persistence; I
   split it into its own data-spine slice (avoid introducing the persistence *interface* mid-slice).
   Right call, or should persistence land with the shell?
2. **Measure oracle tolerance is statistical (`rel=4%`), not exact.** Binds the two engines to the
   `~1–2%` observed; a common-random-number version would be exact. Acceptable gate, or harden it?
3. **`xva_report` is swap-only and scoped to one netting set.** It takes `list[VanillaSwap]`, not a
   `Book` of arbitrary products — extracting swaps from a mixed book (an isinstance filter) and
   non-linear products (engine registry at L6) are deferred. Is `list[VanillaSwap]` the right input,
   or should it consume a `Book`/`BookedTrade` now?
4. **KVA capital in the report = the netting-set SA-CCR EAD runoff at ATM (mark 0).** Consistent
   with the standalone KVA, but the report's *exposure* is stochastic (MC) while its *capital* is the
   deterministic ATM runoff — two different exposure notions in one object. Unify (stochastic-mark
   capital across the set), or keep the regulatory ATM runoff for capital?

## 5. Smell + debt scan

- `Trade._cashflows` / `xva_report` both assume product shape (`.cashflows`; `VanillaSwap`). Not yet
  an isinstance ladder, but a **product-shape coupling** that will want a small protocol at the 2nd
  shape (rule of two). — *proposed: accept, revisit when a 2nd exposure-bearing product arrives.*
- `_combine` normalises `accrued` to `Money(0)` (never `None`) at trade/book level — benign shift. — *accept.*
- `Trade.realized` currency peek is awkward but correct. — *accept.*
- `netting_set_exposure` returns a 2-tuple `(ExposurePair, ExposureProfile)` — mild; a named
  `NettingSetExposure` type is overkill for one consumer (`xva_report`). — *accept (rule of two).*
- No suppressions; no `OPEN.md` (ng) additions; `verify debt` green.

## 6. Ready-for-next / named next checkpoint

CP-1 is **complete and green**. Proposed CP-2 (per the #11 forward map), for Cowork to sequence:
- **(a) Quarry-drawdown reconciliation map (§3)** — I'd argue *first*, so `v1.0 = quarry empty`
  becomes countable and we see where ng is partial.
- **(b) General-curve (bootstrapped) HW** — lift the flat-curve ceiling under the whole XVA stack.
- **(c) Data-spine slice** — the persistence interface + SQLite (`pnl_history`) deferred from L6.

**Requesting Cowork rulings on §3 (drawdown baseline + the map), §4 (the four design choices), and
the CP-2 sequence (a/b/c).**
