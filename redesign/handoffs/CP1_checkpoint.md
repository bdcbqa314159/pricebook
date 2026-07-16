# Checkpoint CP-1 (partial) — measure oracle + first L6 vertical

Date: 2026-07-16   ·   Version: `0.38.0`   ·   Tests: 228 green (`verify.py all`)

First checkpoint under the new cadence (`redesign/11`). Covers the two slices since the A6
rulings (`rulings_A6_measure_and_L6.md`, v0.37.0). Called **at the user's request, mid-cluster**:
CP-1 is a 3-slice cluster and slice 3 (`xva_report`) is **not yet done** — deliberately handed to
Cowork here before it, since it's the first thing that spans L5↔L6. Carries the four mandatory
review inputs (§2–§5).

---

## 1. Slices landed (since A6, v0.37.0)

| Slice | Version | Layer | What |
|---|---|---|---|
| measure-consistency-oracle | 0.37.1 | L5 | A6.1 binding oracle: risk-neutral joint paths reproduce the forward-measure EE/PFE per date (change of numeraire). No behaviour change. |
| l6-trade-lifecycle | 0.38.0 | L6 | A6.2: `Trade.mark` + `Book.value`; realized (benefit table) + mark reconcile over a bond's life; dirty = clean + accrued; book linearity. |

**Ledger deltas:** `risk/exposure.py` gained `_simulate_rate_paths` doc ratification (A6.1) + the
`test_measure_consistency` oracle. `shell/booking.py` extended the A3 stub with `Trade.mark`,
`Book.value`, and a shared `_combine`; `BookedTrade.value` now delegates. Design docs committed:
spine **A6**, cadence artifact **#11**, `CLAUDE.md §6`.

## 2. Oracle-quality audit

| Slice | Oracle | Class |
|---|---|---|
| measure-consistency | two MC engines agree because both match the **analytic** change of numeraire (`P(0,t)·EE = swaption strip`) + OU moments | **cross-check** (anchored to closed form; MC-vs-MC to `rel=4%`) |
| l6-trade-lifecycle | mark at issue = `Σ cf·DF`; mid-life = `Σ future cf·DF`; accrued = half-coupon by hand; realized = `Σ paid`; book linear | **closed-form** |

No slice rests on self-consistency alone. One thing to note: the measure oracle's tolerance is a
**statistical** `4%` (independent MC draws) rather than exact — see §4.

## 3. Quarry-drawdown reconciliation — THE honest gap

- Denominator confirmed: **768** quarry modules (`python/pricebook/**`, non-init).
- New tree: **48** modules built (L0–L6).
- **Formally "crossed" (quarry module superseded + emptied): ~0.** The L5→L6 run was *forward
  construction adapting concepts*, not systematic quarry emptying — and the ng versions are often
  simplified (flat-curve HW, single-curve, one hedging set), so they do not yet fully supersede
  their quarry counterparts. **Drawdown ≈ 0 / 768.** This matches Cowork's own note that "a
  periodic quarry-crossing reconciliation is due."
- **Proposed action (early, before CP-2's systematic drawdown):** a one-pass **reconciliation map**
  — for each ng module, the quarry module(s) it supersedes and the gap to "quarry-deletable"
  (e.g. ng-HW needs general curve before it supersedes quarry `models/hull_white`). That map turns
  drawdown into a real progress bar and exposes where ng is a *partial* adaptation vs a full cross.

## 4. Design choices to challenge (push back before they harden)

1. **Deferred SQLite persistence out of the L6 slice.** The A6.2 ruling bundled `pnl_history`
   persistence; I split it into its own data-spine slice to keep the vertical checkable in one pass
   and avoid introducing the persistence *interface* (a cross-cutting abstraction) mid-slice.
   Right call, or should persistence land with the shell?
2. **Measure oracle tolerance is statistical (`rel=4%`), not exact.** It binds the two engines to
   `~1–2%` observed, but a common-random-number or analytic-marginal version would be exact. Is 4%
   an acceptable "binding" gate, or do we harden it?
3. **Checkpointing mid-cluster (2 of 3).** `xva_report` deferred to the next batch at user request.
   Fine, or complete the cluster first?
4. **`Trade.mark` is linear-only.** It binds `DiscountingModel`/`DiscountingEngine`; a trade of
   non-linear products (swaptions, options) needs per-product engine selection (the L4 registry) at
   the L6 mark. When do we introduce that — with `xva_report`, or a dedicated engine-registry slice?

## 5. Smell + debt scan

- `Trade._cashflows` assumes `product.cashflows` (works for bonds/fixed legs; float legs need
  fixings). Not yet an isinstance ladder, but a **cashflow-shape coupling** that will want a small
  protocol once a 2nd cashflow shape arrives (rule of two). — *proposed: accept, revisit at seasoned-float.*
- `Trade.realized` currency peek (`next(self._cashflows())` on the empty-paid branch) is awkward. —
  *proposed: accept (works); tidy if a Money-sum helper lands.*
- `_combine` normalises `accrued` to `Money(0)` (never `None`) at trade/book level. Benign, but a
  minor semantic shift vs single-product `PricingResult(accrued=None)`. — *proposed: accept.*
- `BookedTrade` is mutable (appends observed marks) — intentional (the shell is stateful). — *note only.*
- No suppressions, no `OPEN.md` (ng) additions, `verify debt` green.

## 6. Ready-for-next / named next checkpoint

- **Immediate:** complete CP-1 with `slice/xva-report` (A6.2) — per-counterparty over a **book**
  (L6), simulate exposure **once** → CVA/DVA/BCVA/FVA/KVA/MVA + EE/PFE/EAD; oracle = each equals its
  v0.25–v0.34 standalone value from one pass.
- **Then CP-2** (per the #11 forward map): general-curve (bootstrapped) HW — lift the flat-curve
  ceiling — *and/or* the quarry-drawdown reconciliation map (§3), which I'd argue should come first
  so v1.0 becomes countable.

**Requesting Cowork rulings on §3 (drawdown baseline), §4 (the four choices), and the sequence of
§6** (xva_report vs drawdown-map first).
