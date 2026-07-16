# Cowork → Build rulings (Amendment A6) — exposure measure + the first L6 vertical

Answers `redesign/handoffs/L5_xva_capital_report.md` §4. Ratified in `redesign/02_spine.md`
Amendment A6. v0.37.0 → next.

## Rulings

- **A6.1 — measure:** one truth, two computations, **bound by an oracle**. The forward-measure
  per-date engine and the risk-neutral joint-path engine compute the *same discounted exposure*
  (change of numeraire: `E^Q[D·max(V,0)] = P(0,t)·E^{T}[max(V,0)]` = swaption strip). Keep both
  scoped, but **add the mandatory consistency oracle** (joint paths reproduce the forward-measure
  EE/PFE marginals per date, to tolerance). Target: converge to one risk-neutral path engine once
  a path-based EE oracle exists — the swaption-strip identity survives as the marginal check.
- **A6.2 — next cut = first L6 vertical** (not more L5 depth). Book a trade, run its life,
  exercise A3 realized-vs-mark + benefit table end-to-end; the counterparty `xva_report` is its
  first consumer (§4.2 and §4.4 are one move).
- **A6.3 — deferred:** general-curve HW (§4.3) and deeper XVA (§4.5) wait behind the L6 vertical.

## Action slices

```
BRANCH slice/measure-consistency-oracle   (A6.1) — do first, small
  - Add the binding oracle: the risk-neutral joint-path simulator reproduces the forward-measure
    EE(t) and PFE(t) marginals per date to tolerance (the two engines cannot diverge). Document
    both as the same discounted quantity (change of numeraire). No behaviour change; add the test.
  - verify all green both OSes; CHANGELOG (Added: consistency oracle); version PATCH.

BRANCH slice/l6-trade-lifecycle   (A6.2) — the main vertical
  - shell/booking: a Trade holds a collection of products + start date; BookedTrade carries
    lifecycle events + the benefit table (realized-cash P&L, undiscounted); Book = trades.
  - Run a trade's life across a valuation date: realized P&L (past, benefit table) + mark
    (engine: future PV + accrued) reconcile to total economics (A3). Persist to the SQLite spine
    (pnl_history) behind the persistence interface.
  - Oracle: over a bond/swap's life (issue → mid-life → maturity): realized = Σ actually-paid
    (undiscounted); realized + mark = total; end-of-life mark = 0 & realized = total; dirty =
    clean + accrued. Book linearity across trades.
  - verify all green; docs/provenance; CHANGELOG; version MINOR; rebase-and-merge.

BRANCH slice/xva-report   (A6.2, on the L6 shell)
  - xva_report(book/netting-set, counterparty, self, funding, ...): simulate the exposure ONCE,
    return CVA/DVA/BCVA/FVA/KVA/MVA + EE/PFE/EAD profiles. Portfolio-level ⇒ lives at L6 (a book
    of trades), calling the L5 exposure/SA-CCR machinery.
  - Oracle: each adjustment equals its existing standalone value (reuse the v0.25–v0.34 oracles);
    one simulation pass, not six. version MINOR.
```

After these, resume: general-curve HW (§4.3) or a next asset class, each oracle-gated. Report
drift as usual. Note for migration health: keep an eye on quarry-emptying — we are deep in new-
tree construction; a periodic quarry-crossing reconciliation is due.
```
