# Proposed charter amendment — two-track arcs, audit-as-switch, and 1.0 as purpose

**Cowork, re-audit #9 (4) charter pass, 2026-11-26. A PROPOSAL for Bernardo's §1/§9 ruling — not yet charter.**
Nothing here is adopted until Bernardo rules; §1 and §9 are his and he may veto. Worked out in the
2026-11-24 direction conversation, brought here because re-audit #9 is the arc boundary §1 reserves for the
1.0 question ("the question returns when it is ripe, from a ruling, not a guess now").

## Why now
Ten literature arcs are complete. The library is a genuine library at the *pricer* level (one registry
dispatch, one `price` contract, a shared de-duplicated foundation) but a *catalog* above the pricers — the
§0-promised representation half (a **pricing request**, and a **position/portfolio**) was deferred every arc,
and there is no calendar/business-day engine and no greeks beyond a Black MC demo. The stated purpose has
not changed since the start: **a Python package that prices and risks a position, provider-agnostic, to
retire the expensive third-party black box.** The book-driven method served the pricing breadth well; it does
not, on its own, reach the desk-capability and representation layers, because its *selector* is the bookshelf.

## The amendment, in three parts

### (A) Two arc TYPES, one gate
The oracle gate and the arc/tranche discipline are unchanged — they are the method's asset and they are
selector-agnostic. What is added is a second *kind* of arc:

- **Literature arcs** (the current kind) — book-grounded; add pricing capability for a model or asset class;
  oracle = the held source's closed forms. Unchanged.
- **Structural arcs** (new) — library-level, cross-cutting, pluggable with any asset/model: representation,
  mark-to-market, greeks, scenario/stress, P&L attribution, lifecycle, XVA/collateral on a position,
  time-series/estimation. **Architecture-first, not transcription-first.** Oracle = the *existing verified
  catalogue* (analytic greek as the exact target; FD → analytic at a measured rate, the T25 detector;
  scenario = reprice under a shocked snapshot; P&L = a decomposition identity against a full reval). The
  rigour does not drop — it anchors against the catalogue instead of a book. The time-series/VaR end will
  press on the **statistical-oracle tier** §1/`out_of_method` hold open; that tier becomes a real §4 ruling
  when a structural arc reaches it, not before.

### (B) The re-audit is the track switch
The whole-tree re-audit already assesses the tree at each arc close. It gains one question — *which track
next?* — defaulting to a swing (literature ⇄ structural) but free to repeat a track when the coverage map
says one side is starving. No new cadence machinery; the existing switch point does double duty.

### (C) 1.0 is a PURPOSE, not a module count
§1 says 1.0 "will not be a module count" and returns "from a ruling when ripe." Proposed finish line:

> **1.0 = the library can represent, value, and risk a *position* — price + greeks + calibration +
> scenario/P&L — for the asset classes a target desk trades, ingesting any market provider through the
> representation contract.**

Not "N products." Provider-agnosticism is already met by design (§0). The open columns are representation,
greeks, and book-level risk — which the structural track fills. This replaces the open-ended "grows by
interest" with a *destination* for the interest to grow toward, without reintroducing the old product-list
sprawl (the gate and arc-closing still stop infinite listing).

## What does NOT change
- The oracle gate, the two-bin deferral (§1), the layer spine (§2), the stateless engine contract (§3), the
  §4 oracle discipline, the §10 Cowork/Code split, every gate in §9. Structural arcs pass the same gates.
- Depth-by-interest within a track. The swing sets *which* track; interest still picks the unit inside it.

## First move if adopted
The first structural arc is **R1 — representation & mark-to-market** (`docs/arcs/R1_representation_mtm.md`,
drafted alongside this). FX is the literature arc after R1 — the test that the representation is
currency-general. The sequence: re-audit #9 remediation → (this ruling) → R1 → re-audit #10 → R2 (greeks) →
re-audit #11 → swing (FX or interest).

## Charter staleness scan (the rest of pass (4)) — clean, with one standing watch
No clause is stale or in tension after the vol arc: the arc's two method events filed as **instances of
existing clauses** (T34 for the `ρ` price-insensitivity, T43-read-generally for the carry-zeroing), zero new
clauses — the charter covered new cases rather than growing, as at every capstone since the equity arc. The
one standing watch is **size**: §4 in particular is very large, and this amendment would add §1/§9 text. If
adopted, a condensation pass on §4 (merging the degenerate-fixture catalogue's now-numerous instances into
their named families) is worth its own tooling/docs tranche — recorded, not done here.
