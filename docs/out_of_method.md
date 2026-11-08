# Out of method — capabilities our oracle discipline cannot currently verify

Charter §1. The deferral line splits in two. `to_go_deeper.md` holds routes that *would* pass §4's
**oracle gate** (oracle-able with our discipline, maybe after an acquisition) but are not pulled yet —
come-back-same-method debt. **This file is the other bin:** capabilities a held source offers that
**cannot pass the oracle gate as the method stands** — a result anchored only by a plot or a
stochastic twin-Monte-Carlo run, with no closed form, cross-check, or requirement-identity to pin it.

**This is not debt.** A `to_go_deeper` route is repayable by pulling it and writing the oracle we
already know how to write. An out-of-method route is *not* repayable that way: it would require a
deliberate **§4 extension** — a statistical/stochastic-oracle tier that accepts convergence-based
validation (a price within a twin-MC band, a convergence *rate*, a falsifiability test) as a named
green oracle. No such tier exists, and building one is itself a ruling, not a tranche. So every entry
here is a **candidate** for that extension, recorded honestly so the deferral is not mistaken for a
roadmap — ruled on only if interest ever pulls that way.

**Rules.** One section per worked source (by its `references.md` key). Each entry names: the
capability, *why* it fails the oracle gate (what the source's own validation actually is), and what a
§4 extension accepting it would have to commit to. An entry graduates to `to_go_deeper` **only** if a
closed-form, cross-check, or requirement-identity anchor is later found for it — at which point it was
never really out of method, and the move is noted. Nothing here is built under §4 as it stands.

The boundary is drawn per *capability*, not per chapter: a closed-form **identity** inside an
otherwise-numerical chapter (e.g. a static one-period KVA = `(h/(1+h))·ES(L◦)`) is oracle-able and belongs in
`to_go_deeper`, while the chapter's **numerical method** (the neural-net solver that computes it for a
realistic portfolio) belongs here. Read the split by what anchors the number.

---

## `CREPEY-2026` — *XVA Analysis* (the XVA spine arc: T25–…)

The elementary static/one-period spine is built and oracled (see `docs/xva_spine_arc.md`). The book's
**own contribution** — the cost-of-capital XVA computed by machine learning — is, by construction,
out of method: its validation throughout is *twin Monte Carlo*, and its results are *figures*
(ch X is even titled around "Validation or Falsifiability?"). The candidates:

1. **Neural-net (and quantile) regression for pathwise CVA/FVA/EC/KVA (ch IV–VII, XI §7).** The
   computational engine of the book: parameterize the conditional XVA by a neural net, train on
   simulated paths, validate by an *a posteriori twin Monte Carlo* procedure (§IV.2.A). **Fails the
   oracle gate** because the anchor is a second stochastic run, not a known value — exactly §4's
   "runs and looks right" writ large, and "no number off a plot" binds on its convergence diagrams.
   *A §4 extension accepting it* would have to commit to: a twin-MC band as a named oracle with its
   **resolution asserted** (the T17 banded-resolution rule — a green is evidence only about the
   defect size the band can resolve), and the network's training treated as a *procedure* oracled
   step-wise (the T18 rule — a price bound cannot see the algorithm that produced it). Heavy; a real
   method decision.
2. **The continuous-time anticipated-BSDE XVA framework (ch II §3, VII §1).** The dynamic
   cost-of-capital equations as anticipated BSDEs, solved numerically for any realistic portfolio.
   **Fails the oracle gate** for a realistic case (no closed form). *Partial graduation path:* the
   framework's **well-posedness and comparison identities** (ch II §3) and the static reductions
   (the one-period KVA = `(h/(1+h))·ES(L◦)`, eq 25; `FV = CVA − DVA`, eq 18) *are* requirement-checkable — so
   those specific identities belong in `to_go_deeper`, not here; only the general numerical BSDE
   solution is out of method.
3. **Risk-measure case studies (ch VIII–X): expected-shortfall economic capital, Radner-equilibrium
   default resolution, bottom-up reverse stress testing.** Case studies whose numbers come from MC
   over a market/credit model and are reported as figures and stress scenarios. **Fails the oracle
   gate** — no closed form or identity anchors the reported magnitudes; the "champion vs challenger /
   falsifiability" framing (ch X §4) is itself an admission that validation here is statistical, not
   exact. *A §4 extension* would be the same statistical-oracle tier as (1).

**The honest boundary for this arc:** everything in this list stays out until a statistical-oracle
tier is ruled; the oracle-able identities it leans on (the CVA/DVA/FVA closed forms and Crépey's
balance-sheet identities) are the *spine* and are built, and the oracle-able-but-unpulled routes
(static KVA, collateral/CSA, CDS/defaultable-bond calibration) are `to_go_deeper`. Fixed at the arc
checkpoint `CP_xva_spine.md`.
