# Checkpoint — the XVA spine arc (project three, `CREPEY-2026`)

**Closed 2026-11-10 at v0.35.0.** The third arc: the elementary XVA vertical, from Crépey's *XVA
Analysis*, scoped to the oracle-able static/one-period spine and honest about the frontier it leaves.
Same shape as the inflation (ch 1–5) and Jäckel (9-of-14) closes — build the spine, ledger the rest,
write it down. Arc plan: `docs/xva_spine_arc.md`.

## What was built — CVA · DVA · FVA · collateral
The vertical the archived predecessor **faked** (phantom `risk/xva.py`, pruned by T24) now exists, oracled:

| tranche | version | what | layer |
|---|---|---|---|
| T25 | v0.30.0 | survival curve + unilateral CVA (closed-form EPE, Bachelier) | L1 `market/` + L5 `risk/` |
| T26 | v0.31.0 | the Monte-Carlo exposure engine (EPE/ENE by simulation over a real swap) | L5 |
| T27 | v0.33.0 | bilateral CVA + DVA (both parties default); `credit_adjustment` extracted (rule-of-two) | L5 |
| T28 | v0.34.0 | FVA — Crépey's semilinear fixed point, priced by its closed form | L5 |
| T29 | v0.35.0 | collateral / CSA — the collateralized exposure over the margin period of risk | L5 |

Plus **T26b** (v0.32.0) — the `verify.py references` governance gate, built mid-arc (see Governance).
L5 is now a substantial layer; the CVA/DVA/FVA functionals price any `ExposureProfile`, so the
unilateral / bilateral / collateralized cases differ only in the exposure handed to one shared integral.

## The method advances this arc produced (charter accretions)
- **§4 — "a reference is code too", caught by the convergence *rate*** (T25): an EPE quadrature wrong in
  a regime surfaced as a CVA *plateau*, not a value — a flat band would have swallowed it.
- **§4 — the distributional-symmetry degenerate fixture** (T29, the trap's seventh face): a driftless
  mark made an exposure gap symmetric, so reversing its *sign* (the wrong direction of risk) passed all
  986 tests. The guard is "break the symmetry the mutation exploits."
- **§4 — the fixed-point oracle shape** (T28): the closed form must *satisfy its own fixed point*; the
  iteration is a telescoping guard with a bounded domain (it diverges for a funding spread ≥ 100%, which
  the closed form does not — the closed form is *strictly stronger*, not merely convenient).
- **§5 — "our own docs are not an authority either"** (the KVA eq-25 reflow near-miss): a formula parked
  in a ruling is re-transcribed from the source at build, never copied forward.
- **§1 — the three-gate spine test + the `to_go_deeper` / `out_of_method` two-bin split**, which this
  arc forced: Crépey is mostly *out of method*, and the cut needed to be a test, not taste.

## Crépey, handled honestly (a research/ML source, a stretch for the method)
The book's own contribution — cost-of-capital XVA by neural-net regression, validated by twin Monte
Carlo — is **out of method** (`out_of_method.md`). The spine used Crépey for the **identities as
requirement oracles** (`FV = CVA − DVA` eq 18; the FVA fixed point eq 19) and `CAPINSKI-ZASTAWNIAK`
for the credit closed forms. Findings worth the record:
- **The KVA eq-25 reflow** — `(h/(1+h))` split across lines; caught before it reached a `to_go_deeper`
  anchor by the dual rendering and the printed bound.
- **The eq-18 convention, measured not argued** (T27): first-to-default conditioning worth 2.68% of CVA,
  Crépey's `(1−γ)⁻¹` fininsurance normalisation a further 4.71% — ruled the standard risk-neutral form
  ("half a measure is no measure"), the 4.71% recorded in `to_go_deeper`.
- **The FVA contraction error** (T28) was *Cowork's*, in the spec, and execution caught it — §10 working.

## Honest scope — what the spine does NOT do
- **`to_go_deeper` (oracle-able, unpulled):** the amortising exposure *hump* over a swap's life (route 1,
  the deflated-value martingale); Crépey's `(1−γ)⁻¹` normalisation (route 2, worth 4.71% of CVA);
  thresholds/MTA, initial margin & MVA, ColVA, CDS/bond calibration of the survival curve, Merton
  structural default.
- **`out_of_method` (needs a §4 statistical-oracle extension):** the neural-net/quantile regression
  chapters (IV–VII), the anticipated-BSDE dynamic XVA, KVA, and the risk-measure / equilibrium /
  reverse-stress-test chapters (VIII–X). The cost-of-capital KVA — the book's actual thesis — is not in
  the spine by design.
- **No wrong-way risk** throughout: exposure/default independence holds, which is the simplification WWR
  would recover.

## The two-surface split, measured
Both directions fired. **Cowork → Code:** the independent-anchor ruling (T26's martingale, which a
mutation proved was the *only* test that could see a numéraire error); the standard-measure convention
(T27); the fixed-point oracle ordering (T28). **Code → Cowork, refuting a Cowork claim (§10):** the FVA
contraction factor was `γ`, not the spec's `γ/(1+γ)` (T28); the references-status drifts (T26b, the gate
catching two the hand passes missed); the re-audit cadence count (eight, not eleven). Neither surface is
the reviewer of its own work.

## Governance advances this arc
- **`verify.py references`** (T26b) — the register's status field can no longer drift from what `src/`
  cites; five historical drifts earned it, two found only by the gate's first run.
- **§9 — the floor gate's own discovery must be total** (T26b): a new test dir ran unfloored until fixed.
- **The Substack pipeline** (`_substack/`, gitignored) — the journal, its capture discipline (a beat
  appended at every step-4 review), and the two-role manual for a separate Publications-Cowork session.

## Close
**Builds complete: T25 → T29.** Next is **re-audit #2** — fired by this arc close (§9), run full-tree
(T1–T29, *not* just the arc built, per §9's own signature-defect lesson) and independently of the
builder (Code built every tranche). Findings → `AUDIT_FINDINGS`, ruled by Cowork, remediated, then
project four (Bernardo's next source). The cadence counter is unaffected — eight tranches since
re-audit #1, re-arming at ≈ T33 or the next arc close.
