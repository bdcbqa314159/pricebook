# Checkpoint — the credit-derivatives arc (project five, `OKANE-2008`)

**Closed 2026-11-16 at `v0.52.0`, 1780 tests, eleven gates green.** The fifth vertical and the second
capstone — single-name credit (the CDS, its curve, the option on it) and then the multi-name
correlation core (the Gaussian copula, baskets, tranches, the exact loss distribution, base
correlation). The full spine was committed by Bernardo on 2026-11-14 and built straight through.

## What was built — single-name, then multi-name

**Phase 1 — the CDS vertical.**
- **T38 — the credit default swap.** O'Kane's premium + protection legs, par spread, RPV01, and the
  mark-to-market. Anchored by the **credit triangle** `S = λ(1−R)` (exact in the flat-hazard limit) and
  by reconciling the protection leg against the **T25 CVA integral** — a cross-arc, different-code-each-
  side oracle. `v0.45.0`.
- **T39 — the CDS survival-curve bootstrap.** A par strip inverted for the piecewise-flat hazard curve,
  an L1 construction (it *produces* market data, §2's calibration split). Anchored by the **known-curve
  round trip** (bootstrap, then reprice every CDS to par 0) — T8's pattern, the one that caught BERHAN's
  misprints. The leg arithmetic was lifted to L1 so the bootstrap and the pricer share one copy.
  `v0.46.0`.
- **T40 — forward CDS + the knockout default swaption.** Black on the forward spread under the
  risky-annuity measure, with the forward-start → spot reduction and payer/receiver put-call parity.
  Reused `models/black` (T5). `v0.47.0`.

**Phase 2 — the correlation products.**
- **T41 — the one-factor Gaussian copula.** Conditional independence through one market factor; the
  conditional default probability closed form, the `ρ→0` reduction to independent defaults. Reused
  `Φ`/`norm_cdf`. `v0.48.0`.
- **T41b — `ReferenceEntity` keying.** Infrastructure the multi-name products forced: survival curves
  re-keyed by entity so N names in one currency are representable. Earned a duplicate-`C11` finding in
  the conventions register. `v0.48.1`.
- **T41c — `verify.py conventions`.** The register-integrity gate (unique ids + resolving citations),
  earned by T41b's duplicate id that two review passes — including a Cowork by-name ratification —
  missed. Statically decidable, so gateable (§9). `v0.48.2`.
- **T42 — the first-to-default basket.** The FtD spread through the copula — the first product priced
  off the latent-variable model. Reductions to a single-name CDS (basket size → 1) and to the
  independent closed form at `ρ=0`; an independent Jäckel-engine MC within SE. `v0.49.0`.
- **T42b — the gates-gate widened.** `verify.py gates` extended to the standalone CI gates
  (`ruff`/`pyright`), which sat outside the T38 gates-gate's domain and could be dropped silently.
  Earned by `pyright` catching a defect a local run missed. `v0.49.1`.
- **T43 — the LHP synthetic-CDO tranche.** The large-homogeneous-pool tranche — the correlation
  market's closed form in `Φ`/`Φ₂`. The LHP formula transcribed pointwise, with the equity-tranche and
  `ρ→0` reductions and the tranche survival curve. `v0.50.0`.
- **T44a — the exact loss distribution by recursion.** The conditional loss recursion, with the **LHP as
  its `O(1/N)` limit** — a measured two-method companion, the loop between the approximation and the
  exact object closing. `v0.51.0`.
- **T44b — base correlation.** The market's quoting convention: a base-correlation curve bootstrapped
  from tranche quotes by inverting the T43 LHP engine (an engine-inverting calibrator → L4), and any
  `[K₁,K₂]` tranche priced as the difference of two base equity tranches. The arc's last build.
  `v0.52.0`.

## What the arc reused — the map that makes it the second capstone
The arc spent its new effort on the genuinely new credit objects and *connected* four arcs' worth of
machinery. The `SurvivalCurve` and the discount curve (T25/T1/T8) underlie every credit product; the
**protection leg is the T25 CVA integral** in different clothing (the T38 cross-product oracle);
`models/black` (T5) anchored the swaption; **`Φ`/`Φ₂` (T36b)** — built for wrong-way risk one arc
earlier — are the copula and LHP apparatus; the **Jäckel MC engine (T15–T22)** is the basket's
independent check. Project five is where the survival curve, the CVA integral, Black, and the Gaussian
machinery meet.

## Method advances this arc produced (charter accretions)
- **§4 — the reduction-anchor blind spot (T39).** A companion oracle must be non-degenerate in the
  *specific* parameter the reduction zeroes, because an exact reduction is often exact *because* it
  degenerates the very parameter a discretisation error lives in — so the anchor is structurally blind
  to that error class. (The flat-hazard credit-triangle anchor cannot see a first-vs-second-order
  protection-leg integration defect; a term-structured companion measured by its O(1/M) vs O(1/M²) rate
  catches it.)
- **§4 — the scale-invariance oracle-set blind spot (T43).** If every oracle in a set is
  scale-invariant — a par-reprice, a ratio, an ordering, a linearity — they pin the *shape* and say
  nothing about the *level*, so a uniform rescaling of a leg survives all of them at once; add one
  oracle asserting an absolute magnitude. (A doubled `(1−R)` — the protection leg 40% off — survived the
  par-reprice, the spread-linearity and the equity/mezz/senior ordering together.)
- **§4 — the tenth degenerate-fixture shape, and its escalation (T40/T42/T44b).** A *cross-tranche
  fixture convention* — premium schedules that begin at the valuation date, so no fixture produces a
  historical flow and the invariant-6 filter has nothing to filter. It **bit a third time** (T40
  shipped a 73% annuity overstatement, T42 and T44b were caught), which fired the charter flag: this
  degeneracy now **joins `τ=1` as one the periodic re-audit proactively sweeps for**, and re-audit #4 is
  its first run. The third bite also produced a rule one level in from §5's "our docs are not an
  authority": **a surface's report that it DID something is content, not the artifact; the reviewing
  surface re-derives a load-bearing premise, never ratifies on the recollection.** (Cowork nearly
  ratified #62 on Code's report that all three fixtures straddled; T44b's did not.)
- **§9 — the CI-liveness gate (T38) and its two extensions.** "A gate named in the charter is not live
  until `ci.yml` invokes it" became **`verify.py gates`**, which covers itself; T41c added
  **`verify.py conventions`** (register integrity); T42b **widened the gates-gate** to the standalone
  `ruff`/`pyright` steps. Three infrastructure tranches, each earned by a real drift, none speculative.
- **§9 — local re-time is not the measurement, CI is (T44a).** For a numerically-heavy file the slow
  budget is sized for the CI runners; a comfortable local reading is not evidence the gate passes there.
  (Broken twice — T42 and T44a — then again at T44b one tranche after the line went in, which is the
  rule catching its author inside one tranche.)
- **§3.5 — config lives on the model (T44a).** A numerical-config value belongs to the model whose
  evaluation cost determines it, not inherited from a sibling: a base-correlation tranche costs two `Φ₂`
  per step (one model per leg of the difference), so its step count is its own, not the LHP's.

## The lessons that did NOT grow the charter — the healthy outcome
- **Three questions dissolved rather than needing rulings** — the T43 index handle, T44b's interpolation
  convention, and `HomogeneousPool`'s shape — each had *no consumer in scope*, so the right move was to
  refuse the knob (§6/§3.4), not invent a convention. A well-posed spine does not accumulate convention
  knobs; this arc's signature is three of them declining to exist.
- **The vacuous-oracle restatement (T44b)** — Cowork's no-arbitrage-monotonicity oracle was vacuous
  (a positive quote makes monotonicity hold by construction), and the fix was ruled an *instance* of the
  existing "an oracle that cannot fail is empty" + "make the suite state what it does/does not cover"
  clauses, not a new rule. The real content — refusing an out-of-achievable-range quote, and stating
  monotonicity as a theorem — needed no accretion.
- **Code's recurring self-caught errors** (bit-exactness over-claims, size-not-shape of an error
  approach, config inherited not measured) were all covered by existing clauses. The charter closed the
  arc leaner-relative-to-coverage than it opened, as it did at the equity capstone.

## The notable oracles — the arc's best method moments
- **The cross-product protection leg (T38)** — the CDS protection leg and the T25 CVA integral, two
  independently built towers, reconciled by different code each side.
- **The convergence-rate companion (T39)** — the flat-hazard anchor is exact and *therefore blind*; the
  defect it cannot see is caught by a term-structured companion's O(1/M) vs O(1/M²) rate, not by any
  single value.
- **The absolute-level protection-leg oracle (T43)** — three mutation survivors with one shared cause
  (every oracle was scale-invariant), killed by a single absolute-magnitude check.
- **The LHP-as-`O(1/N)`-limit (T44a)** — the exact recursion and the closed-form approximation tied
  together by their *measured* convergence rate, the approximation becoming the limit of the exact
  object.
- **The reachability refusal (T44b)** — the non-vacuous half of a vacuous oracle: a quote outside the
  achievable range has no base correlation at all, refused in the bootstrap's own domain vocabulary,
  with the oracle asserting the message is ours and names the reachable range (a bare `raises` would
  pass without the guard).

## Honest completion ratio — what the spine does NOT do (the deferred ledger)
Built as the spine: the CDS + curve + swaption, the Gaussian copula, the FtD basket, the LHP tranche,
the exact loss recursion, and base correlation. Everything below is `to_go_deeper` (oracle-able,
unpulled) unless marked:
- **Bonds & asset swaps** (ch 4); **CDS indices + index options** (ch 10–11, a clean later pull, Black
  revisited); **CMDS** (ch 9.4).
- **Implied & compound correlation** (ch 19 — the non-arbitrage-free measure base correlation replaced;
  pulled only if a reason to show the smile arises).
- The **copula-skew zoo** (ch 21 — student-t, double-t, Marshall–Olkin, random-factor-loading, implied
  copula — a correlation sub-arc); **CPPI / CPDO / CDO² / tranchelets** (ch 22).
- The **base-correlation surface** across maturities (ch 20.6), **bespoke-tranche mapping** (ch 20.9),
  **tranche risk/hedging** (ch 20.7), and the **ρ_B interpolation convention** (ch 20.4 — ruled at T44b
  to have no consumer; the convention is ruled when a bespoke tranche pulls it).
- **`out_of_method` candidate:** the **dynamic credit models** (ch 23 — intensity-gamma, affine
  jump-diffusion) if their market calibration has no closed form or requirement-identity anchor, only a
  twin-MC — ruled at the §1 gate only if interest ever pulls there.

## Governance this arc
- **The split at the loss-distribution tranche (T44 → T44a/T44b)** — ruled on Code's measurements
  (independent, self-anchored, disjoint oracle + cost profiles; the recursion carries the arc's only
  expensive quadrature), the T20/T33/T36 precedent.
- **Three infrastructure tranches (T41b/T41c/T42b)** each landed a real gap the build exposed —
  entity-keying, register integrity, CI-gate coverage — rule-of-two and earn-its-place deciding each,
  none speculative.
- **The draft-PR mechanism held**, with one instructive exception that *proved* the rest: at T44b Code
  stopped its own ready-flag (§10 step 6) to report that Cowork's step-4 ratification rested on a false
  fact it had just re-derived — the two-surface split catching a bad ratification before merge, exactly
  the independence the flow exists to produce.

## Close
The credit-derivatives arc is complete on its builds. The close triggers **re-audit #4** (§9, arc-close
trigger — the eleven-tranche counter sits at ten since re-audit #3 at T37, so arc-close fires first, as
it has every time). Its scope and findings are recorded in `docs/audit/reaudit_04_findings.md`, with the
**proactive invariant-6 / historical-flow sweep as its first named item** (the tenth-shape escalation's
first run) and the oracle-power sweep run **oldest-first**. After it, the library is at a fresh **arc
boundary**: the next direction is Bernardo's interest plus a Cowork spec, written to the queue.
