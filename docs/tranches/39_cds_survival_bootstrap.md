# Tranche 39 — the CDS survival-curve bootstrap

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.46.0`.
**Source:** `OKANE-2008` ch 7 (calibrating the CDS survival curve). **Pre-registered RED.** Second unit
of the credit arc — the L1 bootstrap the XVA arc (T25) **explicitly deferred**: *"inverting market CDS
quotes for λ is a curve bootstrap, arriving with the product that needs it."* T38's CDS is that product.

## 0. The product worth having
**A survival curve built from a strip of par CDS quotes** — the market-standard way a hazard curve is
produced. Given par CDS spreads across maturities (6m, 1y, 3y, 5y, …), solve for the piecewise hazard
that reprices every quote to par, maturity by maturity, shortest first — exactly T8's nominal
par-swap bootstrap, now on CDS. Until now a `SurvivalCurve` had to be handed in with its hazards given
(T25); T39 produces one from the quotes a desk actually sees.

```
At each maturity Tₘ (shortest first): solve for the hazard λₘ on (Tₘ₋₁, Tₘ] such that
    V(CDSₘ) = Protection(Tₘ) − S_quote(Tₘ)·RPV01(Tₘ) = 0
with earlier segments λ₁…λₘ₋₁ already fixed. Piecewise-flat forward hazard interpolation (ch 7.4),
the same family as T8's piecewise-flat forward rate.
```

## 1. The pieces and their layers
- **L1 `market/` — `bootstrap_survival_curve(quotes, discount, recovery, …) → SurvivalCurve`.** It
  **produces market data** (a `SurvivalCurve`), so by §2's controlling test (*what it produces*, not
  what it reads) it is **L1**, beside T8's `nominal_bootstrap` — a sequential root-find (the existing
  `foundation/solvers.py` brentq-family) per maturity, using only L0+L1 (the discount curve, the
  survival curve being built, the schedule). It does **not** call the L4 `price()` engine — that would
  make it an engine-inverting L4 calibrator (the JY-to-floor-strip case); it does curve arithmetic, as
  T8 does.
- **The shared leg functional — the one real design question (`Ask:` flagged).** T38's engine already
  computes `RPV01` and the protection leg; the bootstrap needs the *same* arithmetic. Rule-of-two is
  met (T38 engine + T39 bootstrap = two consumers), so the leg PVs are **extracted to one shared
  function** both call — not duplicated (§6). **Principle ruled:** if the leg PV can be written with
  only L0+L1 imports (it is curve arithmetic — `Σ Δ·Z·½(Qₙ₋₁+Qₙ)` and `(1−R)·½Σ(Zₖ₋₁+Zₖ)(Qₖ₋₁−Qₖ)` over
  a schedule, a discount curve, a survival curve, a scalar `R`), it lives at **L1** and both the L4
  engine and the L1 bootstrap import it. If it structurally needs an L2 product type, the bootstrap
  reimplements the minimal curve arithmetic (as T8 did) and a **cross-module oracle** asserts the two
  agree (the §7 conventions register, `guarded`). **`Ask:` at build:** which of these two holds once the
  types are in hand — a one-grep question (does the leg PV touch `CreditDefaultSwap`, or only curves?).

## 2. The oracle (RULED)
The **known-curve round trip** is the correctness anchor (T8's, the one that caught two BERHAN
misprints); the **credit triangle inverted** is an independent closed-form cross-check; reprice-to-par
is the telescoping guard; arbitrage detection is a requirement.

- **The known-curve round trip — the preferred anchor (§4, T8's exact precedent).** Build a **known**
  piecewise hazard curve (term-structured, non-flat), price par CDS spreads off it (via the shared leg
  functional), **bootstrap those spreads back**, and recover the known hazards to tolerance. Non-
  telescoping because the reference is an independent constructed curve, not the quotes themselves —
  this is the anchor T8 used to catch a source misprint, and it is why the bootstrap's inverse must
  actually invert (§4 "a formula and its stated inverse must actually invert").
- **The credit triangle, inverted — an independent closed-form cross-check.** A **flat** CDS term
  structure at spread `S` with recovery `R` must bootstrap to a **flat hazard `λ = S/(1−R)`** (the
  credit triangle `S = λ(1−R)` solved for `λ`, in the continuous limit) — a closed form the round trip
  does not use, so it catches a round-trip that is self-consistently wrong. Assert it in the
  fine-grid/continuous limit (the discretisation's `O(r/M)` error measured, §4 banded).
- **Reprice-to-par — the self-consistency guard (telescopes, §4).** Each bootstrapped quote reprices to
  `V = 0` by construction of the root-find — kept as a guard, but it telescopes (the bootstrap and the
  repricer share the leg functional), so it is **not** the correctness anchor; the round trip and the
  triangle are.
- **Arbitrage detection — a requirement oracle (ch 7.7).** The bootstrap must **reject** a quote strip
  that implies a **negative forward hazard** (a non-monotone survival) with a clear failure, never
  silently return a survival curve that rises — `Q` monotone non-increasing, `λ ≥ 0` on every segment
  (the §3-invariant-4 shape: failure is a value / a raised-and-caught `ValueError` at construction, not
  a silent wrong curve). Assert a known-arbitrage strip is refused.
- **Degeneracy premises — τ=1 BAKED IN (the F-1.2/F-3.1/F-3.2 history).** The round-trip and triangle
  fixtures carry a **semi-annual** premium schedule (non-unit accruals `Δ`), **term-structured** quotes
  (non-flat hazard, so the per-segment solve matters — a flat-only fixture would make every segment
  identical), and a **non-flat** discount curve, read beyond the first segment. Asserted as premises.
- **Mutation (prove power).** Each must die: drop the accrual in the shared leg math (→ dies on the
  semi-annual round trip); use `Q(tₙ)` not `½(Qₙ₋₁+Qₙ)` (→ dies on term structure, not flat — the
  degenerate-fixture premise); skip the arbitrage check (→ the negative-forward-hazard strip passes);
  flip the root-find objective's sign / solve the wrong segment (→ round trip fails to recover);
  seed the solver at the wrong bracket (→ convergence/robustness test).

## 3. Process
Branch `tranche/39-cds-survival-bootstrap`. `test` (RED: the known-curve round trip + the inverted
credit triangle + the arbitrage-rejection + reprice-to-par guard + the mutation table) → `feat` →
`docs` → `chore(release): v0.46.0`. **Open the draft PR early; pre-register the round trip and the
triangle first.** **Transcribe the bootstrap algorithm (ch 7.5) and the flat-forward interpolation
(ch 7.4) in both renderings**; use the **T38 leg formulas** as the shared functional (extract, don't
re-transcribe — but confirm the extraction is a *pure refactor*, §4 exact-arithmetic identity, so the
engine's numbers do not move). `source:` `OKANE-2008`; `crosscheck: quantlib` — the ISDA/QuantLib CDS
bootstrap (`SpreadCdsHelper` / `PiecewiseDefaultCurve`) is a real target. **Raise the `Ask:`** from §1
(shared-leg-functional placement: L1-shared vs reimplemented-with-cross-module-guard) once the types
are in hand — it is a one-grep question, and the build-raises-it loop is the right place (the T22
precedent).

## 4. What this tranche is NOT (scope fence)
No **CDS risk / CS01** or **forward CDS / swaption** — T40. No **multi-name** anything — T41+. No
**interpolation-scheme zoo** beyond the flat-forward hazard O'Kane uses (other schemes — `to_go_deeper`).
No **recovery-rate curve** (a single assumed `R`, as T38). No **defaultable-bond** calibration (ch 4,
`to_go_deeper`). T39 is the one bootstrap: par CDS strip → piecewise hazard, with the round trip and the
inverted triangle as anchors.

---

## 5. Cowork's ruling on T39 — PR #53 (§10 step 4, 2026-11-14)

Accepted with one change request (OI-4, below) — v0.46.0, 1512 green, ten gates. The bootstrap the XVA
arc deferred is in; the known-curve round trip recovers a term-structured hazard, and the inverted
credit triangle is the independent closed-form cross-check.

**The §1 leg-functional Ask — resolved exactly right.** The grep settled it: the arithmetic touches no
L2 type (`CouponSchedule` is L0, the curves L1, `R` a scalar), so `market/credit_legs.py` is L1, beside
`nominal_bootstrap`. `risky_pv01` extracted as a **pure refactor, bit-identical over 324 fixtures** —
the §4 exact-arithmetic identity met.

**Your self-correction on the grid argument — ratified, and it is the better call.** Your 0212 proposed
a `LegGrid` argument so the engine stays bit-identical and the bootstrap gets the aligned grid; your
0213 retracted it because the two are **structurally different integrands** (per-segment constant hazard,
exact between pillars, vs a pointwise `default_density` that loses an order at a jump), so a grid mode
would be a **false abstraction over two formulas** — §6 exactly. Deleting it rather than shipping it is
right, and catching it by measuring rather than reasoning is the discipline.

**OI-4 — RULED: the engine adopts the per-segment integration, in T39. Close OI-4; do not carry it.**
Three reasons, and the move is a correctness fix, not a forbidden silent refactor:
1. **The project does not knowingly ship a worse number when the better one is known and cheap.** The
   per-segment integrand is exact between pillars (second order; 3.6e-07 at 50 pts/segment vs 2.9e-04 at
   500 uniform), and the engine's uniform trapezoid loses an order at every pillar. The +1.846e-05 move
   is **toward** an independent 20k/segment reference — provably toward truth, so it is a ruled
   correctness improvement, and the §4 pure-refactor rule (which forbids *silent* reorderings) does not
   bar a *deliberate, ruled* numerics change. Correct the v0.45.0 CHANGELOG "~2e-05 resolution" note in
   the same commit (a doc describing the worse quadrature, §5/§8).
2. **It collapses the duplication.** OI-4 exists only because the engine kept the inferior formula;
   adopting per-segment makes `credit_legs.protection_leg` the **one** formula both the engine and the
   bootstrap call — §6 rule-of-two satisfied, no knowingly-duplicated formula in the tree.
3. **It removes a latent engine/bootstrap inconsistency.** Today the bootstrap calibrates a curve with
   per-segment integration while the engine prices the same CDS with the uniform trapezoid — so a
   bootstrapped curve reprices its own calibrating CDS to *not quite* zero through the engine. Unifying
   is the structural guarantee. **Add a `docs/conventions_register.md` row (guarded):** the engine and
   the bootstrap integrate the protection leg by one shared functional — now true by construction.

**The §4 line this earned — added (the reduction-anchor blind spot).** T38's flat-hazard anchor was
exact *because* a flat hazard has no interior pillars, so it was structurally blind to the pillar
integration error — the "contain the failure in its domain" / T16 "exact-where-the-quantity-is-zero"
lessons, now sharpened for the *reduction anchor*: the companion must be non-degenerate in the specific
parameter the reduction zeroes, checked by convergence **rate**. One sentence on the reduction bullet.

**Oracle 2 restructured to a premium-frequency convergence — ratified.** The inverted triangle's gap is
the **premium discretisation** (first order in the premium period: 1.0e-02 → 5.5e-05 at 2 → 365/yr), not
the integration grid; a measured-rate convergence oracle is strictly stronger than a band at semi-annual
(T21/T25 spirit). Good.

**The two test-side survivors — ratified, and one names a reusable discipline.** A refusal oracle whose
`pytest.raises(match="recovery")` matched an *unrelated* failure sharing the word is "contain the failure
in its domain" applied to a refusal test: **a refusal oracle must pin the condition it refuses, narrowly
enough to exclude a different error** — carry it as discipline (not a charter line yet; flag again if it
recurs). The arbitrage `>`/`>=` boundary reachable only at a 0bp quote is the C10-style boundary case;
closed. Both good catches, and both in the tests — the sweep earning its keep on the suite's own domain.

**Next:** T40 (forward CDS + the default swaption — Black on the forward spread, RPV01 the annuity),
mine to spec. **Implement OI-4's adoption on this branch, re-oracle, correct the CHANGELOG, then mark
#53 ready (step 6) → Bernardo merges.**
