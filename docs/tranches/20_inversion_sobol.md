# Tranche 20a — the inverse normal CDF: closing tranche 15's deferred inversion

**Status:** BUILT. **Version:** `v0.23.0`. **Source:** `ACKLAM-2000` (`in-hand`, registered this
tranche), recommended by `JAECKEL-2002` §9.3 and printed in neither.

**There is no Cowork spec document for this tranche.** The arc plan
(`docs/jaeckel_completion_arc.md`) carries T20's ruled design — *"`Φ⁻¹` is built primarily by
inverting our own exact `norm_cdf` ... with Acklam's rational approximation as the fast path, the two
cross-checked"* — and Bernardo ruled that the next unit is T20 per that plan, with nothing further
for him to decide. So this was built against the arc plan's ruling; the oracles follow from §4 and
from tranche 15's standing deferred requirement rather than from a fresh oracle choice, and every
place a judgement call arose is raised in §7 rather than settled here.

## 0. Scope — the inversion only; Sobol is T20b

The arc plan pairs the inversion with Sobol (ch 8) because *"Sobol needs a monotone transform, not
Box-Muller"*. The inversion is the **enabler** and is self-contained; Sobol is a second substantial
piece (direction numbers, the Gray-code recurrence, dimension bookkeeping) with its own oracle set.
Landing them separately keeps each one's oracles legible, and tranche 19 set the precedent — Cowork
offered exactly this split there. **This tranche delivers the inversion, fully oracled; Sobol is
not started.** Said plainly rather than implied.

## 1. What landed

`foundation/distributions.py` gains `inverse_norm_cdf` — Acklam's rational approximation (21
coefficients, three regions, `p_low = 0.02425`) followed by **one Halley step** against an exact
CDF, reaching a measured worst miss of **7.66e-16** against an independent implementation. Sub-ULP.

Two things in the build were not in the plan, and both were forced by measurement.

## 2. Our `norm_cdf` is relatively inaccurate in the lower tail — by nine orders of magnitude

The refinement needs `Φ(x)`, and the obvious choice was our own `norm_cdf`. It broke the deep tail.
`norm_cdf` computes `½(1 + erf(x/√2))`, and as `erf → −1` the sum **cancels catastrophically**.
Measured against an independent high-precision CDF:

| x | `½(1+erf(x/√2))` — ours | `½·erfc(−x/√2)` | ratio |
|---|---|---|---|
| −7.03 | **6.1e-06** | 7.4e-15 | ~10⁹ |
| −6.00 | 1.3e-10 | 7.1e-15 | ~10⁴ |
| −5.00 | 3.9e-11 | 3.7e-15 | ~10⁴ |
| −2.00 | 4.6e-16 | 7.6e-16 | — |

Acklam prints **both** forms side by side (*"use this line ... OR this; not both"*) and says he used
Cody's `erfc`; the reason is exactly this. So the refinement uses `erfc`, via a private
`_lower_tail_cdf`, and the upper half is mapped onto the lower by the exact oddness of `Φ⁻¹` so the
CDF is only ever evaluated where `erfc` is accurate.

**The gap in `norm_cdf` itself is raised, not patched (§7 item 1).** That function has shipped since
tranche 5 and nineteen tranches of oracles read it; changing its form is not this tranche's call.
Worth noting *why its own oracle missed this*: it asserts *"tails → 0/1"*, which is an **absolute**
statement and passes however bad the relative error becomes. §4's domain rule, found again — and this
time in a function that had been green for fifteen tranches.

## 3. The upper tail cannot be recovered from its own input, and the oracle had to say so

`Φ⁻¹(1−p) = −Φ⁻¹(p)` is exact in arithmetic and the implementation uses it. But testing it needs
`1 − (1 − p)` to return `p`, which in binary holds only for **dyadic** `p`. At `p = 0.1`,
`1 − (1 − 0.1) = 0.09999999999999998`; at `p = 1e-10` only about four digits of `p` survive. A first
version of the oddness oracle asserted exactness at `p = 1e-10` and failed — measuring asymmetry in
the **inputs**, not in the function. Split: exact on dyadics (`0.5, 0.25, …, 2⁻¹⁰`), and elsewhere
bounded by each input's own measured representation loss rather than by a tolerance chosen to pass.

The same limit removed `1 − 1e-12` from the root-find sweep: for `p` near 1 the reference must solve
`Φ(x) = 1 − ε` where `Φ` is flat at 1 in double precision, so **the reference becomes unresolvable
long before the implementation becomes wrong.**

## 4. Three times the reference was the weaker side

The pattern is worth naming because it repeated within one tranche:

1. The root-find built on `norm_cdf` disagreed with the implementation by 3.9e-07 in the deep tail.
   **Ours was right**; the reference carried the cancellation.
2. At `p = 1 − 1e-12` the root-find could not resolve a root at all. **Not a disagreement, an
   unresolvable reference.**
3. At `p = 0.999` the root-find missed an independent implementation by 3.74e-15 while ours was
   **bit-exact**. The 1e-15 assertion that failed was measuring Brent's tolerance.

So the primary cross-check became `scipy.special.ndtri` — an independent *implementation*, different
algorithm and different provider, which the charter's own tranche-11 note anticipated (`distributions.py`
moved off scipy *"so the oracle and the implementation no longer share a provider"*). The root-find
survives as a corroborating second check at a tolerance reflecting **its** accuracy, not ours. The
lesson: *a reference is not the authority merely because a test calls it one* — it needs its own
accuracy argument, in the regime where it is used.

## 5. Halley's third-order term is not measurable from Acklam's start

Acklam specifies Halley (`x ← x − u/(1 + x·u/2)`); dropping the denominator gives Newton
(`x ← x − u`). From a start already good to 1.15e-9, Newton squares the error to ~1e-18 — below
double precision — so **both land on the same answer**: worst absolute gap 4.44e-16 across the lower
half, which is machine epsilon. A first version of this oracle asserted Halley was *measurably
better* and failed with `4.44e-16 < 4.44e-16`: the assertion was wishful, the code was fine.

The denominator is still transcribed (the source specifies it; the cost is one multiply-add), and its
necessity is demonstrated from a **deliberately degraded start** — a 1% perturbation, where cubic and
quadratic convergence separate by orders of magnitude. Without that companion it would be
unfalsifiable surface; §4's unreachable-branch rule asks for exactly that relocation.

## 6. Mutation — sixteen run, and the four survivors are the resolution, not holes

Twelve killed outright. The four survivors were all **tiny** perturbations, and measuring them is
what turned a worrying result into a precise one:

| mutation | `worst_raw` | output |
|---|---|---|
| none | 1.128781e-09 | — |
| `c`-coefficient +1e-14 | 1.128781e-09 | **bit-identical** — proved equivalent |
| break-point +1e-4 | 1.128781e-09 | **bit-identical** — proved equivalent |
| `a`-coefficient +1e-12 (~20 ULP) | 1.128782e-09 | sub-ULP |
| `a`-coefficient +1e-6 (4.5e-9 rel) | **4.40e-08** | caught — 38× the bound |

The Halley refinement **repairs** small coefficient errors, which is good engineering and awkward for
verification: the refined output is nearly indifferent to the transcription. The only oracle with
grip is the published-accuracy bound on the *unrefined* approximation — measured at **1.1288e-09
against a published 1.15e-09, 98% of it** — and its resolution is ~20 ULP of a coefficient. Below
that, errors are provably immaterial. An oracle proving the bound **can fail** was added, since
otherwise the one check guarding 21 magic numbers would itself be unverified.

**A wrong diagnosis on the way, recorded because I stated it confidently.** The four survivors were
exactly the length-preserving edits, so I concluded Python's bytecode cache (keyed on mtime+size) had
silently run stale code and reported false survivors. It had not: re-running with `-B` and a cleared
`__pycache__` gave the same four, and a direct probe confirmed the mutations execute. The correlation
with edit length was a coincidence — small numeric edits happen to preserve length. The real cause is
§6's resolution limit.

## 7. Findings for Cowork (§10 — raised, not decided)

1. **`norm_cdf` should probably move to `erfc(−x/√2)/2`** (§2): same quantity, nine orders of
   magnitude better relative accuracy in the lower tail, identical in the bulk. Not done here — it
   is read by nineteen tranches of oracles and is not bit-identical, so §4's pure-refactor rules
   apply and the change wants its own unit. Its existing oracle's *"tails → 0/1"* is an absolute
   claim and cannot see this; a relative-accuracy oracle is owed either way.
2. **The arc plan calls the root-find "primary" and Acklam the "fast path"; I shipped one function**
   (Acklam + Halley) with the root-find living in the oracle. A production root-find would have **no
   consumer** once the refinement is machine-precise, which §6 forbids, and the cross-check the plan
   wants is fully achieved with the reference in the test. Rule whether that reading stands.
3. **Scope: Sobol is not built** (§0). Confirm T20b, or rule that the two must land together.
4. **Candidate §4 note, from §4 above:** *a reference needs its own accuracy argument in the regime
   where it is used.* Three times here the implementation was right and the reference wrong — a
   cancelling CDF, an unresolvable bracket, a root-finder's tolerance. Where a test computes its own
   reference, that reference is code too.

---

## 8. Cowork's ruling on T20a — PR #27 (§10 step 4, 2026-11-07)

Reviewed; accepted — v0.23.0, worst miss 7.66e-16, the Acklam coefficients cross-checked against the
root-find and the round-trip. **My process miss first:** there was no T20 spec when you built — my
0097 queued a one-line arc entry, not a spec. That is step 1 owed by me; §10 now says "read the queue"
settles the direction, not the spec, and the handoff carries the spec. You built against the ruled
arc-plan design and raised every call rather than settling — the right fallback. Owned.

**1 — `norm_cdf`'s tail cancellation: its own tranche (T20c), ledgered OI-3.** You were right to raise
rather than patch — it is not bit-identical and nineteen tranches read it. The fix is `0.5·erfc(−x/√2)`,
provably the same quantity (`erfc(−y) ≡ 1+erf(y)` in ℝ), so §4's pure-refactor treatment applies: the
identity where the forms agree (centre, to ULP — existing oracles stay green) plus the **relative**-
accuracy oracle it never had, vs `scipy.special.ndtri`; the T20a `erfc` duplicate retires when it moves.
Queued after T20b, before T21. The lesson (absolute-tolerance blindness to relative tail error; a
reference is code too) is charter §4.

**2 — one production function, root-find in the oracle: correct, and the right reading of the arc plan.**
My reframe said "build both"; §6 sharpens it — a production root-find would have no consumer, so the
root-find belongs in the oracle as the independent cross-check (different code, Acklam's coefficients
untrusted until they match), and Acklam+Halley is the single shipped `Φ⁻¹`. That is what you built.

**3 — T20b (Sobol) confirmed as next.** Spec below.

**Also:** Acklam-in-§9.3 correction noted (references.md already right); the Halley-vs-Newton negative
(third-order term unmeasurable from a 1e-9 start, necessity shown from a degraded start) and the
dyadic-only oddness oracle both accepted as honest.

## 9. T20b spec — Sobol / low-discrepancy (§8)

**Target:** `v0.24.0`. **Source:** `JAECKEL-2002` §8 (Sobol §8.3: primitive polynomials mod 2, the
construction, the Gray code, initialisation). **Pieces:** `foundation/sobol.py` (L0 — the direction
numbers and the Gray-code generator, finance-free), feeding the existing `variates` via the T20a
inversion (Sobol → uniforms → `Φ⁻¹` → normals; NOT Box-Muller — the monotone-transform reason).

**Oracle (RULED unless marked TO MEASURE):**
- **RULED — equidistribution / the low discrepancy is the point:** the sequence's 1-D and 2-D
  projections fill the unit square far more uniformly than pseudo-random; assert the star-discrepancy
  (or an L2 proxy, Jäckel §8.6/§8.8's closed form) falls below the pseudo-random expectation at the
  same N. Pointwise-transcribe the direction-number construction (§8.3) — the load-bearing formula.
- **RULED — beats the −½ rate on a smooth integrand:** pricing Black-by-simulation (T15's anchor)
  on Sobol+inversion converges faster than 1/√N — fit `log err` vs `log N`, slope steeper than −½
  (toward −1). The closed form (`black_price`) is the anchor; this is the whole reason Sobol exists.
- **TO MEASURE — how much faster, and the dimension at which it degrades:** report the observed rate
  and the dimension where Sobol's advantage washes out (the Brownian-bridge partner, T21, is what
  front-loads variance into the low dimensions to hold the advantage — not built here).
- **RULED — determinism / failure-is-a-value:** the sequence is a pure function of its dimension and
  index; an unsupported dimension → a value, not a silent wrong sequence.
- **Degenerate-fixture guard:** do not oracle discrepancy in 1-D alone (Sobol's whole point is the
  *joint* low discrepancy); use a multi-dimension projection.

**Cross-check:** `crosscheck: quantlib (SobolRsg)` on `foundation/sobol.py`. Mine
`_stock/.../numerical/_qmc.py` for shape only; the direction numbers are read from **Jäckel §8.3**.
`source:` cites `JAECKEL-2002` §8. **Open as a DRAFT PR** (§10 step 3).


---

# T20b — BUILT (appended at build; §9 above is Cowork's spec, unedited)

`v0.24.0`. 756 tests, all eight gates green, ruff and pyright clean. `foundation/sobol.py` (L0,
finance-free), consuming tranche 20a's inversion. **Fifteen mutations, all fifteen killed on the
first pass.**

## 10. A ~350-number table, extracted twice by machine rather than read twice by eye

Table 8.3 gives, for each of 32 dimensions, a degree, a primitive polynomial in binary, and ten
initialisation numbers — about 350 magic numbers. §4's dual-rendering rule exists for exactly this,
but *reading* it twice by eye would be the **weaker** method at that scale, not the stronger one: eye
fatigue is correlated between passes in a way `pdftotext` is not.

So both renderings were **parsed and reconciled mechanically**: 32 rows from `-layout`, 32 from
`-raw` (which reflows the table to one value per line, so it needed a different parser — itself a
useful sign the two renderings are genuinely independent), **zero disagreements**, and every exponent
confirmed to equal `32 − l`. The implementation's table was then *generated* from the reconciled
extraction, never hand-typed.

## 11. What makes the table verifiable is that the source states a property of it

A transcribed table is only as good as the oracle over it, and T20a showed how weak that can be —
there, the refinement repaired small coefficient errors and the only grip was a published error bound.
Here the source hands over something much stronger in its own caption: *"Initialisation numbers
providing **property A** up to dimension 32."*

Property A is sharp and testable: in any binary segment of length `2^d` aligned to a multiple of
`2^d`, each of the `2^d` sub-hypercubes contains **exactly one** point. One mistyped direction integer
moves a point across a half-boundary and a count becomes 0 or 2. Verified over four aligned segments
per dimension, `d = 1..8`. Beyond `d = 8` a segment is more draws than is practical (`d = 32` would be
4 billion), so the higher dimensions rest on the two bit constraints and the recurrence — **stated,
not glossed**.

The bit constraints carry more weight than they first appear. The source prints only ten direction
integers per dimension; the other 22 come from eq 8.19. Their correctness is established *structurally*
— *"only the `l` leftmost bits can be non-zero, and the `l`th leftmost bit must be set"* — over all
32 × 32 of them. That is how a recurrence is verified where the source prints no values to compare to.

## 12. The first segment is short by one, and distinguishing that from a defect was the work

Property A failed on first run at "15 of 16 sub-hypercubes". The cause is in §8.3.4 and is not a
defect: property A is stated over the sequence **including `n = 0`**, the origin — which we skip,
because a `0.0` maps to `−inf` under `Φ⁻¹` and tranche 20a's domain guard refuses it. The source says
precisely what follows: *"if we explicitly exclude the point at the origin ... the segment of the
sequence starting with the first element will have no point in the sub-hypercube `[0, ½)^d` and will
be most uniform for a length of `2^d − 1`."*

So the oracle now asserts the **exact** expected shape — first segment hits `2^d − 1` cells with the
all-lower cell missing, later segments hit all `2^d` — rather than loosening a count. That matters
because "one cell short" is *also* what a transcription error looks like, and the two have to be told
apart rather than both tolerated. A first version indexed segments from our own skipped numbering and
was measuring the skip, not the table.

## 13. Dimension 1 is the construction evaluated at `g_k = 0`, not a special case

Its polynomial is `p_1(z) = 1`, degree 0, so there are **zero** freely-chosen initialisation numbers
and eq 8.19 reads `v_1l = (v_1l >> 0)` — it references itself and says nothing (the first
implementation raised `IndexError` there, which is the honest symptom of a vacuous recurrence). What
remains is `v_1l = 2^(b−l)`, i.e. the unit initialisation of eq 8.24 — and that is exactly what Table
8.3 prints for `k = 1`, every multiplier being 1. So dimension 1 is the van der Corput / radical-
inverse sequence in base 2, which is what it should be.

Worth noting the asymmetry the source flags: unit initialisation is a *valid* choice for every
dimension and *"leads to surprisingly bad results ... and should be avoided"*. It is right for
dimension 1 only because there the construction forces it.

## 14. The rate — RULED and TO MEASURE, both answered

**RULED, and it holds:** on a Black call priced by simulation with Sobol + the T20a inversion, the
fitted convergence exponent is **−0.906**, against pseudo-random's **−0.481** on the same integrand.
Theory is −1 and −0.5. At `N = 32,768` the Sobol error is 8.4e-07 against pseudo-random's 2.7e-05 —
**32× better**.

| N | Sobol error | pseudo-random error |
|---|---|---|
| 1,024 | 1.98e-05 | 3.16e-04 |
| 4,096 | 5.67e-06 | 7.66e-05 |
| 16,384 | 1.59e-06 | 6.29e-05 |
| 32,768 | 8.39e-07 | 2.74e-05 |

Asserted as a **fitted slope steeper than −½**, which is the only shape that can fail honestly: a
single `N` could beat pseudo-random by luck, a rate cannot.

**TO MEASURE — where the advantage washes out.** Discrepancy proxy at 1,024 points:

| d | Sobol | pseudo-random | advantage |
|---|---|---|---|
| 1 | **0** (exact) | 8.79e-03 | perfect |
| 2 | 9.77e-04 | 9.77e-03 | 10× |
| 3 | 9.77e-04 | 1.95e-02 | 20× |
| 5 | 1.19e-03 | 1.17e-02 | 9.8× |
| 8 | 4.73e-03 | 8.32e-03 | 1.8× |
| 16 | 2.05e-03 | 2.21e-03 | 1.1× |
| 24 | 6.31e-03 | 6.31e-03 | **1.0× — gone** |
| 32 | 3.64e-03 | 1.69e-03 | **0.5× — worse** |

So the advantage is 10–20× through `d = 5`, decays to 1.8× by `d = 8`, and is **gone by `d = 24`**;
at `d = 32` Sobol is *worse* than pseudo-random on this proxy. **With the caveat that matters:** this
is a statement about `N = 1,024`, not about Sobol. Property A at `d = 32` needs `2^32` draws, so at
1,024 points the sequence has had no opportunity to equidistribute in that many dimensions. The
honest reading is *"at feasible path counts the advantage is a low-dimension phenomenon"* — which is
exactly what tranche 21's Brownian bridge exists to exploit, by front-loading the variance into the
dimensions where the advantage lives.

**In 1-D the discrepancy proxy is exactly zero**, which crashed the first measurement on a division.
Not a bug but the point: with `N = 2^k`, Sobol hits every dyadic box exactly. It is also why the
spec's degenerate-fixture guard forbids oracling discrepancy in 1-D alone — there the sequence is
*perfect* and tells you nothing about the joint behaviour that matters.

## 15. Findings for Cowork (§10 — raised, not decided)

1. **Sobol is NOT wired into the LMM path engine.** The spec's anchor is Black-by-simulation (1-D),
   and that is what is built. An LMM path needs one dimension per forward per step — a 6-forward,
   8-step simulation is 48 — which §14 shows is past where the advantage has washed out at feasible
   path counts. Wiring it without the Brownian bridge would add surface that does not pay, so it is
   left for T21 where the bridge makes it worth having. Rule whether that stands.
2. **Dimension 33 is refused rather than extended.** The source's polynomials beyond Table 8.3 are
   *"on the accompanying CD"* we do not hold — the **third** time this book has sent a reader to
   something it does not contain (after `Φ⁻¹` and the `(1−ρ²)` factor). Recorded in
   `references.md`'s standing-pattern note. If higher dimensions are ever needed it is an
   acquisition, not a build.
3. **Candidate §4 note:** *at scale, transcribe by machine and reconcile, rather than reading twice.*
   The dual-rendering rule's purpose is independence, and two passes by the same pair of eyes are not
   independent in the way two parsers are — 350 numbers is past where a human second read is the
   stronger check. Worth stating because the rule currently implies eyes.

---

## 16. Cowork's ruling on T20b — PR #28 (§10 step 4, 2026-11-07)

Accepted — v0.24.0, fifteen mutations killed. The convergence anchor is exactly right: a fitted
exponent (−0.906 vs pseudo-random −0.481) asserted as "slope steeper than −½", the only shape that can
fail honestly — your 0103 carry-in, correctly built. The source-resolved apparent defects (property A
including n=0 which we skip → assert the exact §8.3.4 shape; d=1's degree-0 polynomial → van der
Corput) are the read-the-source discipline, not loosened tolerances. Good.

**15.1 — Sobol not wired into the LMM path: stands.** 48 dimensions is past the washout (your measured
d≈24), so wiring it before the Brownian bridge adds surface that does not pay — §6. T21's bridge
front-loads variance into the low dimensions where Sobol *does* pay; that is the natural place to wire
it. Deferred, correctly.

**15.2 — d=32 limit: accepted; recorded as an acquisition, not a gap.** The higher-dimension primitive
polynomials are on the book's CD (third CD instance), which we do not hold, and refusing d>32 rather
than fabricating is §7. d≤32 covers every consumer where Sobol pays; a high-dimension path does not
benefit at feasible N anyway. Noted in `to_go_deeper.md`.

**15.3 — the machine-transcription of Table 8.3: charter-bound (§4).** Your argument is sound and more
faithful to §4's *spirit* (independence) than its letter (two eye-reads): at 350 numbers a human's two
passes are not independent (fatigue correlates), so two parsers reconciled mechanically are the
stronger check — and the oracle is property A, which one wrong entry breaks, not the transcription.
The Independence bullet now carries it.

**And 0103's process failure — the commit-hygiene rule is charter-bound (§10).** Committing a ruling
blind via `git add -A` into a test commit, then enforcing it unread, is exactly the invisible-in-the-log
harm §10 now forbids: Cowork's artifacts commit under a named `docs:` message, staged by path, and Code
reads what it commits. Your two mechanical fixes are the right response; the charter makes them law.

**Next: T20c** — fix `norm_cdf`'s tail (OI-3), specced at `docs/tranches/20c_norm_cdf_erfc.md`. Then
T21 (Brownian bridge), where Sobol finally gets wired.
