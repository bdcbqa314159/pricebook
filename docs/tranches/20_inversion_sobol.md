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
