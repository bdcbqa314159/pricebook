# Tranche 53 — share `φ(u)` across a maturity's strikes (OI-5 removal)

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.56.1` (a behaviour-preserving
refactor that retires an exemption — version Code's to confirm by the commit convention). **Source of the
unit:** `OPEN.md` OI-5's named removal trigger, raised at T50 (`docs/tranches/50_oi4_fourier_truncation.md`).
The **last queued unit** of the Gatheral tail; its merge retires the one standing exemption and the slate is
clean.

## 0. What this is
`gil_pelaez_probabilities` (and the Heston/SVJ pricers through it) re-evaluate the characteristic function
`φ(u)` at every integration node **per strike**, though `φ` **does not depend on the strike** — only the
Gil-Pelaez phase does. So pricing a smile costs **4–10×** the `φ` evaluations it needs. Sharing one `φ(u)`
vector across a maturity's strikes retires that waste — **in production as much as in the test**, which is
why it is a product worth having and not only an exemption-remover. It is the "no reshaping where already
correct" fence from T50 lifting now that the waste *is* the unit, raised on its own rather than smuggled
into a correctness fix.

## 1. The pieces and their layers
- **L4 — `engine/fourier_inversion.py`** (the shared Gil-Pelaez, T49): hoist the `φ(u)`-at-nodes evaluation
  out of the per-strike loop so it is computed **once per (model, maturity)** and reused, varying only the
  strike-dependent phase. The pricers that call it (`heston_fourier`, `bates_fourier`) consume the shared
  vector. No product, model, or contract shape changes; `price(product, model)` is unchanged (§3).
- **No new abstraction (§6)** — this is compute-once-reuse, a loop hoist, not a new type or polymorphism.
  No speculative generality (a batch-pricing API) beyond what sharing `φ` across one maturity's strikes
  needs now.

## 2. The oracle (RULED) — a pure refactor, proved by the §4 identity discipline
This changes no price; the whole burden is proving that.
- **The purity proof is in-process bit-identity IF op order is preserved; the exact-arithmetic identity +
  a few-ULP band IF the reduction is reordered (§4, T49/T12a).** Hoisting `φ` out of the strike loop does
  not by itself reorder any single price's node-sum — the same `φ(u)` values, the same phase, the same sum
  — so the natural result is **bit-identical in-process** (the T49 "available when op order is preserved").
  **If** the refactor vectorises the node reduction across strikes and that changes the summation order
  (pairwise vs sequential), it is no longer bit-identical: then the proof is the **exact-arithmetic
  identity** (the two expressions are the same quantity in ℝ) plus agreement **bounded to a few ULP with
  the reordering named** — never claimed "bit-identical" when reordered. **Code measures which case holds
  and reports it**; the spec does not assume.
- **The committed regression is a TOLERANCE, not the bit pattern (§4, T49).** Bit-identity is the in-process
  *proof*; the committed cross-platform test is a **relative** tolerance (an accumulated transcendental sum
  differs relatively, not in ULP — measure the worst and set ~1e-9 or tighter against it). Do not commit the
  360-double bit patterns as an equality or ULP-distance — that is the T49 mistake.
- **The pre-existing oracle set stays green, but that is not the proof** (§4) — OI-4's no-arbitrage region
  sweep, the SVI/Heston/SVJ/vol-swap oracles all hold at their tolerances (necessary; a loose tolerance
  could hide a real change). The identity is the proof.
- **The performance property is an ORACLE, not a changelog line (§7 guarded-vs-folklore; T50 item 1, "a
  rationale that is not a test erodes").** Assert that `φ` is evaluated **once per maturity, not per
  strike** — a test that counts `φ` calls (or asserts the cost scales with nodes, not nodes × strikes) —
  so the 4–10× win cannot silently erode at the next refactor. The win is the point; guard it.

## 3. OI-5's retirement (RULED, conditional on the measurement)
The exemption existed because OI-4's region sweep (many `(σ_v, T, K/F)` points, strikes within maturities)
ran at ~11s on Windows. The φ-sharing speeds exactly that loop. **So: re-time the OI-4 oracle file on
BOTH CI runners after the refactor (§9 — local is not the measurement, CI is).** If it is under the `slow`
ceiling without the exemption, **remove the `slow-test-exempt` and close OI-5 in `OPEN.md`** with the
measured new timing. If it is still over (surprising at 4–10×, but measured not assumed), the exemption
**stays** with a note on the residual — never remove an exemption on a local reading or an expectation.

## 4. Process
Branch `tranche/53-phi-across-strikes`. `test` (RED: the φ-call-count oracle + the purity regression at the
ruled tolerance) → `feat` (the hoist) → `docs` (close OI-5 in `OPEN.md` if the CI re-time clears it; the §4
identity result — bit-identical or ULP-with-cause, as measured) → `chore(release)`. Draft PR; ready only at
step 6. **Raise an `Ask:`** only if the refactor forces a reduction reorder that pushes the tolerance past
a few ULP (then the purity argument needs a ruling), or if the CI re-time does *not* clear the ceiling.

## 5. What this is NOT
No change to any price beyond the proved tolerance (a real change is a finding, not shipped). No new
product/model/API. No **batch or vectorised pricing surface** beyond sharing `φ` across one maturity's
strikes (eSSVI-style whole-surface batch pricing is `to_go_deeper` if ever pulled). No reshaping of the
inversion's *mathematics* — only which quantity is computed how many times. On merge, OI-5 is closed (if CI
clears it) and the Gatheral tail is complete.

---

## 6. Cowork's ruling on T53 — (reserved for §10 step 4, after the build)
