# Tranche 20c — `norm_cdf` to the erfc form (OI-3): the tail defect fixed

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.25.0`.
**Source:** none external — a correctness refactor of our own L0 `foundation/distributions.py`, pulled
by T20a's finding (OI-3). **Pre-registered RED, as always.** In the arc queue after T20b, before T21.

## 0. The defect, and why it is its own tranche
`norm_cdf(x) = 0.5·(1 + erf(x/√2))` cancels catastrophically in the lower tail (`erf → −1`): relative
error 6.1e-06 at `x=−7.03` versus 7.4e-15 for `0.5·erfc(−x/√2)` (OI-3). Green for fifteen tranches
because its oracle asserted an **absolute** "tails → 0/1" bound, blind to relative error where the
value is ~0 (§4). Not patched inline at T20a because the change is **not bit-identical** and nineteen
tranches of oracles read `norm_cdf` — it earns the §4 pure-refactor treatment.

## 1. The change
`norm_cdf` becomes `0.5·erfc(−x/√2)`, accurate across the whole range (the small tail is computed
directly, no cancellation, for either sign of `x`). The T20a `erfc`-based sibling added in
`distributions.py` is **retired** — `norm_cdf` itself is now that form, one function again (§6).

## 2. The oracle (RULED)
- **Purity where the forms agree — §4 pure-refactor, exact-arithmetic identity:** `erfc(−y) ≡ 1+erf(y)`
  in ℝ, so the new form equals the old in the centre to ULP. The proof is the identity; the whole
  existing suite staying green (nineteen tranches) is the corroboration, not the proof. Where the two
  differ — the lower tail — the old was wrong; report the ULP agreement in the centre and the
  divergence in the tail with its cause (cancellation), per §4.
- **The relative-accuracy oracle `norm_cdf` never had — the point of the tranche:** across the full
  range including `x ≤ −7`, assert **relative** error below a stated bound against
  `scipy.special.ndtr` (the independent high-precision reference, off the production path per tranche
  11's arrangement; "a reference is code too" — scipy has the tail accuracy our root-find did not).
  Pre-register the lower-tail point the old form fails (`x=−7.03`) as the RED.
- **Requirement:** `norm_cdf` monotone increasing, `∈ (0,1)`, `norm_cdf(0)=0.5` exactly, and
  `norm_cdf(x) + norm_cdf(−x) = 1` to tolerance (the symmetry the erfc form must still satisfy).
- **Mutation** on the sign/argument of the `erfc` call; the relative-tail oracle must fail a wrong one.

## 3. Process
Branch `tranche/20c-norm-cdf-erfc`. `test` (RED: the `x=−7.03` relative-error failure, pre-registered)
→ `feat` → `docs` → `chore(release): v0.25.0`. Closes **OI-3**. `crosscheck: scipy.special.ndtr` is the
oracle reference, not a deferred QuantLib target (this is an exactness fix, not a finance convention).
`source:` none. **Open as a DRAFT PR** (§10 step 3). Expect the whole suite green unchanged (centre is
ULP-identical); if any existing oracle asserted a *wrong* tail value, that is a finding — raise it.
