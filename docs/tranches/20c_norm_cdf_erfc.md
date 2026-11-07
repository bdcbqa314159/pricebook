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


---

# BUILT — results (appended at build; §0–3 above are Cowork's spec, unedited)

`v0.25.0`. 761 tests, all eight gates green, ruff and pyright clean. Six mutations, all six killed —
including a revert to the superseded form, which is the one that proves the new oracles guard the fix.
**OI-3 closed.**

## 4. The defect was worse than OI-3 recorded — 100%, not 6.1e-06

OI-3 and this spec's §0 both cite 6.1e-06 relative error at `x = −7.03`. Pre-registering the oracle
showed the worst case over `[−9, 9]` is **1.0**: below about `x = −8.3` the old form returns
**exactly 0.0**. At `x = −9` the true probability is ~1.1e-19 and the function returned zero — not an
inaccurate probability but a lost one, and a 100% relative error rather than a six-digit one.

Worth stating because it changes how the original finding reads: the T20a measurement at `x = −7.03`
happened to land just above the collapse. The honest characterisation is not "degrades in the tail"
but **"returns zero past a threshold"**, and the threshold is inside the range a deep-tail probability
can legitimately be asked for.

## 5. Purity as a measured partition — a flat ULP claim would have been false

§4 asks a pure refactor to be proved by the exact-arithmetic identity plus bounded agreement. The
identity holds (`erfc(−y) ≡ 1 + erf(y)` in ℝ). The *bounded agreement* half cannot be stated as one
tolerance here, and claiming "a few ULP" would have been exactly the false claim §4 names:

| region | agreement | reading |
|---|---|---|
| `x ≥ 0` | **1 ULP** (bit-identical at 88% of a fine grid) | both accurate: 2.19e-16 vs 2.50e-16 against the reference |
| `x = −1, −2, −4` | 1–3 ULP | rounding |
| `x = −3` | 36 ULP (5.8e-15) | rounding, the local outlier |
| `x = −5` | 2.1e5 ULP (3.9e-11) | **the old form's error** |
| `x = −6` | 6.0e5 ULP (1.3e-10) | **the old form's error** |

So the proof is a **partition**, and the lower half's divergence is *attributed* rather than merely
noted: both forms are compared against `scipy.special.ndtr`, and the oracle requires the new one to
track it (`< 1e-14`) **and the old one not to** (`> 1e-11`). Without that second assertion the
divergence would be unexplained, and an unexplained divergence in a pure refactor is a drift.

The partition also bounds the blast radius for the nineteen tranches of oracles that read `norm_cdf`:
anything evaluating `Φ` above about `x = −4` moves by at most tens of ULP, which no tolerance in the
suite can resolve. The full suite staying green at 760 tests is the corroboration this predicts —
**not** the proof, per §4.

**My own claim was too strong first time, and my own oracle caught it.** I asserted *bit*-identity on
`x ≥ 0`; it failed at `x = 1.2`. For positive `x` both routes produce a value near 1 and round
differently in the last bit — no cancellation, so no accuracy loss, but not identical either. One ULP
is the claim that is true; "identical" was the claim I wanted.

## 6. The workaround was removed, not left alongside

T20a added a private `erfc`-based sibling inside `distributions.py` because `norm_cdf` could not be
trusted in the tail. `norm_cdf` **is** that form now, so the sibling is deleted and there is one CDF
again (§6). The refinement in `inverse_norm_cdf` and the root-find reference in the oracle both call
`norm_cdf` directly.

I checked my T20b carry-in before removing it: **no oracle depended on the two forms differing.** Its
only uses were the refinement's CDF and the test's root-find reference, both of which *become*
`norm_cdf`. What did need updating was a comment — the T20a oracle carried a note explaining why the
reference deliberately avoided `norm_cdf`, and that reason no longer exists; it now records the history
instead of the live rationale.

## 7. A side finding: we resolve further than the reference

Beyond about `x = −37` there is no reference to compare against. At `x = −38` scipy's `ndtr` underflows
to **exactly 0.0** while `erfc(−x/√2)/2` still returns a subnormal (~2.9e-316). So the deep-tail
oracle's bound loosens with depth — 1e-13 to `x = −20`, 1e-12 to `−37` — and that is **the reference's
accuracy, not ours**: two different implementations computing a value spanning three hundred orders of
magnitude agree relatively less well the further out they go. A first version asserted 1e-13
everywhere and failed at `x = −30` with 1.74e-13, which was asserting the reference is better than it
is.

That is §4's *"a reference is code too"* — the note this very arc put into the charter — biting one
tranche after it was written, in the tranche whose oracle *is* a reference comparison. A separate
oracle records the limit explicitly, so that `x = −37` is not later mistaken for where **we** stop.

## 8. Findings for Cowork (§10 — raised, not decided)

1. **No candidate rule from this tranche, and that is worth saying.** Every discipline it needed was
   already in §4: the exact-arithmetic identity for a non-bit-identical refactor, the reference's own
   accuracy argument, the absolute/relative tail trap, and "assert the wrong variant fails". The
   charter earned its keep rather than growing. The only thing I would consider adding is a sentence
   that a pure-refactor proof may be a **partition** rather than a single bound — but that is arguably
   already implied by "report the ULP gap with its cause", and §4 is long.
2. **`OPEN.md` OI-3's figure was wrong by five orders of magnitude** (6.1e-06 against an actual 1.0),
   because T20a measured at one point rather than sweeping. Corrected in the closure. Worth noting as
   a pattern: a ledger entry written from a single measurement inherits that measurement's domain, and
   OI-3 sat open for two tranches with an understated severity.
