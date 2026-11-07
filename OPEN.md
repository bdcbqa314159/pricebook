# OPEN — the debt ledger

Charter §8: no debt is silenced silently. Every suppression, shim, skip or load-bearing TODO is
here with a rationale and a re-open trigger, or it does not exist.

## Suppressions — the gated ledger

`verify.py debt` asserts **suppressions − entries = 0**, counting entries that begin `- [NG-`.
Nothing else in this file is counted, so a non-suppression item must not use that prefix.

**Currently zero.** No `# type: ignore`, `# noqa`, `# pragma: no cover` or `@pytest.mark.skip`
exists in `src/` or `tests/` (verified at v0.6.0).

## Other open items

### OI-3 · `norm_cdf` lower-tail cancellation — CLOSED (fixed at v0.25.0, tranche 20c)

`foundation/distributions.py`'s `norm_cdf` computes `Phi(x) = (1 + erf(x/√2))/2`, which cancels
catastrophically in the lower tail (`erf → −1`): at `x = −7.03` its *relative* error is **6.1e-06**
against an independent high-precision CDF, versus **7.4e-15** for `erfc(−x/√2)/2` — nine orders of
magnitude. Green for fifteen tranches because its oracle asserted an **absolute** "tails → 0/1" bound,
blind to relative error where the value is ~0 (§4).

**Not patched inline, because it is not bit-identical and nineteen tranches of oracles read it.** The
robust single form is `0.5·erfc(−x/√2)`, provably the same quantity (`erfc(−y) ≡ 1 + erf(y)` in ℝ), so
the fix is the §4 pure-refactor: the identity where the forms agree (centre, to ULP — existing oracles
stay green) plus a **relative-accuracy oracle across the full range** it has never had, cross-checked
against `scipy.special.ndtri`/`ndtr`. T20a worked around it (used the `erfc` form in the Acklam
refinement) and added a second `erfc`-based CDF alongside; that duplication is retired when `norm_cdf`
itself moves.

**What the gap costs today:** little for the existing pricers — option `N(d)` weights need *absolute*
tail accuracy (a deep-OTM probability ≈ 0 contributes ≈ 0 to the price), which the `erf` form has. The
relative error bites only where a small tail probability is itself the answer — the inverse-CDF
refinement, already worked around. So nothing ships wrong today; the defect is latent.

**Resolution (2026-11-07, v0.25.0 — tranche 20c).** `norm_cdf` is now `0.5·erfc(−x/√2)`. The T20a
`erfc` sibling is **retired**: it existed only because `norm_cdf` could not be trusted in the tail, so
fixing the cause removed the workaround rather than leaving both (§6). The relative-accuracy oracle it
never had is in place — below 1e-13 across `[−9, 9]`, with the superseded form asserted to **fail** it.

**The defect was worse than OI-3 recorded.** This item said 6.1e-06 relative error at `x = −7.03`;
pre-registering the oracle showed the worst case over `[−9, 9]` is **1.0** — a 100% relative error,
because below about `x = −8.3` the old form returns **exactly 0.0**. Not an inaccurate probability but
a lost one.

**Purity, as a measured partition rather than one tolerance** (§4 — a flat "a few ULP" claim would have
been false): `x ≥ 0` agrees to **1 ULP** (bit-identical at 88% of a fine grid, both forms accurate to
~2e-16); down to `x = −4`, tens of ULP; below `x ≈ −5` the forms diverge and the divergence is the
**old** form's error, attributed by comparing both against `scipy.special.ndtr` rather than merely
noted. The whole suite stayed green unchanged at 760 tests — corroboration, not the proof.

A side finding: beyond `x ≈ −37` our form resolves **further than the reference** (at `x = −38` scipy's
`ndtr` underflows to exactly 0.0 while `erfc` still returns a subnormal), so the deep-tail oracle's
bound reflects the reference's accuracy, not ours. **Closed.**

### OI-2 · §3 invariant 7 — CLOSED (accepted by Bernardo, implemented at v0.20.1)

Charter §3 gained invariant 7 at v0.20.0 (Cowork's T17 ruling): *"a `PricingResult` from a
Monte-Carlo pricer carries its **standard error**"*. **No pricer does this yet.** `PricingResult` has
`pv`, `accrued` and `basis`; `engine/swaption_mc.py` computes a standard error inside `estimate(...)`
and discards it.

**Ledgered rather than implemented, and the reason is whose call it is.** Cowork ruled it and
explicitly deferred the code (*"when you land it, the caplet and swaption pricers set it"*), while
flagging that **Bernardo owns §3 and may veto**. Landing a change to the engine contract before that
veto window has opened would pre-empt the person who owns the section. But §9 is equally explicit
that a charter stating a practice the repo does not follow is worse than one that is silent — the
error §8 and §9 both had to correct — so the gap is recorded here instead of left to be discovered.

**What the gap costs today:** every banded oracle over an MC pricer re-simulates to recover the SE
(`tests/L4/test_swaption_mc.py::_estimate_from_paths`). That is one extra simulation per banded case
and, as the ruling notes, a *correctness* smell rather than only an efficiency one — a re-simulated
SE describes different paths than the price it bands. It is not currently wrong, because the helper
re-simulates with the **same seed, grid and config**, so the paths are identical by construction; the
oracle asserts exactly that by pinning the engine's own PV to `mean × discount × notional` at
`rel=1e-12`. So the band does describe its own price. The smell is that nothing structural keeps it
that way.

**Resolution (2026-11-05, v0.20.1).** Bernardo **accepted** invariant 7, which was this item's
re-open trigger. `PricingResult` gained `standard_error: float | None = None`;
`price_european_swaption` sets it, scaled by the same deterministic `discount x notional` as the PV;
the banded oracles now read `out.standard_error` instead of re-simulating. Exact pricers inherit
`None` from the default and were not touched, so no other engine changed.

**One part of Cowork's build note did not apply.** It said *"the caplet and swaption pricers set
it"*; `mc_caplet_price` returns a `MonteCarloEstimate`, not a `PricingResult`, and that type has
carried `mean`, `standard_error` and `n_samples` since tranche 15 — its docstring already states
invariant 7's principle (*"an MC engine that returned only a price would make every downstream oracle
invent a tolerance"*). So the caplet pricer already honoured the invariant by a different route and
needed no change. Converting it to return a `PricingResult` would be a design decision nobody ruled,
so it was not done.

`_estimate_from_paths` survives in the oracle with a narrower and better purpose: an independent
recomputation of the payoff on the same paths, pinning the engine's mean *and* its error bar, rather
than supplying a number the engine could have been wrong about. **Closed.**

### OI-1 · §8's crosscheck markers — CLOSED (spelling fixed; discipline was working, six of six)
**Opened:** 2026-11-01 (T1–T6 checkpoint). **Ruled closed** by Cowork the same day on the premise
that the markers had zero instances. **REOPENED by Code, 2026-11-01, because that premise is false
and it was Code's own error.**

**There are SIX markers, covering the model-dependent pricers and the inflation curve:**

```
models/jarrow_yildirim.py:44    quantlib YoYInflationSwap / JY convexity
market/inflation_curve.py:15    quantlib ZeroCouponInflationTermStructure
engine/yoy.py:24                quantlib YoYInflationSwap
engine/inflation.py:16          quantlib ZeroCouponInflationSwap
engine/inflation.py:101         quantlib CPIBond
engine/capfloor.py:20           quantlib YoYInflationCapFloor
```

**Why it read as zero.** The charter specifies the marker as `# crosscheck: …` — a *comment*. The
practice writes `crosscheck: …` as a line in the module's **provenance header docstring**, with no
`#`. Code grepped the charter's literal spelling, got nothing, and reported the mechanism empty.

**This is charter §4's own defect class, committed by the person who had just invoked it: the
search's domain excluded the form the thing actually takes.** The discipline was never dead — it
is at six of six model-dependent sites, which is what "complete by construction" was asking for.

**What is actually open, and it is small:** the charter says `# crosscheck:` and the tree says
`crosscheck:` in the provenance header. One of the two spellings is wrong. The header form is
arguably the better one — it sits with `stock:`/`source:`/`oracle:`/`slice:`, which is where a
reader looks for provenance — in which case §8 should say so, and a gate becomes trivial (the
`provenance` gate already parses those headers).

**Not ruled here.** Cowork's narrowing (on disk, **not committed**) argued "an ungated prose rule
reached zero instances across six tranches — gate a discipline or do not legislate it." The
conclusion may still be right, but **that rationale does not hold**: the rule reached six of six.
A second claim in the narrowed clause also does not hold yet — it says the sites are "enumerated in
`docs/papers/coverage-berhan-2012.md`", and that document has no cross-check column at all.

**RE-RULED and CLOSED (Cowork, 2026-11-01), on the corrected facts.** The narrowing is **retracted**
— its rationale ("an ungated rule reached zero") was false (six of six) and its second claim (sites
"enumerated in the coverage map") named a cross-check column that does not exist; both were my own
unverified assertions, the same §4 defect (a domain excluding the form the thing takes) that made
Code's grep miss the markers. The discipline was working. **Ruling:** keep it; fix only the
spelling. §8 adopts the **provenance-header `crosscheck:` form** (no `#`) the tree already uses — it
belongs with `stock:`/`source:`/`oracle:`/`slice:`, and the `provenance` gate already parses that
header, so validating the `crosscheck:` line's *form* is a cheap gate extension (small, non-blocking,
owed to the provenance gate). **Presence** stays a review/checkpoint check — it is six of six now —
until drift ever argues for a presence gate. Charter §8 edited to match. **Closed.**
**Blocks nothing.**

---

_History (the original finding, kept because it was wrong and the wrongness is the lesson):_

Charter §8 requires a `# crosscheck: <library> <thing> — deferred …` marker "at every site that
wants a known-library cross-check, so the optional QuantLib tier is complete by construction, not
from memory." `grep -rn '# crosscheck:' src/` returns **nothing** across T1–T6 — so the mechanism
that is supposed to make the QuantLib tier complete by construction is currently empty, and the
tier would have to be assembled from memory, which is the exact thing the clause forbids.

`docs/checkpoints/CP_inflation_vertical.md` §4 originally asserted the markers were "placed at
every site that wants it". They are not; the claim is corrected in that document.

**Not silently fixed, because the choice is not mine.** Either the clause is enforced — and then
markers are owed at every JY and cap/floor site across five tranches, which is a tranche of work
and wants a gate (`verify.py crosscheck`) or it will drift again — or the clause overreached for a
library this young and should be narrowed to "the QuantLib tier names its own sites when it is
built."

**Re-open trigger:** when the QuantLib cross-check tier is actually scheduled, or at the next
checkpoint, whichever comes first. **Blocks nothing** — every tranche's own oracles are green and
the QuantLib tier is explicitly outside the reproducibility contract and every merge gate.
