# OPEN — the debt ledger

Charter §8: no debt is silenced silently. Every suppression, shim, skip or load-bearing TODO is
here with a rationale and a re-open trigger, or it does not exist.

## Suppressions — the gated ledger

`verify.py debt` asserts **suppressions − entries = 0**, counting entries that begin `- [NG-`.
Nothing else in this file is counted, so a non-suppression item must not use that prefix.

**Currently zero.** No `# type: ignore`, `# noqa`, `# pragma: no cover` or `@pytest.mark.skip`
exists in `src/` or `tests/` (verified at v0.6.0).

## Other open items

### OI-1 · §8's crosscheck markers — REOPENED: the finding that opened this was WRONG
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

**Re-open trigger:** immediate — Cowork to re-rule on the corrected facts.
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
