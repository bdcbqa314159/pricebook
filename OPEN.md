# OPEN — the debt ledger

Charter §8: no debt is silenced silently. Every suppression, shim, skip or load-bearing TODO is
here with a rationale and a re-open trigger, or it does not exist.

## Suppressions — the gated ledger

`verify.py debt` asserts **suppressions − entries = 0**, counting entries that begin `- [NG-`.
Nothing else in this file is counted, so a non-suppression item must not use that prefix.

**Currently zero.** No `# type: ignore`, `# noqa`, `# pragma: no cover` or `@pytest.mark.skip`
exists in `src/` or `tests/` (verified at v0.6.0).

## Other open items

### OI-1 · §8's `# crosscheck:` markers have ZERO instances in the tree
**Opened:** 2026-11-01, at the T1–T6 checkpoint. **Found by:** verifying the checkpoint's claims.

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
