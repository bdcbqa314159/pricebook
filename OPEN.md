# OPEN — the debt ledger

Charter §8: no debt is silenced silently. Every suppression, shim, skip or load-bearing TODO is
here with a rationale and a re-open trigger, or it does not exist.

## Suppressions — the gated ledger

`verify.py debt` asserts **suppressions − entries = 0**, counting entries that begin `- [NG-`.
Nothing else in this file is counted, so a non-suppression item must not use that prefix.

**Currently zero.** No `# type: ignore`, `# noqa`, `# pragma: no cover` or `@pytest.mark.skip`
exists in `src/` or `tests/` (verified at v0.6.0).

## Other open items

### OI-7 · `MarketSnapshot` carries a `# fields-exempt` marker — OPEN (taken at tranche 60, 2026-11-19)

**The exemption.** `market/snapshot.py`'s `MarketSnapshot` declares `# fields-exempt:` with its
reason in place. Tranche 60 added a `commodities` store — the commodity quote side, keyed by
`CommodityAsset` — which took the class from five fields to six.

**Why the marker rather than a bundle.** `verify.py fields`'s own docstring names the category: *"a
legitimately-wide aggregate / output-record / config type is allowed with an explicit
`# fields-exempt: <reason>` marker."* `MarketSnapshot` is the tree's one aggregate — a clock, the
term structures, and **one quote store per keying dimension** (inflation prints by index, equity
quotes and skews by ticker, commodity quotes by symbol). Nothing in it is a loose primitive wanting
grouping, which is the smell §7's limit exists to catch; and the §6 precedent set by `survivals` and
`projections` is that a different keying dimension gets its own store rather than a `None` hung on an
existing one. So this is not §7's "un-bundled vocabulary" case, and raising the limit was never on
the table.

**What the alternative would cost, measured rather than asserted.** Bundling the four quote stores
(`fixings`, `equities`, `equity_skews`, `commodities`) behind one `quotes` value object takes the
class to three fields with room for the whole arc. It would read `market.quotes.equities` at **11
sites in `src/`** and change **26 construction sites across 12 test files**. That is affordable — but
it changes the **durable public contract** §0 names as in scope for this library, which is a design
ruling, and tranche 60 was specced as the arc's lightest tranche.

**Severity: none numerically.** No number moves either way; this is about where a boundary is drawn.
The risk the marker carries is that a seventh and eighth store accrete behind it unexamined, which is
exactly why it is ledgered rather than left silent.

**Re-open trigger:** Cowork's ruling on the `quotes` bundle (raised at the T60 handoff), **or** the
next tranche that would add a *seventh* field — whichever comes first. If the ruling is to bundle,
this entry closes with that refactor; if it is to keep the aggregate, this entry closes by recording
the ruling and the marker stands on it.

### OI-6 · `nominal_bootstrap._validate` admits a quote set that skips a pillar, then crashes — CLOSED (fixed at v0.61.0, tranche 59)

**The defect.** `market/nominal_bootstrap.py`'s `_validate` checks that each par-swap quote's
schedule *extends* the previous one (a prefix match on the payment dates). That admits a set which
**skips** a pillar — a 3y schedule does extend a 1y one — and such a set leaves **two** unknown
discount factors in a single par condition. The bootstrap then reaches for the pillar that was never
solved and dies with a bare

    KeyError: datetime.date(2028, 1, 1)

**Severity: low, and the bound is the point.** It crashes on malformed input; it ships no wrong
number. A quote set that skips a pillar is underdetermined, so there is no right answer being
computed badly — the only defect is that a **precondition surfaces as a crash instead of a
refusal**, which §3.4 is explicit about ("failure is a value — never raise-and-hope"). Every quote
set in the tree's own tests and fixtures is pillar-by-pillar, so nothing in the library is affected.

**How it surfaced, which is the familiar shape.** Tranche 54's projection bootstrap needed the same
precondition and stated it directly — **one equation, one unknown**: a par quote determines exactly
one new pillar, so a set must step one at a time — and refuses in domain vocabulary naming the
unsolved pillars. Writing the new thing correctly is what exposed the old roughness, the same way
tranche 49's adaptive-grid work surfaced OI-4. T54 did **not** reach into tranche 8's module: one
unit per tranche, and this is tranche 8's.

**The fix** is the one-line condition T54 already carries, plus its test: count the pillars a quote
introduces and refuse unless it is exactly one. It belongs in `_validate` beside the prefix check,
whose intent it completes.

**Re-open trigger:** the next tranche that touches `market/nominal_bootstrap.py`, or re-audit #6 —
whichever comes first.

**Closed at tranche 59**, by the re-audit #6 remediation, on the second of those triggers. The fix is
the one this entry prescribed: `_validate` counts the pillars a quote introduces — the payment dates
not already solved — and refuses unless it is exactly one, in domain vocabulary naming the unsolved
dates. A `ValueError` beside the prefix check, matching every other guard in `_validate` and T54's
identical refusal; **not** a `PricingFailure`, which is §3's *engine* return and would change an L1
curve constructor's type for every caller. *(The T59 spec's one-line shorthand said `PricingFailure`;
Code built to this entry's own prescription and raised the discrepancy rather than settling it — §10.)*

Two killers, not one: dropping the guard still **raises** (the original `KeyError`), so only a test
asserting the message is *ours* and names the unsolved dates discriminates — the T44b reachability
shape. A third test asserts a pillar-by-pillar set still bootstraps, so the guard is not vacuous.

One pre-existing fixture moved: `test_quotes_out_of_MATURITY_ORDER_are_refused` used `[3y, 1y, 2y]`,
whose **first** quote introduces three unknown pillars at once — legitimately caught by the new guard,
but for the wrong reason. It now walks `1y → 2y → 1y`, which isolates the ascending-maturity rule with
every quote legal under the pillar guard.

### OI-5 · `tests/L4/test_oi4_adaptive_truncation.py` was `slow-test-exempt` — CLOSED (removed at v0.56.1, tranche 53)

**The exemption.** The file declares `# slow-test-exempt:` with its reason in place. It establishes
the **no-arbitrage requirement over OI-4's mapped region** — twelve `(σ_v, T)` cells, four strikes
each — and every price there is a Fourier inversion at an extended grid (up to 32,001 nodes, two
characteristic-function evaluations per node). Measured **2.8s locally, ~11s on the Windows CI
runner** against the 10.0s ceiling.

**Why the exemption rather than a trim.** The region *is* the oracle's domain. §4's floor rule is
that an oracle must contain the failure, and this region is what caught a defect that had shipped
for sixteen tranches; cutting cells to fit a stopwatch trades the thing that works for the thing
that measures it. The work was already reduced as far as it goes without that trade: memoising the
pricer (lossless — the pricer is pure, §3.1), merging two redundant sweeps into one (prices that
fall with the strike and are non-negative at the highest strike are non-negative throughout), halving
the high-precision reference grid to twice the widest range the fix selects, and splitting the
arithmetic-floor oracles into their own file along a real subject seam. That took it from 5.9s to
2.8s and no further.

**The real fix, which is why this is debt and not a standing exemption.**
`gil_pelaez_probabilities` re-evaluates `φ(u)` at every node **for each strike**, although `φ` does
not depend on the strike — only the `exp(-iuk)` phase does. A smile therefore costs 4-10x what it
needs to, in production as well as in this file. Sharing one `φ` evaluation across the strikes of a
given model and maturity would bring this file comfortably back under the budget *and* make real
smile pricing several times faster. It was **out of tranche 50's scope by ruling** ("No reshaping of
the inversion where it was already correct" — the waste is not incorrect, just wasteful), so it is
raised rather than folded in.

**Resolution (tranche 53, `v0.56.1`).** The named trigger fired as intended: `φ(u)` is now evaluated
**once per `(model, maturity)`** rather than once per strike, so the region sweep's cells price their
whole strike ladder on one `φ` sweep. **The exemption marker is removed and the gate protects the
file again.**

**The measurement, since the closure was explicitly conditional on it (§9 — CI is the reading, not a
local stopwatch).** The marker was taken off *before* pushing, so the gate actually measured the file
rather than skipping it, and both runners pass: **ubuntu 3m49s, windows 6m55s**, with `slow` green.
Locally the file went **2.8s → 2.13s** wall (its test work 2.54s → 1.62s).

**And the honest note on the size of the win here.** OI-5's entry predicted 4–10×; the file improved
by ~25%, because its cost was never dominated by the region sweep alone — the high-precision
reference grid (64,001 nodes, 16× the committed one) is the dearest thing in it, and the SVJ sweep
next. Batching all three is what cleared the ceiling. The 4–10× figure is the saving on **one
maturity's `φ` evaluations**, which is what a smile-pricing caller sees; it was never the whole
file's runtime, and the ledger entry should have said so.

**What the fix is not.** No production *product* prices a smile today — `price(product, model)`
prices one option per call — so the saving is realised by the four test files that sweep or fit a
smile at one maturity, and prospectively by any smile consumer. The waste was in `src/` and the fix
is in `src/`; claiming a realised production speedup would be the prospective claim §6 refuses.

### OI-4 · The Heston/SVJ Fourier grid returned a NEGATIVE call price in the far short-dated wing — CLOSED (fixed at v0.55.1, tranche 50)

**The defect.** `engine/heston_fourier.py`'s committed inversion grid (`upper=200`) does not resolve
the deep out-of-the-money, short-dated corner at high vol-of-vol. Measured at
`HestonParams(kappa=2.5, theta=0.06, sigma_v=0.9, rho=-0.3, v0=0.02)`, 30 days, `K = 130` on a spot
of 100:

| grid | price |
|---|---|
| committed `upper=200, count=4001` | **−2.499781e-04** |
| reference `upper=1200, count=24001` | **+1.747280e-05** |

A **negative call price** — an arbitrage violation, not merely an inaccurate one — wrong by 2.7e-04
absolute. It has been on `main` since tranche 33, sixteen tranches.

**Why no oracle saw it.** This is §4's *"a reference is code too"* shape (T20a, T25) in its sharpest
form: tranche 33's oracles checked the engine against *the same kind of grid the engine uses*, so in
the one regime where that grid is wrong, the reference was wrong with it. Nothing in the suite priced
a deep-OTM short-dated option at high vol-of-vol against a high-precision reference. It is the same
defect **tranche 49 fixed one regime over** — SVJ's jump tails make `φ` decay even more slowly, which
is why `engine/bates_fourier.py` commits `upper=400` and oracles that constant's value; chasing that
measurement backwards is what surfaced this.

**Interim handling, and what was deliberately NOT done.** `tests/L4/test_fourier_inversion.py`'s
regression set is 54 of 60 points: the six that fall below `1e-3` absolute (quadrature residue rather
than prices, this one among them) are **excluded with the reason asserted in the test**, not pinned.
Pinning would have regression-locked a wrong number, which §8 forbids — debt may never buy a green
oracle. The exclusion is itself guarded by `test_THE_EXCLUDED_REGIME_is_named_and_still_fails`, which
asserts the committed grid *does* disagree with a fine one there, so the exclusion cannot quietly
become a hole.

**The fix** is a remediation tranche out of **re-audit #5** (the T45-from-re-audit-#4 precedent), not
a patch bolted onto the arc close: an adaptive or regime-aware truncation for the Gil-Pelaez
inversion, oracled against a high-precision reference *in the regime the present grid cannot
resolve*. A fixed wider constant is not obviously right either — it would cost every pricing call to
fix a corner — so the shape of the fix is part of the tranche, not assumed here.

**Resolution (tranche 50, `v0.55.1`).** Fixed by an **adaptive truncation** keyed to the integrand's
measured decay, `engine/fourier_inversion.adaptive_grid` — not a wider constant, which would have
taxed every pricing call to repair a corner. The range doubles from tranche 33's `upper = 200` until
`|φ(u)|/u` falls below `1e-13`, at fixed node spacing so only the truncation error moves; everywhere
the old bound already sufficed the search stops at once, so the benign path selects the same grid and
returns the same numbers.

**And the severity sweep found a second thing the ledger did not predict:** truncation error falls
under refinement only until it reaches an *arithmetic* floor and then stalls. `P1 = ½ + I/π` cancels
two `O(½)` quantities, so it carries an absolute noise floor of ~2e-13 whatever the grid, and a price
inherits `forward × 2e-13`. So `checked_call_price` reads a raw inversion output in three branches: a
price above the floor; `0.0` within it (zero is the honest rendering of a number smaller than the
arithmetic that produced it); and a **refusal** below it, because a materially negative call price
means the method failed rather than that the option is cheap. The refusal is what keeps the zero
honest — a bare clamp would have turned this very defect into a clean-looking `0.0`, and seven orders
separate the two.

Over a 600-point `(σ_v, T, K/F)` sweep: **113 materially negative prices on the shipped grid, zero on
the fixed one.** The mapped region reprices to a high-precision reference at 1e-6 relative or better.
Mutation: applied 17/17, killed 17, survived 0 — after two gaps the sweep found and the oracles did
not (the rule could walk to its 25,600 cap with every price still correct, and the SVJ engine never
reached the floor rule at all).

**The six excluded regression points stay excluded**, and deliberately: their bit patterns were
captured on the broken grid, so restoring them would pin the wrong numbers (§8). The regime's
correctness now lives in `tests/L4/test_oi4_adaptive_truncation.py`;
`tests/L4/test_fourier_inversion.py`'s guard records the repair rather than the defect.

**What remains, stated rather than closed silently:** prices below `forward × 1e-12` are returned as
zero because the Gil-Pelaez form cannot resolve them — a limit of the *parameterisation*, not of the
grid. A formulation that integrates the option value directly (Carr–Madan, or a COS expansion) would
not cancel that way; that is a new object, not a fix, and is recorded in `docs/to_go_deeper.md`.

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
