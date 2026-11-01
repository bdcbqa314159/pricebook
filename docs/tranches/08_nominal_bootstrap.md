# Tranche 8 — the nominal curve bootstrap (§5.1), the cross-check presence gate, and the dispositions

**Status:** BUILT (v0.8.0). Decided by Cowork (§10 step 1), built by Code. **Results in §7.**
**Source:** `BERHAN-2012` (`worked-from`) §5.1 — the nominal discount curve bootstrapped from par
swap rates ("recovered from bootstrapping the nominal discount curve", p. 5). Single-curve, as the
thesis is.

This tranche carries three things that belong together: the §5.1 feature, the cross-check
**presence gate** (ruled in 0045), and the **per-module dispositions** that gate needs. All land in
one PR with the parked charter §8 edit and the §2 correction below.

## 0. The feature — bootstrap the nominal curve from par swaps

Today the nominal curve arrives **built** (handed to the snapshot). §5.1 constructs it: given par
swap quotes `S(Tᵢ)`, solve for the discount-factor pillars that reprice each swap to par. This is
**model-free market-data construction** — quotes in, a `DiscountCurve` out, no model, no engine.

## 1. Layer — L1, and the charter §2 line is corrected to say so

The bootstrap is **L1** (`market/`), beside T2's break-even bootstrap. It imports only L0 (the
solver seam, day counts, `Money`) and L1 (`DiscountCurve`); it does **not** touch a model or the
engine. §2 currently calls a curve bootstrap "L3" and names T2's break-even inversion as the
example — but that lives in `market/inflation_curve.py` at **L1**, so the clause misdescribes its
own example. **Charter §2 corrected** (on disk, lands with this PR): a *curve bootstrap* (quotes →
curve, L0+L1 only) is **L1**; *model-free parameter* calibration is **L3**; *engine-inverting*
calibration is **L4**. The split is by what the calibration produces and reads, not by taste.

## 2. Pieces

```
L0  (reuse) foundation/solvers.py — the sequential root-find per pillar
L1  a par-swap quote representation (schedule + the quoted par rate) — minimal, ≤5 fields
    market/nominal_bootstrap.py — quotes → DiscountCurve, composing ONE annuity atom
    the annuity/`Σ τ·df` atom, shared — so a future vanilla-IRS pricer composes the same one (§3d)
```

No new product priced through the engine; no model; nothing above L1.

## 3. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **the anchor — known-curve round trip, pointwise.** From a known `DiscountCurve`, compute the par swap rates it implies **by an independent formula in the test**, bootstrap a curve from those rates, and recover the known `df` at every pillar to 1e-12 | **1** | non-telescoping: the test computes the par rates itself, so a wrong bootstrap recovers the wrong curve |
| 2 | **companion: a sloped par-rate input yields a NON-flat curve** | structural | without it, oracle 1 passes an implementation that returns flat / ignores the later quotes |
| 3 | **reduction:** a single flat par rate ⇒ the curve whose constant zero reproduces it, closed form | **1** | the degenerate case with a known answer |
| 4 | **consistency guard (telescopes — kept as a guard, not the anchor):** the bootstrapped curve reprices the input quotes to par | self-consistency | guards that construction and the par formula share one annuity (§3d); cannot see a shared convention error, which is why oracle 1 is the anchor |
| 5 | **negative rates do not crash — failure is a value.** A negative par rate bootstraps to a `df > 1` fine; a genuinely unsolvable pillar returns `PricingFailure`, never a raise | structural | the stock's bootstrap hard-codes a `[1e-6, 1.0]` bracket and raises on a negative-rate root — **do NOT inherit it** (recorded in `_stock`'s own audit); the bracket must admit negative rates |
| 6 | monotone/duplicate/empty quotes → `PricingFailure` | structural | ill-posed input is a value, not an exception |

Identifiability is **not** a concern here as it was in T6: a sequential bootstrap is *exactly
determined* — pillar `i` is fixed by quote `i` given pillars `< i`, a triangular Jacobian — so there
is no valley. Oracle 1's pointwise recovery is the proof; no multi-start needed. State that in the
doc so the absence of a T6-style discriminator is a reason, not an omission.

## 4. The cross-check presence gate + the 7 dispositions (ruled)

The parked §8 edit (presence gate, explicit exemption over `market/`/`models/`/`engine/`) lands
**with its implementation** here. The `provenance` gate gains the presence check (a filter on the
`_src_modules()` loop it already runs). The dispositions Code asked me to rule — a target or
`none — <reason>`, my call per §8's criterion:

| module | disposition |
|---|---|
| `market/curve.py` | **`crosscheck: quantlib InterpolatedZeroCurve / log-linear zero interpolation`** |
| `market/nominal_bootstrap.py` (new) | **`crosscheck: quantlib PiecewiseYieldCurve / par-swap bootstrap`** |
| `market/curve_set.py` | `crosscheck: none — a keyed container of curves, no numerics` |
| `market/snapshot.py` | `crosscheck: none — a frozen market-state value bag, no numerics` |
| `models/discounting_model.py` | `crosscheck: none — carries the MarketSnapshot, computes nothing` |
| `engine/cash.py` | `crosscheck: none — PV = N·df; the df convention is cross-checked at market/curve.py` |
| `engine/jy_calibration.py` | `crosscheck: none — inverts the cap/floor forward map (tagged at engine/capfloor.py); QuantLib's inflation vol is surface-based, not a JY-to-floor calibrator` |
| `engine/registry.py` | `crosscheck: none — product→pricer dispatch, no numerics` |

The `jy_calibration` reason is a positive statement about QuantLib's design, not "I am unaware of
one" — the inverse has no distinct counterpart because QL's inflation vol is surface-based; its
forward map (the cap/floor) is where the external check lives, and that is tagged.

## 5. Stock
**Mine for content:** the stock's nominal bootstrap for the par-swap par formula and the sequential
method. **Do NOT inherit** its `[1e-6, 1.0]` bracket and uncaught `brentq` raise — the audit in
`_stock` records it crashes on negative rates, which violates failure-is-a-value (oracle 5 exists
because of it). Build the solve on the `foundation/solvers.py` seam with a bracket that admits
negative rates.

## 6. Process
Branch `tranche/08-nominal-bootstrap` (the §8 presence-gate commit is already parked there). `test`
(RED) → `feat` → `docs` → `chore(release): v0.8.0`. One PR carries: the bootstrap, the presence
gate + the 7 dispositions, the parked §8 edit, and the §2 correction. `source:` cites
`BERHAN-2012` §5.1.


---

## 7. Result — and §5.1 is MISPRINTED, twice, in both renderings

Built as specced: the bootstrap, the presence gate, the seven dispositions, the parked §8 edit and
the §2 correction. 146 tests, eight gates green, ruff and pyright clean, zero suppressions.

**The headline is a source defect, not a code defect.** Transcribing 5.1.1 and 5.1.2 in `-raw` and
`-layout` per §4, the two renderings **agree with each other** — and the equations still disagree
with each other:

* **5.1.1 carries a spurious leading minus**, which makes every par rate negative;
* **5.1.2 sums `j = 0 .. i−1`** where inverting 5.1.1 requires `j = 1 .. i−1`. The thesis defines
  `P(0,T0) := 1` on the same line, so the printed form adds a spurious unit to the annuity.

Verified numerically against a known flat 3% curve, recovering the 1y pillar:

| form | recovered `P(0,T₁)` | truth |
|---|---|---|
| `j = 1..i−1` (implemented) | **0.97044553** | 0.97044553 |
| `j = 0..i−1` (as printed) | 0.94089107 | 0.97044553 |

**This is a new sub-case of §4 worth ruling on.** Transcription independence and the dual-rendering
rule protect against a *rendering* mangling a formula — the v0.4.0 failure. They are powerless when
the **source itself** is wrong, because both renderings faithfully reproduce the typo. What caught
it is the **requirement** that a formula and its stated inverse actually invert: oracle 1 starts
from a known curve, so a misprint cannot survive it. Both misprints now have their own tests
(`test_the_printed_5_1_2_SUMMATION_would_NOT_recover_the_curve`,
`test_the_printed_5_1_1_SIGN_would_make_every_par_rate_negative`), so the correction is asserted
rather than asserted-in-a-comment.

## 8. Findings for Cowork (§10 — raised, not decided)

### 8.1 No root-find: 5.1.2 is closed form, and the repo already ruled on this
§2's pieces list "reuse `foundation/solvers.py` — the sequential root-find per pillar". Given the
pillars below `i`, corrected 5.1.2 is an exact rearrangement, so a numerical solve would be — in
this repo's own words at `market/inflation_curve.py:54` — "solving a closed form". Built closed
form, and the solver seam is untouched.

It is also the **strongest** version of oracle 5: with no bracket, there is nothing for a negative
rate to fall outside. `df > 1` comes out with nothing configured.

### 8.2 The stock was not the trap the spec describes — it had already learned this
§5 says the stock "hard-codes a `[1e-6, 1.0]` bracket and raises on a negative-rate root — do NOT
inherit it". Checked: `_stock/.../fixed_income/ois.py:222` reads
`brentq(objective, 1e-6, 3.0)  # wider bracket: handles negative rates`. The only `[1e-6, 1.0]` in
the whole stock is `credit/cds.py:950`, solving for a **survival probability**, where `[0,1]` is the
correct domain and not a bug.

The *instruction* is right and oracle 5 is kept. But the attribution is backwards: the stock's
comment is the lesson, not the trap — and "the stock had already fixed this" is worth knowing,
because it is the first time in this vertical the stock was ahead of us rather than behind.

### 8.3 Malformed input raises; it does not return `PricingFailure`
§3's oracles 5–6 ask for `PricingFailure`. Charter §3's failure-is-a-value governs
`engine.price(product, model)`; a market-data constructor is a trust boundary, and the established
sibling `real_curve_from_breakeven` raises `ValueError`. Two sibling constructors with opposite
error conventions is worse than either choice, so this one raises. Flagged rather than settled —
if the ruling is `PricingFailure`, both constructors should change together.

### 8.4 My own annuity test had no power, and the test caught it
`test_the_annuity_READS_the_schedules_own_day_count` asserted `swap_annuity != Σ df` to prove `ψ`
is read. On the annual fixture every period is exactly 365 days, so on ACT/365F `ψ = 1.0` and the
weighted sum **is** the plain sum: it asserted `!= ` against its own value and failed. Fixed with a
semi-annual fixture (`ψ ≈ 0.49`) and an assertion that the accruals are non-unit. The flat-fixture
trap, fourth appearance, this time in a test I wrote in this tranche.

## 9. The presence gate, and it was mutation-tested
`verify.py provenance` now requires every module under `market/`, `models/` and `engine/` to carry
either a real `crosscheck:` target or `crosscheck: none — <reason>`, the reason **required**. Both
halves were checked by mutation rather than assumed:

| mutation | result |
|---|---|
| delete T7's `fixing_series.py` marker — the exact omission that slipped through | **FAILS**, naming the module |
| `crosscheck: none` with no reason | **FAILS**, demanding the reason |
| restored | ok |

13 modules gated: 8 with real targets, 5 exempt with reasons.
