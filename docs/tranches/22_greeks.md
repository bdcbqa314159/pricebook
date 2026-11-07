# Tranche 22 — Greeks by simulation: the last Jäckel unit, and the first L5 risk module

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.27.0`.
**Source:** `JAECKEL-2002` ch 11 — FD with path recycling (§11.3), pathwise differentiation (§11.5),
the likelihood-ratio method (§11.6), and the LR Vega/Rho appendix (§11.9). **Last of the Jäckel arc.**
Closes ch 11 (the ninth chapter), **opens the empty L5 risk layer**, and closes BERHAN's §1.3 route.
On landing, the §9 whole-tree re-audit fires at the Jäckel checkpoint.

## 0. The point — make the price *move*, anchored on Greeks we already own in closed form
Every model so far prices; none reports a sensitivity. Greeks by simulation give delta/gamma/vega from
the same paths, and — the anchor — **we have Black's analytic Greeks** (T9), so an MC delta/gamma/vega
must converge to them. This is the price-convergence discipline applied to the derivative of the price.

## 1. The methods (from Jäckel) and the anchor (ours)
- **FD with path recycling (§11.3):** bump `S₀`, reprice on the *same* paths (common random numbers),
  difference. The seed-sharing is what makes the difference low-variance — a direct use of the §3
  determinism invariant.
- **Pathwise (§11.5):** differentiate the payoff along the path, `E[∂payoff/∂S₀]` — valid only for a
  Lipschitz payoff.
- **Likelihood ratio (§11.6):** differentiate the density, not the payoff — a score-function weight,
  valid for a *discontinuous* payoff where pathwise is not.
- **Anchor:** all three converge to **Black's closed-form delta/gamma/vega** (T9), within the Greek
  estimate's own standard error (invariant 7 extends to a Greek — it is an estimate).

## 2. Pieces and layers — this OPENS L5
```
L5  risk/sensitivities.py — the first L5 module: a sensitivity computed ON the L4 engine. Per §2, L5
                            sits on the engine + a Priceable protocol. Whether L5 needs a NEW Priceable
                            protocol or bumps an input and calls the existing registry `price()` is a
                            build question — settle by what it reads, raise if unobvious (rule-of-two:
                            the FD method prices many products via `price()`, so the "thing that can be
                            bumped and repriced" may already be the registry interface).
L4/L0  the pathwise and LR estimators attach to the MC engine (they differentiate its payoff/density);
                            the FD method is pure bump-and-reprice over the existing engine.
```
This is a milestone: the layer box has had L5 empty since T1. It fills here, and the charter's "L5 on
the engine + a Priceable protocol" stops being an aspiration.

## 3. The oracle (RULED unless TO MEASURE)
- **RULED — exact anchor: MC delta/gamma/vega converge to Black's analytic Greeks** (T9) within the
  Greek's measured SE. The closed form we own is the judge, as for the price.
- **RULED — cross-method agreement: pathwise, LR and FD-path-recycling agree** on a smooth payoff
  (three independent estimators, different code each side — a cross-method oracle). A wrong one
  diverges from the other two.
- **RULED — the reduction+companion that is the whole point of having two methods: on a Lipschitz
  payoff (a call) pathwise is valid and matches; on a DISCONTINUOUS payoff (a digital) pathwise is
  invalid and LR is the one that holds.** Assert both — pathwise agreeing on the call is vacuous
  without the digital showing why LR exists.
- **RULED — the −½ rate on the Greek estimate**, and the null/averaging discipline (§4) where a
  sensitivity's expected value is zero (e.g. a Greek that must vanish by symmetry — average it).
- **RULED — determinism; failure-is-a-value** (a Greek that cannot be formed → a value, not a NaN),
  and the Greek carries its SE (invariant 7).
- **TO MEASURE — variance ranking:** report which method is sharpest for delta vs gamma vs vega (the
  §11.7 comparison), measured, not asserted from the book's figures (no number off a chart, §4).

## 4. Stock & process
Mine `_stock/.../models/mc_greeks*.py` and `risk/pathwise_greeks.py` for shape and the migration
edge cases; the pathwise/LR formulas are read from **Jäckel §11.5/§11.6/§11.9**, not the stock. The LR
Vega/Rho formulas (§11.9) are load-bearing — transcribe pointwise, and check whether Jäckel prints
them in full or leaves them to the reader (the Acklam/`(1−ρ²)` pattern — derive-and-assert if unprinted).
Branch `tranche/22-greeks`. `test` (RED pre-registered: MC delta vs Black delta) → `feat` → `docs` →
`chore(release): v0.27.0`. Mutation on the pathwise derivative and the LR score; the digital-payoff
companion must fail a pathwise estimate. `crosscheck: quantlib (MC Greeks)` on the L5 module.
`source:` cites `JAECKEL-2002` ch 11; names Black's analytic Greeks as the anchor. **Open as a DRAFT
PR** (§10 step 3).

## 5. What this closes
The nine-chapter Jäckel spine (2,3,4,7,8,9,10,11,12) is complete. Write `CP_jaeckel_arc.md`, finalise
the `to_go_deeper` Jäckel section, and **run the first §9 whole-tree re-audit** (T1–T22, independent,
via the `analysis/` pattern) — recording it in `docs/audit_log.md`. Then the next arc is Bernardo's
interest-pick (he has a source in mind).


---

# BUILT — results (appended at build; §0–5 above are Cowork's spec, unedited)

`v0.27.0`. 818 tests, all eight gates green, ruff and pyright clean. **Nineteen mutations, all
nineteen killed on the first pass.** `models/black.py` gains the analytic Greeks;
`risk/sensitivities.py` is **the first L5 module**. Chapter 11 closes, and with it the nine-chapter
Jäckel spine.

## 6. The anchor did not exist — the spec assumed a capability we never built

§0 and §1 both name *"Black's analytic Greeks (T9)"* as the anchor. **There were none.** Tranche 9
built the implied-vol *surface* (`FloorQuote`, `ImpliedVolSurface`, `implied_vol_surface`); a grep of
`src/` for any `delta`/`gamma`/`vega` definition returned a `flat` correlation constructor and an
unrelated `delta` in a Jarrow–Yildirim comment. Nothing else.

So the tranche built its own anchor first, and the important consequence is that **the anchor needed
its own oracle rather than being assumed**. The chain is now four links, each independently checkable:

`black_price` (verified since tranche 5) → its central difference → the analytic Greek → the Monte
Carlo Greek.

Worth noticing as a pattern: a spec can assert a *capability* as confidently as it asserts a number,
and this is the first time the missing thing was a whole function rather than a value. The cheap check
is a grep before the build, not after.

## 7. The analytic Greeks are proved as a LIMIT, not as a point comparison

The first version compared each Greek to one central difference at `h = 1e-5` and **failed** on the
deep out-of-the-money case by ~4e-6 relative. The finite difference was the weaker side. Varying the
step settles it:

| step | delta's relative gap |
|---|---|
| 1e-3 | 5.63e-02 |
| 1e-4 | 5.60e-04 |
| 1e-5 | 5.60e-06 |
| 1e-6 | 5.60e-08 |
| 1e-7 | 3.28e-10 |

Textbook `O(h²)`, converging **onto** the analytic value — so the formula was right and the reference
was truncating, by a hundred times the comparison bound. The oracle now asserts the **rate**, which
proves the closed form is the *limit* of the numerical derivative: a different and stronger claim than
being near one approximation of it, and one a wrong `d₊` could not fake with a clean quadratic
approach to the wrong number.

Gamma's floor is looser (`1e-5`, steps from `1e-2`) because a second difference divides by `h²` and
feels cancellation sooner. That is the reference's limit, parameterised and stated rather than papered
over with one bound for both. Measured convergence ratios: **87 to 102** per tenfold step.

## 8. The reduction and its companion — what earns the second method

Pathwise matching on a call is **vacuous on its own**; finite differencing would do. What justifies a
second method is a payoff where pathwise is not merely less accurate but *wrong*. A digital's payoff is
flat almost everywhere, so its pathwise derivative is **exactly 0.0 on every path**, while the true
delta is strictly positive — and the oracle asserts the zero as a number, against an analytic digital
delta built for the purpose. Jäckel: the method *"cannot be readily applied since the payoff function
is not even continuous, let alone Lipschitz-continuous."*

Both halves are asserted, which is the reduction-plus-companion rule: pathwise agreeing where it is
valid, and failing by a stated amount where it is not, with the likelihood ratio holding there.

**Pathwise is also refused for gamma**, because §11.5 refuses it: *"for the calculation of Gamma, we
still have to implement a finite differencing scheme for two individually calculated Deltas."* It
gives a first derivative per path, not a second. A method list that quietly accepted it would be
inventing a capability, so it raises instead.

## 9. The variance ranking — and it inverts the obvious guess

**TO MEASURE, answered.** 200,000 paths, at-the-money, `σ = 20%`, one year:

| Greek | finite difference | pathwise | likelihood ratio |
|---|---|---|---|
| delta | **0.25%** | **0.25%** | 0.56% |
| gamma | 4.35% | *refused* | **1.44%** |
| vega | **0.42%** | **0.42%** | 1.44% |

(standard error as a percentage of the analytic value)

The likelihood ratio is ~2–3× noisier for delta and vega — as Jäckel's caveat implies, its weights
carry `1/(σ√Δt)` — but it is **3× sharper for gamma**, which is the reverse of the naive reading that
the most general method is always the loosest. The reason is on the other side: a second finite
difference divides by `bump²` and amplifies the noise it is differencing.

**And FD's gamma is visibly off-centre:** 55.47 against an analytic 49.62, which is 2.7 standard errors
— inside the oracle's 4-SE band, but a truncation bias rather than noise. So for gamma the likelihood
ratio is both sharper *and* better centred, and the ranking is not a matter of taste.

## 10. L5 is open, and deliberately without a protocol

The charter's layer box has carried `L5 RISK & CAPITAL — on the engine + a Priceable protocol` since
tranche 1, and L5 has been empty for twenty-one tranches. It fills here.

**No `Priceable` protocol, and the spec left that to the build.** What this module reads is Black's
closed form and the Monte Carlo machinery directly — one shape of consumer — so a protocol would be an
abstraction with a single implementation, which §6 forbids. It arrives at the second kind of priceable
that needs differentiating. The charter clause is now half-satisfied and says so.

`BlackState` is the bundle §7's five-argument limit forced: `black_mc_vega` was six arguments, and
§7's prescribed answer is a frozen value object rather than raising the limit or suppressing the
check. Four consumers share it. `years` is read only by vega — not a carried-but-unread field, because
`total_variance = σ²·years` means the horizon is what separates a one-year 20% option from a four-year
10% one, which is §6's constitutive-of-identity test.

## 11. Findings for Cowork (§10 — raised, not decided)

1. **The spec's anchor did not exist (§6).** Recorded as the pattern rather than just the fix: a spec
   can assert a *capability* with the same confidence it asserts a number. Three tranches have now
   corrected a spec premise about a measurement; this is the first about a function. Worth a line in
   §10 if you think the spec-writing side can carry one — *check a named anchor exists before it is
   ruled* — or nothing, if the build-raises-it loop is considered sufficient (it was, here).
2. **`norm_pdf` is private to `models/black.py`.** Four Greeks consume it; `foundation.distributions`
   is where it would belong if anything else wanted it, and nothing does — §6's rule-of-two has not
   fired. It is the obvious thing to promote the moment a second module needs a density.
3. **No `Priceable` protocol (§10).** Confirm that leaving the charter's L5 clause half-satisfied is
   right, versus building the protocol now on one consumer.
