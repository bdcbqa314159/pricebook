# T68 — Bermudan swaptions in the Hull-White model

**Cowork spec, 2026-11-20.** Third tranche of the single-curve short-rate arc (project nine,
`MCINERNEY-ZASTAWNIAK`; ch 3.6). The tree already prices a Bermudan swaption — but in the **LMM** (T18); this
prices it in a **one-factor short-rate** model, where the single Markov state `r` makes an *exact* backward
induction possible and gives a second, independent route to the same instrument. The headline oracle is the
**cross-model agreement** (HW Bermudan vs LMM Bermudan on a matched calibration), which no single-model check
can be.

## Target
Price a Bermudan swaption under the Hull-White model —
`price(BermudanSwaption, HullWhiteModel) → PricingResult`.

## Two genuinely-new pieces (step-1 greps, 2026-11-20)
The step-1 grep found the reuse is **not** free, and flagging it is the point:
- **There is NO Hull-White European swaption in the tree.** `engine/swaption_mc.py::price_european_swaption`
  is LMM/MC; `swap_rate_vol.py` is the LMM swap-rate vol. So the one-exercise reduction anchor must be built.
- **The T18 exercise-boundary machinery is LMM-coupled.** `exercise_boundary.py` (`train_slice`,
  `roll_continuations`, `exercise_now`, `BoundaryState`, `TrainingSlice`) is a general Longstaff–Schwartz
  regression, but `engine/bermudan_swaption_mc.py::price_bermudan_swaption` takes a `LiborMarketModel` and an
  LMM `BermudanSwaption` with forwards/tenors. The HW Bermudan is the **second consumer** of the
  Longstaff–Schwartz backward induction — so §6's rule-of-two fires on *that regression*, and whether to
  generalise it is the design question below.

### Piece 1 — the Hull-White European swaption (Jamshidian), new, L4
A one-factor swaption decomposes (Jamshidian) into a **portfolio of zero-coupon bond options at a common
critical rate `r*`**: find the `r*` at which the underlying swap is at the money (a 1-D root-find), then each
coupon's bond option is `zero_bond_option` (T10, **reused**) struck at the bond price at `r*`. The new code
is the `r*` root-find and the portfolio assembly; the option valuation is the held closed form. **Oracle:**
transcribe the Jamshidian decomposition from the source (or the standard reference the source cites — if
McInerney states it without proof, read-before-asserting applies, and the decomposition is a *requirement*
oracle too: at `r*` every bond option is struck exactly at the forward bond, so the portfolio value equals
the swaption by construction). Cross-check against an **MC** HW swaption (simulate `r`, value the swap,
discounted payoff) — the closed form inside the MC band.

### Piece 2 — the Hull-White Bermudan, new, L4
Backward induction over exercise dates. **The method is a design call (below); the spec names the oracle
obligation, not the method** (§4): whatever the method, (a) a **one-exercise Bermudan equals the European
swaption** (piece 1) exactly, and (b) the Bermudan is bracketed by the T18 **structural sandwich** (foresight
bias above, policy suboptimality below — a width that does *not* shrink with paths, T18/T19, so it is not
sampling error).

## The design question — raise at review with counts, do not pre-settle (§6/§10)
The Longstaff–Schwartz regression (`exercise_boundary.py`) is a shared **shape** with a second consumer now.
Two honest options, and it is Code's to rule *with the coupling measured*, as you did for `MarketSnapshot`
and the Asian family:
- **Generalise the regression** to operate on simulated paths + a payoff callback, model-agnostic (LMM or HW
  supply the paths), so both Bermudans call one backward induction. The rule-of-two abstraction — *if* the
  coupling is a shared shape and not merely a shared word.
- **A separate HW Bermudan** (an `r`-lattice / trinomial tree, exact and MC-free — the classic HW method, and
  arguably the *better* approach since one Markov state needs no regression). Then the two Bermudans share the
  word "Bermudan" and not the method, and stay separate (the §6 real-leg-vs-YoY-leg line).
Report the read/construction-site counts and the coupling, and rule. My weak prior is the **lattice**
(exact beats a regression where the state is one-dimensional, and it sidesteps the LMM coupling entirely) —
but it is your call at the build, and a lattice is new infrastructure that later tranches (American options,
the ch-8 smile models) would reuse, which is a point in its favour.

## Oracles (RED-first, §4)
1. **Piece 1 pointwise:** the Jamshidian `r*` and the portfolio; the requirement that at `r*` the portfolio
   *is* the swaption (exact by construction); MC cross-check of the HW European swaption.
2. **One-exercise Bermudan = the European swaption** (exact reduction) — the companion exercising a genuine
   multi-exercise Bermudan (T39: the one-exercise reduction is blind to the exercise policy).
3. **Cross-model: the HW Bermudan vs the T18 LMM Bermudan on a matched European-swaption calibration** — the
   two must agree **within the structural bracket** (not a tolerance on a number — the bracket width is the
   foresight+suboptimality gap, invariant under path count, T18/T19). This is the arc's headline oracle: two
   independent models, two independent methods, the same Bermudan. The calibration-matching premise is
   load-bearing — assert the two models reproduce the **same** European swaption first (else the Bermudans
   price different instruments).
4. **Requirement oracle: Bermudan ≥ European** (more exercise rights never hurt), strict for a genuine
   multi-exercise schedule; and `≤` its perfect-foresight upper bound (T18's `perfect_foresight_price`
   pattern).
5. If the lattice route: oracle the backward-induction **steps pointwise** (the continuation roll, the
   exercise decision, the `r`-grid transition), not just the output — a price bound cannot see the algorithm
   (T18's §12.8 lesson).

## Degeneracy premises (§4 / §9)
- The one-exercise reduction's companion is **non-degenerate in the exercise count** (≥3 exercise dates,
  spread in time — T39/T18).
- A **non-flat curve** and **non-trivial mean reversion** so the Jamshidian `r*` is a real root (a flat curve
  can make the swaption degenerate).
- The cross-model fixture must calibrate both models to the **same non-trivial** European swaption (the
  matching premise is where a silent mismatch would hide).

## Provenance / gates
- Header `source: MCINERNEY-ZASTAWNIAK §3.6` (+ the Jamshidian reference if cited); `oracle:`; `crosscheck:`
  (QuantLib has a HW swaption engine and a tree-based Bermudan — a real target).
- Version → **v0.70.0**; one floor bump; one `CHANGELOG` entry. Re-time any MC/lattice on **both** CI runners
  (§9).

## Out of scope (carried / deferred)
- **Bermudan in G2++** (two-factor) — `to_go_deeper` (a two-factor lattice/regression is a real step up).
- **American (continuous-exercise) swaptions** — `to_go_deeper`; a dense-exercise Bermudan approximates one,
  but the continuous limit is its own unit.
- Full co-calibration of HW and LMM to a market surface — the cross-model oracle matches a *single* European
  swaption, which is all the Bermudan comparison needs; a surface co-fit is `to_go_deeper`.
