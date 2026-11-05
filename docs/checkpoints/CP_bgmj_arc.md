# Checkpoint — the BGM/J arc, and Monte Carlo in the library (T15–T18)

**Date:** 2026-11-05 · **By:** Cowork (step 4) · **Baseline:** v0.21.0, PR #25 · **Source:**
`JAECKEL-2002` (`worked-from`), ch 2/9/12 + §12.5–12.8; `MCINERNEY-ZASTAWNIAK` §5.9/§5.12 (the
swaption vol). Second source of the reset repo, after `BERHAN-2012`. First checkpoint of the MC line.

## 0. What is built — Monte Carlo, from a seeded draw to a Bermudan

| tranche | capability | anchor |
|---|---|---|
| T15 | the MC spine — seeded PCG64 + Box-Muller + estimator (L0), Black by simulation | our `black_price` (convergence within SE, −½ rate) |
| T16 | the LMM evolved by MC — factorisation, the drift (eq 12.8), the path engine (L4) | a caplet reprices to Black; each forward a martingale |
| T17 | the European swaption under the LMM | exact: a one-period swaption = the T16 caplet; bounded: Rebonato §5.12 |
| T18 | the Bermudan swaption — Jäckel's parametric exercise boundary | exact: a one-date Bermudan = the T17 European; sandwiched otherwise |

**v0.21.0 · 657 tests · eight gates green on both OSes · ruff + pyright clean · zero suppressions.**
Three MC products behind one path engine; one closed-form anchor (`black_price`) under the whole
tower, each tranche anchored on the one before it.

## 1. The spine of the method, in one line
A simulation is only as trustworthy as the closed form it is made to reproduce. The arc never had an
MC "reference number" from Jäckel (his results are convergence diagrams, §14.4) — every price was
proved against *our own* exact forms: Black, then the caplet, then the European. The library built its
own judges first (the whole inflation arc) and the MC tower stands on them.

## 2. Oracle shapes this arc added to §4
The MC material forced genuinely new oracle discipline, each earned from a surviving mutant or a failed
premise, each now in §4:
- **statistical-but-deterministic** — a fixed-seed MC is a pure function, so `|MC − closed form| < k·SE`
  is a hard assertion, and the **−½ convergence rate** is the requirement oracle no single N can fake.
- **presence-and-sign complete yet magnitude-silent** (T16) — thirteen drift oracles blind to the
  drift's size because the anchor's measure zeroed it; killed by a two-numeraire one-seed identity.
- **the one-seed identity's boundary** (T17) — it cancels only a quantity in the *drift*; a correlation
  in the factorisation is pinned by realised covariance instead.
- **bands must separate SE from approximation error and assert their resolution** (T17) — a flat 3%
  band resolved only a 20% defect; rebanded `k·SE + measured slack` with a power test.
- **two sources are independent only if they render the same formula** (T17) — McInerney froze the
  swap-rate weights, Jäckel differentiated them; different-order approximations, not a cross-check.
- **a price bound cannot see the algorithm** (T18) — procedure steps need pointwise oracles; four of
  five survivors were induction steps no sandwich caught.
- **dead code is defined by test-reachability, not improbability** (T15); **invariant 7** (§3) — an
  estimate carries its standard error.

## 3. Source defects — Jäckel, two, both internal or method-level
Recorded in `docs/references.md`. (1) **eqs 12.38/12.39 sign** — the printed exercise condition
exercises a deep-OTM payer; ruled by the finance, invisible wherever `p_i3` is fitted. (2) **eq 12.41
cannot be trained by the prescribed BFGS** — a piecewise-constant objective; Nelder-Mead substituted.
**Tally, corrected at commit** (the draft read "seven across the two books (five BERHAN, two
Jäckel)", which omitted McInerney's and counted two books where three are involved): **five
`BERHAN-2012`** (5.1.1, 5.1.2, 4.1.3, 4.2.2, §5.3.1 — the 2.5.11 mis-nesting was *ours*, not the
source's, and `CP_inflation_book_complete.md` §2 says so), **one `MCINERNEY-ZASTAWNIAK`** (`d₋`'s
dropped radical, caught because the literal reading prices a call below zero), and **two
`JAECKEL-2002`** — of which only the 12.38/12.39 sign is a misprint; eq 12.41's untrainability is a
limitation of the *method*, not of its printing. Every one found by an oracle whose domain was widened
to reach it, **none by review**.

## 4. The two-surface split, measured again
Cowork's review set the oracle choice and caught weak discriminators; Code's execution corrected
**four Cowork spec premises in one tranche (T17)** and two more in T18 — the cross-source independence
that did not exist, the one-seed instrument that did not generalise, the training-monotonicity that did
not hold. Each correction made the oracle stronger and is recorded. The split works in both
directions, which remains the whole argument for it (§10).

## 5. Honest scope — what the arc did NOT build
- **No non-recombining tree** (Jäckel ch 13) — his independent cross-check for the Bermudan; the
  sandwich self-diagnoses instead. `to_go_deeper`.
- **No tight upper bound** (Andersen–Broadie / Rogers) — the Bermudan sandwich is **15.75%** wide; the
  lower bound is the quotable price, the width the honest resolution. `to_go_deeper`.
- **No variance reduction, no Sobol, no Greeks-by-simulation** — each a real unit, none yet pulled by a
  product (§1). The L0 machinery is ready for them.
- **The Bermudan exercise boundary is three parameters** — at capacity for the test trade; a richer
  boundary is a route, not a gap.

## 6. Next — Bernardo's
The arc's headline (the Bermudan) is landed. Open directions, none urgent: variance reduction or Sobol
(speed, pulled by a slow product), Greeks-by-simulation (and BERHAN's §1.3 risk route with it), the
tree or the tight dual bound (tightening the Bermudan), the smile (a new source), or the first Substack
piece — the inflation and MC arcs are both a written spine now.
