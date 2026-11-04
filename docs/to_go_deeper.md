# To go deeper — per-document routes not taken

What each worked source has **more to give** that we chose not to build, stated plainly with a hint
of the route back in. This is not `OPEN.md` (in-flight debt with a re-open trigger) and not
`references.md` (sources held). It is the honest back-cover of each document: where we stopped, why,
and what it would take to go further — so a later return, or a Substack "part two", starts from a
map rather than a cold read.

**Rules.** One section per worked source (by its `references.md` key). Each route names: the gap, why
we stopped there, what you'd build, and the source it needs — marking whether that source is already
`in-hand`/`worked-from`, `referenced-by` (named by a held source, not obtained — §7's no-citation-by-
inference still applies), or new. A route is a *hint*, not a spec; it becomes a tranche only when
interest picks it (§1) and Cowork writes the real plan (§10).

---

## `BERHAN-2012` — *Pricing Inflation Derivatives* (ch 1–5 built, T2–T14)

Built: all four ch-1 instruments, the three models (JY, MM-I, MM-II) behind one seam, and the three
ch-5 calibrations — pricing analytics, verified, seasoned. What the document still holds:

1. **Risk & sensitivities (§1.3).** We priced; we never made the price *move*. L5 is empty — no
   deltas, no scenario engine. **Route:** an L5 risk layer over the existing engine + a `Priceable`
   protocol; bump-and-reprice first, analytic/AD sensitivities later. **Source:** none new — the
   pricers we have are enough. The most self-contained next step.

2. **HJM dynamics & simulation (§2.1–2.4).** We take the curves as given and never reconstruct them
   from drift conditions; there is no Monte Carlo anywhere. **Route:** a simulation engine for the
   nominal/real forwards under the JY/HJM dynamics — the thing that unlocks path-dependent and exotic
   inflation payoffs *and* gives MM-I/MM-II a **distributional** error bound. **Now in progress from
   the other side:** the Jäckel arc (project two) builds exactly this MC capability; once it has a
   forward-rate process, this route closes. **Source:** `JARROW-YILDIRIM-2003` (the drift conditions,
   `referenced-by` — must obtain); `MCINERNEY-ZASTAWNIAK` ch 4 (`in-hand`) for HJM/Gaussian; the MC
   craft from `JAECKEL-2002` (`in-hand`).

3. **The smile — the weakness all three models share.** Every model here is flat in strike; the
   implied-vol surface (T9) *represents* the market skew but nothing fits it, so OTM is underpriced
   and ITM overpriced (reproduced for both JY and MM-II). **Route:** a stochastic-volatility inflation
   model. **Source:** `BRIGO-MERCURIO-2005-SV` and `BRIGO-MERCURIO-2009-SABR`, or `KENYON-2008`
   ("Inflation is normal") — all `referenced-by: BERHAN-2012`, none in hand. The richest "part two"
   and the one the thesis itself keeps pointing at.

4. **A distributional bound for MM-I's freezing (not just a sensitivity).** T12(b) bounded the
   freezing error as a sensitivity (71% of σI per 200bp forward move); a probability-weighted bound
   needs the forwards simulated. **Route:** the LMM simulation (Jäckel ch 12), applied to the freezing
   recursion. **Source:** `MCINERNEY-ZASTAWNIAK` ch 5–6 (`in-hand`), `JAECKEL-2002` ch 12 (`in-hand`).

5. **The "practitioners approach" to the skew (§5.4.2, named, not taken).** BERHAN names implying a
   parameter set per Expiry/Strike pair as the alternative to the ATM-only restriction. **Route:**
   fit a parameter set per (expiry, strike) cell rather than one per expiry; a calibration variant,
   not a new model. **Source:** `BERHAN-2012` §5.4.2 (`worked-from`) — in hand.

6. **External cross-check (the QuantLib tier).** Six `crosscheck:` markers already name the sites;
   the tier itself is unbuilt. **Route:** stand up the optional, dev-only QuantLib tier against those
   six sites (never a merge gate — §8). **Source:** QuantLib (a library, not a paper).

**The one honest caveat that governs any return (FP-1):** BERHAN's empirical results are *plots, not
tables* (Figs 1.1, 1.2, 5.1–5.10). Nothing above licenses reading a number off a chart — a deeper
build reproduces the *structure* (a sign, a monotonicity, a reduction), with every number attributed
to the library and its oracle.
