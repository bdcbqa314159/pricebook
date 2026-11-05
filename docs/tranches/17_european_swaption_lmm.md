# Tranche 17 — the European swaption under the LMM, anchored by an exact reduction and a bounded approximation

**Status:** DECIDED by Cowork (§10 step 1), awaiting build. **Target:** `v0.20.0`.
**Sources — co-equal primary, NOT background (Bernardo, 2026-11-05):** `JAECKEL-2002` §12.3–12.4
(the swaption in the MC framework, calibration to European swaptions) AND `MCINERNEY-ZASTAWNIAK` §5.9
(the swap market model: swaption = Black under the swap measure with swap-rate vol `v_{0,n}`) and
**§5.12 (the LMM approximation for the swaption volatility** — the correlation-weighted swap-rate vol).
Both held. **The two books render the same swap-rate-vol approximation independently → cross-source
transcription**, the strongest independence we have (a misprint must appear in *both* authors to
survive — stronger than one book's `-raw`/`-layout`). **Second of the three-unit BGM/J arc** (T16 →
T17 → T18 Bermudan).

## 0. The crux — the LMM swap rate is NOT lognormal, so there is no exact closed form

McInerney states it directly (p.108): under the LMM the forward swap rate is not lognormal, so a
European swaption has **no exact LMM closed form**. The anchor is therefore not a single closed form
but a pair — an **exact reduction** at the degenerate case and a **bounded approximation** in the
general case — exactly the reduction-plus-companion discipline (§4):

- **Exact: a one-period swaption IS a caplet.** A swaption on a single-period swap reduces to a caplet
  on that forward, which reprices to our Black by construction (T16's caplet anchor). So the
  degenerate `n=1` swaption has an exact anchor we already own.
- **Bounded: the multi-period swaption.** Its price depends on the **correlation** between forwards
  (the swap rate is a basket), which the caplet strip cannot see. Oracled against the §5.12 /
  §12.4 swap-rate-vol approximation to that approximation's **stated error** — bounded, never dressed
  as exact (the MM-I freezing lesson).

## 1. The method (from the sources) vs the anchors (ours)

- **Transcribed (§4), cross-source:** the LMM swaption-vol approximation — the swap rate written in
  terms of the forwards, and its volatility `σ_{0,n}` as a correlation-weighted combination of the
  forward vols (McInerney §5.12, Jäckel §12.4). Read in **both books** and reconcile; read each
  `-raw`+`-layout`. This is the load-bearing formula.
- **Ours:** the one-period reduction to the T16 caplet (exact); the T16 path engine for the MC; our
  `black_price` for the Black swaption shell.

## 2. Pieces and layers

```
L2  products/swaption.py      — a European swaption: the underlying swap schedule, strike, expiry,
                                payer/receiver. Frozen pure data (§6 ≤5 fields; bundle the swap leg).
                                The FIRST genuinely product-shaped MC consumer (cf. the no-vanilla
                                finding — a swaption is the rates world's basic option, and it has a
                                real consumer).
L3  models/swap_rate_vol.py   — the §5.12 LMM swaption-vol approximation (correlation-weighted
                                swap-rate vol) + the Black swaption shell (reuse black.py). Model-free
                                analytic. If nothing beyond the oracle consumes it, it may sit as the
                                black_mc precedent did — Code's call by whether a non-oracle consumer
                                exists; raise it if unobvious.
L4  engine/swaption_mc.py     — the swaption MC pricer: simulate the LMM to expiry via T16's engine,
                                form the swap rate from the terminal forwards, discount the payoff,
                                estimate + SE. The engine's first product-shaped use.
```

## 3. Rulings

- **The engine stays terminal-only; the path-return widening is DEFERRED to T18 (§6, rule-of-two).**
  T16's `simulate_terminal_forwards` returns the forwards at `grid.horizon`. A European swaption reads
  the whole curve at expiry, so set `grid.horizon = expiry` and it is served as-is. The Bermudan (T18)
  is the consumer that needs **intermediate** states (forwards at each exercise date); it is a
  hypothetical future at T17, so widening the engine now would be the speculative surface §6 forbids
  (the same mistake as T15's premature `mc_european`). When T18 arrives the engine widens
  endpoint→path and the swaption reads the last slice — a small, principled refactor at the real
  second consumer. **T17's oracles are written against the terminal shape; that is correct, not
  short-sighted.** If Code finds the widening genuinely cannot be deferred without a corner, that is an
  `Ask:`, not a silent widening.
- **Co-equal sources, cross-source transcription.** McInerney §5.12 and Jäckel §12.4 are both primary;
  the swap-rate-vol formula is transcribed from both and reconciled. Promote McInerney from the
  "background" wording in the LMM header to a cited `source:` on `swap_rate_vol.py`.

## 4. The oracle

| # | oracle | rung | why it can go red |
|---|---|---|---|
| 1 | **exact reduction: a one-period swaption = the T16 caplet = our Black**, to tolerance | **closed form (ours) — exact anchor** | a swap-rate/annuity construction error that survives into the degenerate case |
| 2 | **companion (non-vacuity): a multi-period swaption's price MOVES with the forward-forward correlation** — caplet-strip value at ρ→structure vs the correlated swaption | **structural** | a swaption that ignores correlation (passes 1 trivially) — the reduction-needs-a-companion rule |
| 3 | **bounded cross-check: MC multi-period swaption ≈ Black(§5.12 approx swap-rate vol)** within the approximation's **stated** error — the error measured and reported, never a tuned tolerance | 2 — bounded approximation | a swap-rate-vol transcription error; the bound names the freezing slack honestly (MM-I lesson) |
| 4 | **product-value transcription of the swap-rate vol** (annuity × forward swap rate × correlation-weighted forward vols): assert the **value** against the cross-source (both books) transcription, aimed at the **correlation term** — the factor most likely present-and-signed-correctly yet wrong in magnitude (T16's lesson) | **1 — pointwise + requirement** | a correlation entering with the wrong weight/sign in the swap-rate vol |
| 5 | **correlation contribution pinned NOISE-FREE** (T16's instrument): two LMM configs differing only in ρ, one seed — the swaption-price difference is a near-deterministic function of Δρ, compared to the §5.12 approximation's ρ-dependence, with MC error cancelled rather than fought | 1 — identity-style | the correlation's magnitude in the full MC price, which oracle 3's SE band would blur |
| 6 | **the −½ rate; determinism; failure-is-a-value** — same seed ⇒ byte-identical; a degenerate/empty swap or n_paths≤0 → `PricingFailure` | structural | §3 invariants, a discretisation bias |

Oracle 1 is the exact anchor (the degenerate caplet); oracle 3 is the honest bounded check for the
general case; oracle 4 + 5 are where the correlation — the swaption's whole reason to need the LMM —
is pinned by value, not by presence. **Degenerate-fixture guard:** do NOT oracle correlation with a
flat ρ=1 or a single-forward swap (correlation goes invisible — the recurring trap); use a
multi-forward swap with non-trivial, non-uniform ρ_ij.

## 5. Stock
`parked/ng/.../swaption_mc.py` is the nearest structural hint; read for shape only. The swap-rate
vol, the swaption Black formula and the approximation are read from **McInerney §5.9/§5.12 and Jäckel
§12.3–12.4**, not from the stock. Nothing imported from `_stock/`.

## 6. Process
Branch `tranche/17-european-swaption`. `test` (RED **pre-registered**) → `feat` → `docs` →
`chore(release): v0.20.0`. **Mutation on the swap-rate-vol's correlation weighting** (the T16
magnitude lesson — a present-and-signed-correct-but-wrong-magnitude correlation is the target), and on
the annuity/swap-rate construction; oracle 1 must fail under a swap-rate error, oracle 4 under a
correlation-weight error. `crosscheck: quantlib (European swaption / swap-rate vol)` on the model and
the engine. `source:` cites `JAECKEL-2002` §12.3–12.4 and `MCINERNEY-ZASTAWNIAK` §5.9/§5.12; names the
T16 caplet and `black_price` as the exact anchor. **Open as a DRAFT PR** (§10 step 3).

## 7. What this unlocks (T18, not here)
The Bermudan swaption — the exercise boundary (Jäckel §12.6–12.8), oracled by the
European-swaption lower bound (this tranche) + the non-recombining-tree sandwich (Jäckel ch 13). T18
is where the engine's path-return widening lands, pulled by the exercise-date intermediate states.

---

# BUILT — results (appended at build; §1–7 above are Cowork's spec, unedited)

`v0.20.0`. 604 tests, all eight gates green, `ruff check src/pricebook` and pyright clean.
`models/swap_rate_vol.py` (L3), `products/swaption.py` (L2), `engine/swaption_mc.py` (L4), plus the
correlation widening §2 implied. Thirteen mutations run; twelve killed, one proved equivalent.

## 8. The cross-source independence the spec counted on does NOT exist

§1 specified the swap-rate-vol approximation as transcribed from **both** books and reconciled —
*"the strongest independence we have (a misprint must appear in both authors to survive)"*. Read in
both, they are **not one formula**.

Both write the swap-rate variance as the quadratic form `Z·C^FRA·Zᵗ`. They differ in `Z`:

| source | `Z_k` | what it is |
|---|---|---|
| McInerney eq 5.40 (Rebonato) | `w_k F_k / S` | the weights **frozen** at time `t` |
| Jäckel eqs 12.18 / 12.21 | `(∂SR/∂f_k)(f_k/SR)` | the **exact** derivative |

Jäckel's 12.21 is McInerney's first term plus two more, which are exactly the terms arising because
the weights themselves depend on the forwards. McInerney's own **Remark 5.13** says 5.40 can be
improved by not freezing them — so these are two approximations of *different order*, not two
renderings of one. Measured gap in the vol:

| span | Rebonato | Jäckel exact-Z | gap |
|---|---|---|---|
| (0,6) | 0.14921440 | 0.14854584 | **0.450%** |
| (1,4) | 0.15165680 | 0.15134017 | 0.209% |
| (2,4) | 0.14354306 | 0.14331807 | 0.157% |
| (3,3) | 0.14094285 | 0.14086253 | 0.057% |

So the planned cross-source transcription is unavailable, and claiming it would have been false. What
replaced it is weaker in kind but still real: the pointwise transcription of 5.40 (six spans), plus
Jäckel's form as an **independent companion** whose gap is asserted to be both small *and non-zero* —
a zero gap would mean the companion had frozen the weights too, which is the failure that guards.

**The signature that proves the two `Z` vectors are genuinely different objects:** `Σ Z_frozen = 1`
**exactly** (it is `Σ w_kF_k/S = S/S`), while `Σ Z_exact` is 0.9955, 0.9979, 0.9984, 0.9994 across
the four spans. That identity is also the sharpest available check on the weights, and the weights
are where the magnitude risk lives.

## 9. Two fixtures were degenerate in ways inspection did not catch — found by mutation

Thirteen mutations, three survivors on the first pass. Two were my fixtures, not the code, and
neither is the flat-`τ` degeneracy the charter already names:

- **`ρ_ij = exp(−β|i−j|)` is non-uniform yet TRANSLATION-INVARIANT.** The spec's guard said "use a
  multi-forward swap with non-trivial, non-uniform `ρ_ij`", and this fixture satisfies that reading
  while still hiding the span offset: `ρ[first+m][first+l]` and `ρ[m][l]` are the same number when
  `ρ` depends only on `|i−j|`. A mutation **dropping the span offset entirely survived all 603
  tests.** Jäckel eq 12.12 writes the decay in the actual times `t_i`, not in index distance — so the
  faithful reading is also the non-degenerate one, and my index-distance simplification is what
  created the hole.
- **Flat accruals hid the offset a second time.** All-`0.5` accruals make `accruals[:count]` and
  `accruals[first:first+count]` the same slice, so a mutation dropping that offset also survived.
  Sixth appearance of the degenerate-fixture trap, and the first where the degeneracy is *uniformity
  across a slice* rather than a specific value like `τ = 1`.

Both fixed by non-uniform accruals `(0.50, 0.25, 0.50, 0.75, 0.50, 0.25)` and a correlation decaying
in the cumulative times. Both mutants now die. The non-degeneracy is asserted as a **premise**, and
stated as the property the mutant needed — that the `ρ` block a span reads differs from the block at
offset 0 — rather than as the weaker "the entries are not all equal", which the old fixture passed.

**The third survivor is a proved equivalent mutant, not a hole.** Transposing `ρ`'s indices
(`ρ[i][j] → ρ[j][i]`) changes nothing, because `LmmCorrelation.__post_init__` enforces symmetry
exactly — and that invariant is itself oracled by the symmetry refusal. Proved per §4 by bit-identity
rather than argued: the vol is bit-for-bit equal across seven spans under the mutation.

## 10. Oracle 3 claimed a power it did not have, and its own power assertion caught it

The first version asserted agreement with Black(Rebonato) inside one flat **3%** band, called that
"the freezing slack", and asserted that a 10% vol error would fall outside. **It failed**, and both
halves of the claim were wrong:

- **The band was accommodating the wrong error.** The freezing error, measured at 200,000 paths where
  sampling error is negligible, is **0.15%–0.96%** depending on step count, settling near 0.5%. The
  2.1%–2.5% observed at the committed path count is mostly **Monte Carlo** error. One flat 3% standing
  in for two different errors would have let a real 2% freezing defect pass as sampling noise. The band
  is now `4·SE + 1.5%`, with the SE **measured** from the same paths the engine consumes.
- **The band had no power at all.** At 6,000 paths it resolves only a vol error of ~20%: a 10% error
  moved the price 294 against a band of 344. Measured resolution — ~20% at 6,000 paths, **10% at
  12,000**, 5% at 25,000. So the oracle is split: three spans at the committed count as a *breadth*
  check against gross structural error, with its coarse resolution stated in the docstring, and one
  dedicated test at 12,000 paths where a 10% error provably falls outside. Paying 25,000 everywhere
  would buy a finer resolution than the `slow` gate's ceiling allows.

Had the power assertion been omitted, three green breadth cases would have read as a 10%-accurate
cross-check. They are not, and now they say so.

## 11. Oracle 5's specced instrument does not carry over — replaced by something stronger

§4's oracle 5 was *"two LMM configs differing only in ρ, one seed — the swaption-price difference is a
near-deterministic function of Δρ"*, generalising T16's two-numeraire identity. **It does not
generalise.** T16's cancellation worked because only the *drift* differed between the two runs, so the
diffusion cancelled to the last bit. Here `ρ` enters the **factorisation** of `C = σσρ·dt`, so two
models differing in `ρ` produce genuinely different paths and nothing cancels exactly.

Worse, the price-level sensitivity cannot be tightly bounded: differencing amplifies relative error.
Each price carries up to the freezing slack, the two need not err in the same direction, and the move
is only ~5% of the price — so the finance permits the move to be wrong by over 100%. Measured
disagreement was 13.9% against a guessed 10% band, and no tighter number would have been derived
rather than tuned.

**What replaced it is stronger than the price-level claim.** `ρ` enters the engine in exactly one
place, and that place is directly observable: the realised covariance of the simulated log-increments
must reproduce `σ_i σ_j ρ_ij dt` for **every pair**. That pins all 36 entries individually, where a
swaption price pins one correlation-weighted scalar. Worst pairwise error 3.2% at 20,000 paths and
1.66% at 80,000 — halving as `√N` quadruples, so the `−½` rate is asserted rather than assumed, and a
*bias* in the factorisation would show as a failure to improve. The price-level comparison is kept as
a loose companion with its band **derived** (±50%) and its looseness stated.

## 12. The exact anchor, and what it is worth

A one-period swap's rate is exactly its forward and its weight exactly 1, so `v = σ_i` **exactly**
(1e-15 at every index) and a one-period swaption *is* a caplet. This engine and T16's caplet engine,
on one seed, agree to **1e-12** after the deterministic `τ·B(0,T_{i+1})·notional` scaling — a
cross-product oracle with different code on each side and no Monte Carlo error in the comparison. That
is the strongest check in the tranche, and it is what the whole span-and-measure construction rests
on: it would fail under any error in the annuity, the swap rate, the numéraire bond or the discount
index, and in mutation it did.

## 13. Findings for Cowork (§10 — raised, not decided)

1. **The co-equal-sources ruling needs amending, because its premise does not hold.** §1 promoted
   McInerney to a cited `source:` on the basis of cross-source transcription of one formula. The
   promotion is still right — both books are genuinely load-bearing here — but the *independence
   argument* for it is not available (§8). Rule whether `swap_rate_vol.py`'s header should cite Jäckel
   §12.4 as a **companion form** (how I wrote it) or as a co-equal rendering (what the spec says).
2. **Candidate §4 rule, from §9:** *a non-uniform fixture can still be degenerate.* A parameter that
   varies but is **invariant under the transformation the code performs** — a correlation depending
   only on `|i−j|` when the code offsets indices, a constant-across-a-slice accrual when the code
   slices — hides exactly the error that transformation can make. The test is not "do the values
   differ" but "does the block the code reads differ from the block a mutation would read", and the
   premise should be asserted in that form.
3. **Candidate §4 rule, from §10:** *a tolerance that lumps two error sources together is a tolerance
   that cannot tell them apart.* Band an approximation check as `k·SE + the approximation's own
   measured error`, with the SE measured from the same paths, never one flat percentage; and **assert
   the band's resolution**, because a band wide enough to absorb Monte Carlo error is wide enough to
   absorb a real defect. State what a green is evidence about.
4. **Oracle 5 was respecified during the build (§11).** The specced instrument rests on a cancellation
   that does not exist when `ρ` enters the diffusion. I built the realised-covariance pin instead and
   kept the price-level check as a loose companion. Rule whether that substitution stands.
5. **The engine returns a `PricingResult` with no standard error**, so the oracle recomputes the
   payoff from the same paths to obtain one — one extra simulation per banded case. T16's
   `mc_caplet_price` returns mean *and* SE; this engine cannot, because §3's contract is
   `PricingResult | PricingFailure`. Worth ruling whether an MC engine should carry its SE somewhere
   the charter permits, since every banded oracle over an MC pricer will otherwise pay this twice.
