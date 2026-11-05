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
