# T75 — options on realized variance under Heston (and the cross-method anchor T74 lacked)

**Cowork spec, 2026-11-22.** Second tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`). Two
pieces: the **Heston fair variance** via the MGF route (reusing T48's `integrated_variance_transform`), and
**options on discrete realized variance** by the normal/moment approximation (the T62 Turnbull–Wakeman
pattern). The first piece carries a bonus that is the point of putting it here: it is a **second, independent
METHOD** for the fair variance T74 built by static replication — the cross-check the arc has not had (T74's
anchors were all transcription, reduction and requirement, no independent route). *(Sourcing confirmed from
the TOC, 0397: §4.2.1 the Heston fair variance; ch 5 / §5.2 the options and the normal approximation; §4.3 is
the MGF route — this tranche's cross-check anchor. Not a repeat of T74's §1.4-vs-ch-4.3 mix-up.)*

## Target
Price the **fair variance under Heston** and a **call/put on *continuous* realized variance** —
`price(product, HestonModel) → PricingResult` for each.

> **Ruled continuous, not discrete (Cowork 0401, post-merge).** The original target said "discrete",
> but that contradicted this spec's own out-of-scope (which defers the discrete-monitoring correction
> to T78) and §5.2's "the continuous counterpart **plus** an adjustment term due to discrete
> sampling". Code priced the coherent half — the continuous case — and said so in the module and the
> CHANGELOG. **T75 is the continuous case; the discrete-monitoring adjustment is T78's**, which that
> ruling turns into a *pricing* tranche (not only a convergence study). The arc plan's T78 entry
> carries it.

## Pieces, layer, reuse
- **The Heston fair variance** (new engine, L4) — the expected realized variance under Heston, off
  **`integrated_variance_transform`** (T48, the Laplace transform of integrated variance) and/or the closed
  form of §4.2.1. Reuses T48 whole; the new code is at most the §4.2.1 arrangement.
- **Options on discrete realized variance** (new product + engine, L2/L4) — a call/put on realized variance
  `(RV − K)⁺`. Priced by the **normal approximation** (§5.2): match the first two moments of discrete realized
  variance (exact, off the Heston MGF / transform) to a normal, price the option under that normal
  (Bachelier — **reuse `normal_positive_part`** / the T71 Bachelier). The approximation is the lognormal/normal
  assumption; the moments are exact. The MC companion (reuse `heston_mc`) measures the error (the T62 pattern).
- **The MGF fair variance** (for the cross-check) — §4.2.1 gives the fair variance under Heston directly; this
  is the independent-method value that T74's static replication must agree with.

## The mathematics (re-transcribe from the source — NOT this spec, §5)
Read **Kwok–Zheng §4.2.1 (Heston fair variance), §5 / §5.2 (options on realized variance, the normal
approximation)**, and **§4.3.1/§4.3.2 (the MGF variance/gamma/corridor swaps)** for the cross-check, in `-raw`
and `-layout` and reconcile (§4). Transcribe the first two **moments** of discrete realized variance as exact
quantities (the only approximation is the normal match, §5.2); the moments carry the Heston MGF, so pin them
against `integrated_variance_transform`. The spec names the obligation, not the form (§4).

## Oracles (RED-first, §4)
1. **The cross-METHOD anchor — the Heston MGF fair variance == T74's static replication, the independent route
   T74 lacked.** Price the Heston model's European options → implied vols → feed **T74's `fair_variance`
   strip** → compare to the **MGF fair variance** (piece 1). Two genuinely different derivations of one number
   (replication of a Heston-generated skew vs the MGF in closed form), different code each side — the §4
   cross-product oracle at the level of *method*, which no within-T74 check could be. **Measure the grid per
   weight** (the T74 lens — the replication's truncation regime is the integrand's, not inherited). **Extend
   to gamma/corridor** (§4.3.1/§4.3.2 MGF vs T74's gamma/corridor replication) **if** those MGF forms
   transcribe cheaply — a bonus independent anchor for all three of T74's products; defer to `to_go_deeper` if
   they do not.
2. **The two moments of discrete realized variance, transcribed pointwise** (exact, off the transform);
   cross-check `M₁` (the first moment) against the **continuous** fair variance in the `n→∞` limit (the
   discrete→continuous reduction — a T31/T78 preview, a rate check if cheap).
3. **The option on realized variance — the MC companion MEASURES the normal-approximation error** (§4): band
   `k·SE + measured approximation error`, the approximation error measured at a huge path count, resolution
   stated, swept over the regime where the normal approximation degrades (short tenor / high vol-of-vol — the
   honest negative, the T62 deliverable). Put–call parity on realized-variance options.
4. **Reductions:** the Heston fair variance at **`σ_v → 0`** (deterministic variance) ⇒ the integrated
   deterministic variance (exact, with a genuine `σ_v > 0` companion, T39); the realized-variance option at
   **`σ_v → 0`** ⇒ a deterministic payoff.
5. **Requirement oracles:** the fair variance `≥ 0`; the option monotone in the variance strike; the moments
   ordered (`M₂ ≥ M₁²`, variance non-negative).

## Degeneracy premises (§4 / §9)
- The cross-method fixture uses a **non-flat Heston skew** (under flat vol the replication/MGF agreement is
  trivial — the T74 flat-skew degeneracy); and the grid is **wide enough for the integrand** (the T74
  per-weight truncation lens — measure, do not inherit T74's).
- The `σ_v → 0` reductions' companions are **non-degenerate in `σ_v`** (T39).
- The normal-approximation sweep reaches a regime where the error is **resolvable above the MC SE** (the
  band-resolution rule), else "approx agrees with MC" asserts only that both are noisy.

## Provenance / gates
- Header `source: KWOK-ZHENG §4.2.1, §5.2` (+ §4.3.1/2 if the cross-check is built); `oracle:`; `crosscheck:`
  (QuantLib has a Heston realized-variance engine — a real target).
- Version → **v0.77.0**; one floor bump; one `CHANGELOG` entry. Re-time the Heston MC on **both** CI runners
  (§9).

## Out of scope (carried / deferred → T76+)
- **VIX derivatives** — T76 (VIX² is the forward variance-swap rate, a different product off the same
  transform).
- **The partially-exact-and-bounded approximation** (§5.3) — a sharper options-on-variance approximation;
  `to_go_deeper` if the normal approximation's measured error is too wide (the arc plan's recorded refinement).
- **The gamma/corridor MGF cross-check** — folded into oracle 1 only if cheap; else a `to_go_deeper` note
  (the independent-method anchor for T74's gamma/corridor, available but not load-bearing).
