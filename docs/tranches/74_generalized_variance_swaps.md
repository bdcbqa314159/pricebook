# T74 — generalized variance swaps: the gamma swap and the corridor variance swap

**Cowork spec, 2026-11-22.** First tranche of the vol-derivatives arc (project ten, `KWOK-ZHENG`;
`docs/vol_derivatives_arc.md`). Deliberately the **lightest** tranche: both products are the T31 variance
swap's static replication re-weighted, so the engine **reuses `variance_replication`** and the oracle spine
is the reduction back to the vanilla variance swap. `KWOK-ZHENG` flips `in-hand → worked-from` here.

## Target
Price a **gamma swap** and a **corridor variance swap** by static replication —
`price(product, model) → PricingResult` for each — against the same implied-vol skew the vanilla variance
swap replicates against.

## The two products (re-transcribe from the source — NOT this spec, §5)
Read **Kwok–Zheng ch 4.3** (the generalized variance swaps) in `-raw` and `-layout` and reconcile (§4).
- **Gamma swap** (weighted variance swap): accrues `(S_t/S_0)·dσ²` — realized variance weighted by the spot
  level. Replicated by a **`1/K`-weighted** strip of OTM vanillas (vs `1/K²` for the vanilla variance swap).
- **Corridor variance swap**: accrues variance only while `S_t ∈ [L,U]`. Replicated by the **`1/K²` strip
  restricted to strikes in `[L,U]`**.
The spec states the weighting in words; the exact strip and any `2/T` / log-contract constants are transcribed
from ch 4.3, not from here.

## Pieces, layer, reuse
- **The products** (new, L2) — a `GammaSwap` and a `CorridorVarianceSwap`, each a frozen dataclass:
  underlying, strike (in variance), notional, expiry, and for the corridor the bounds `(L, U)`. Siblings of
  `VarianceSwap` (T31), **not** subclasses — a gamma/corridor payoff is a different accrual (the §6
  real-leg-vs-YoY-leg line). ≤5 fields (§7); the corridor bounds bundle into a value object if over.
- **The engine** (new, L4) — **reuses `engine/variance_replication.py`**: the generalized fair variance is
  the same `black_price`-strip quadrature with the weight changed (`1/K` for gamma) or the range restricted
  (`[L,U]` for corridor). §6 rule-of-two may fire on the strip walk (vanilla + gamma + corridor share it,
  differing only in the weight/range) — if so, parameterise `fair_variance` by a **weight function** and a
  **strike range**, with the vanilla `1/K²`/full-range as the default (raise at review with the coupling, as
  the C16 extraction did).
- **The MC companion** reuses the GBM/Heston path simulation (`heston_mc` or a lognormal sim) — the realized
  gamma/corridor variance from simulated paths against the replication value.

## Oracles (RED-first, §4)
1. **Pointwise transcription of the two fair-variance strips** — the test forms its own `1/K` (gamma) and
   restricted-`1/K²` (corridor) quadrature, independently. The weight and the range are the load-bearing new
   terms; transcribe each pointwise (the F-6.1 rule).
2. **Reduction: the corridor variance swap at `[0,∞) ⇒ the vanilla variance swap`**, exactly — the restricted
   strip becomes the full strip, so the corridor fair variance equals `fair_variance` (T31) to the last bit.
   **Companion required (§4/T39):** a **genuine finite corridor** `[L,U]` checked against the pointwise
   transcription — the full-range reduction is blind to the restriction, which is the whole product.
3. **Requirement oracles — the structural orderings the finance forces.** `corridor fair variance ≤ vanilla`
   (restricting the range cannot increase accrued variance), strict for a genuine `[L,U]`; and the
   gamma/vanilla relation ch 4.3 states (transcribe it as a requirement, e.g. the gamma swap equals the
   vanilla under zero skew). Assert both, swept where they are non-degenerate.
4. **Cross-check — under a FLAT vol (no skew) the gamma swap = the vanilla variance swap** (the `1/K` vs
   `1/K²` difference vanishes when the skew is flat, because the weighting only bites on the skew). A
   consistency guard that is exact under the degenerate skew; the non-flat-skew case is where they differ and
   is the companion.
5. **The MC companion** prices the realized gamma/corridor variance from simulated paths, banded per §4
   (`k·SE + measured discretisation error`, resolution stated) — the replication value inside the band.

## Degeneracy premises (§4 / §9)
- The corridor reduction's companion is **non-degenerate in the corridor** (a genuine finite `[L,U]`, T39) —
  state it; `[0,∞)` is the reduction in disguise.
- **A non-flat skew** in the main gamma-swap fixture — under flat vol the gamma swap equals the vanilla, so a
  mutation swapping `1/K` for `1/K²` survives (the τ=1 shape: the skew is the parameter that makes them
  differ). Assert the premise; prefer a mutation (swap the weight, drop the range restriction, swap `L`/`U`).
- The corridor fixture's `[L,U]` must **exclude real mass** (a corridor so wide it captures the whole
  distribution is `[0,∞)` again).

## Provenance / gates
- Header `source: KWOK-ZHENG §4.3`; `oracle:`; `crosscheck:` (QuantLib has corridor/gamma variance-swap
  engines — a real target, or `crosscheck: none — <reason>`).
- `references.md`: flip `KWOK-ZHENG` to `worked-from` (it gains a `src/` citer here); `verify.py references`
  then holds.
- Version → **v0.76.0**; one floor bump; one `CHANGELOG` entry.

## Out of scope (carried / deferred → T75+)
- **Conditional variance swaps** (another generalized type) — `to_go_deeper` unless ch 4.3 pulls it naturally
  alongside the corridor.
- **Discrete-monitoring correction** — T78 (the discrete-vs-continuous convergence); this tranche prices the
  continuous replication, as T31 does.
- **Options on these generalized swaps** — T75's territory (options on realized variance).
