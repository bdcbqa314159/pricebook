# T62 — arithmetic Asian options, Turnbull–Wakeman

**Cowork spec, 2026-11-19.** Third tranche of the commodity arc (project eight, `CLARK-2014`). The
**arithmetic** average-rate option — the one actually traded — which, unlike T61's geometric average, has
**no exact closed form** (a sum of lognormals is not lognormal). Turnbull–Wakeman matches the arithmetic
average's first two moments to a lognormal and prices with Black-76. The headline discipline of this tranche
is that the result is an **approximation**, and the method must prove it knows that: the MC companion
**measures** the approximation error rather than hiding it in a loose band.

## Target
`price(CommodityArithmeticAsian, CommodityBlackModel) → PricingResult` over a fixing schedule, by the
Turnbull–Wakeman moment-matched approximation (ch 2.7.4 discrete; 2.7.2 continuous as an optional limit).

## Pieces, layer, reuse
- **The product** (new, L2) — `CommodityArithmeticAsian`: identical fields to T61's
  `CommodityGeometricAsian` (asset, strike, direction, notional, fixing schedule). Shared **shape** and the
  same averaging idea → this is the rule-of-two moment for the Asian family. **Raise, do not pre-abstract
  (§6):** if the two Asian products are field-identical and the averaging *schedule* handling is shared,
  propose the shared vocabulary (a common `AsianSchedule`/averaging value object, or one product carrying an
  `average_kind`); but geometric and arithmetic are **different payoffs priced by different methods** (exact
  vs approximate), so the *engines* stay separate. Code rules the exact factoring at build and raises it as
  an `Ask:` — it is the kind of structural call the lightest-tranche rule reserves for the review.
- **The engine** (new, L4) — the **two exact moments** `M₁ = E[A]`, `M₂ = E[A²]` of the arithmetic average
  `A = (1/n)Σ F(tᵢ)`, then the lognormal match (`σ²_TW = ln(M₂/M₁²)`, forward `M₁`) fed to **`black_price`**.
  Reuses T60's forward and `black_price`; the only new code is the two moment formulae.

## The mathematics (re-transcribe from the source — NOT from this spec, §5)
Read **Clark ch 2.7.4** in `-raw` and `-layout` and reconcile (§4). The two moments are **exact** — the
*only* approximation is assuming `A` is lognormal, so `M₁` and `M₂` must be transcribed pointwise as exact
quantities (do not fold the lognormal assumption into them). `M₂` carries a **double sum**
`Σᵢ Σⱼ F(tᵢ)F(tⱼ)·exp(σ²·min(tᵢ,tⱼ))` — the load-bearing term, the same `min(tᵢ,tⱼ)` structure T61 pinned,
and the term a mutation will most plausibly corrupt. Mark every formula "re-transcribe from eq X"; the spec
is a pointer, not the authority.

## Oracles (RED-first, §4)
1. **Pointwise transcription of `M₁` and `M₂`** — the test forms its own `F(tᵢ)` forwards and the
   `exp(σ²·min(tᵢ,tⱼ))` double sum independently; these are exact, so they get an exact (tight) oracle, not
   a band. The F-6.1 lesson again: `M₂`'s double sum must be transcribed in its own right, not only inside
   the final price.
2. **The MC companion MEASURES the approximation error — the heart of the tranche (§4).** Simulate the
   arithmetic average, price by discounted payoff. Band the TW-vs-MC check as **`k·SE + measured TW error`**,
   where the TW error is measured **where sampling noise is negligible** (a huge path count), **never** a
   flat percentage standing in for both (the T17 rule). **State the band's resolution** per load-bearing
   term and add a higher-path power test. Turnbull–Wakeman is a good approximation in low-vol/short-tenor
   and degrades with `σ√T`: measure the error across a `σ√T` sweep and **record where it exceeds, say, 0.5%**
   — the honest-negative is the material (and the journal beat).
3. **Ordering requirement — geometric ≤ arithmetic (AM–GM), the free cross-product oracle (§4).** For the
   **same** schedule and parameters, T61's exact geometric price ≤ this arithmetic price, strictly for
   `n ≥ 2` with distinct fixings. Different code each side (T61's closed form vs T62's moment match), so it
   catches a shared-reading error neither within-product check can — the T3/T5 cross-product pattern. Assert
   strictly, and assert **equality at `n = 1`** (both reduce to the T60 European).
4. **Reduction: `n = 1` ⇒ the T60 European.** `M₁ = F(t₁)`, `M₂ = F(t₁)²e^{σ²t₁}`, so `σ²_TW = σ²t₁` and the
   price is the T60 call exactly — **in ℝ**; expect the same `log`/`exp` round-trip ULP gap T61 hit, so
   **band it (equal in ℝ, banded in float), do NOT assert `==`** (the T61/T12(a) lesson, already paid once —
   this spec says so up front so it is not re-discovered). Companion non-degenerate in fixing count/spacing
   (T39).
5. **Put–call parity** on the TW forward `M₁`: `C − P = df·(M₁ − K)`, swept off the money.

## Degeneracy premises (§4 / §9)
- Distinct, non-uniform fixing times and non-trivial `T` (the `min(tᵢ,tⱼ)` double sum — T17); prefer a
  mutation to a reading (drop a cross term, `min→max`, drop the `1/n²` on `M₂`, drop the `e^{σ²min}`).
- The geometric≤arithmetic companion must use a schedule where the gap is **materially non-zero** (spread
  fixings, non-trivial `σ√T`) — a near-coincident schedule makes AM≈GM and the ordering oracle toothless.
- The TW-error measurement must reach a `σ√T` where the error is **resolvable above the MC SE** (the
  band-resolution rule) — otherwise "TW agrees with MC" asserts only that both are noisy.

## Provenance / gates
- `source: CLARK-2014 §2.7.4`; `crosscheck:` (QuantLib has a Turnbull–Wakeman / discrete-arithmetic-Asian
  analytic engine — a real target); `oracle:`.
- Version → **v0.64.0**; one floor bump; one `CHANGELOG` entry (lead with the measured TW-error curve — the
  honest negative is the point).

## Out of scope (carried / deferred)
- Average-**strike** arithmetic Asians; control-variate TW (the geometric Asian is the natural control — a
  `to_go_deeper` note if not pulled, since it would tighten the MC companion but is not needed to price).
- Partly-fixed (seasoned) arithmetic Asians — the same commodity-fixing-series prerequisite as T61
  (`to_go_deeper`, already recorded).
- Higher-moment (e.g. Levy, Curran) approximations — `to_go_deeper`; TW is the spine approximation.
