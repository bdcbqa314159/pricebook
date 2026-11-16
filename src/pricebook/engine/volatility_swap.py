"""The volatility swap under Heston (L4). `GATHERAL-2006` ch 11, eq 11.6 + the CIR transform.

`K_vol = E[√V_T]` with `V_T = (1/T)∫₀^T v_t dt` the **annualised** realized variance. There is no
static replication for `√`, so this is model-dependent by nature — which is the whole content of
ch 11 — and the quantity worth having beside it is the **convexity adjustment**
`√(K_var) − K_vol ≥ 0`.

**The representation.** Eq 11.6 trades the square root for a Laplace integral,

    E[√W_T] = (1/2√π) ∫₀^∞ (1 − E[e^{−ψW_T}]) · ψ^{−3/2} dψ,     W_T = ∫₀^T v_t dt

which is the deterministic identity `√x = (1/2√π)∫₀^∞(1−e^{−sx})s^{−3/2}ds` under the expectation
(Fubini). Note the **two time normalisations**: 11.6 is in the *total* variance `W_T`, while the
swap pays the *annualised* `V_T = W_T/T`, so `K_vol = E[√W_T]/√T`. At `T = 1` the two coincide,
which is why the oracles run at two maturities and neither is a year.

**The Heston transform, and the overbar neither rendering shows.** `E[e^{−ψW_T}]` is formally the
CIR bond price, `A·e^{−ψv₀B}`. Both `-raw` and `-layout` print `A`'s exponent as `2λv/η²` and the
mean as `(1−e^{−λT})/λ·(v − v) + vT` — the long-run `v̄` and the initial `v₀` are *indistinguishable*
in the extraction, so §4's dual-rendering rule is powerless here exactly as it was for T46's
radical. What settles it is a **requirement oracle against code already held**: the transform's
slope at zero must be `E[W_T]`, and `HestonParams.integrated_variance` (T33) computes
`θT + (v₀−θ)(1−e^{−κT})/κ`, which is precisely the mean the text prints. That pins `A`'s exponent to
the **long-run** variance and the exponential to the **initial** one — and it only discriminates on
a fixture with `v₀ ≠ θ`, which the source's own figure parameters (`v = v̄ = 0.04`, Figures 11.2 and
11.3) are not.

**`K_vol` does not depend on `ρ`.** `W_T` is a functional of the variance path alone and the
variance SDE carries no correlation, so neither the transform nor the convexity sees `ρ`. Asserted,
because it is the kind of thing an implementation can quietly get wrong.

Provenance:
  stock:  none — the archived predecessor had no volatility swap
  source: GATHERAL-2006 (docs/references.md) ch 11 — eq 11.6 (the `√` Laplace representation,
          p. 144), the CIR Laplace transform of total variance with `A`, `B` and
          `φ = √(λ²+2ψη²)` (p. 144, attributed to Cox-Ingersoll-Ross 1985), and the printed mean
          `E[W_T] = (1−e^{−λT})/λ (v₀−v̄) + v̄T` (p. 145)
  oracle: tests/L4/test_volatility_swap.py — the transform's Laplace requirements and its slope at
          zero against T33's `integrated_variance` (which pins the overbar), the Jensen bound over
          a parameter sweep, the exact `η→0` reduction with a non-degenerate `η≠0` companion, an
          independent Monte-Carlo of `√V_T` banded at 4·SE with the band's resolution stated, the
          log-strip cross-product for `K_var`, and the convexity's SECOND-order scaling in `η/λ`
  crosscheck: quantlib AnalyticHestonEngine / volatility-swap tooling — deferred to the optional
          QuantLib tier
  slice:  tranche 48 — volatility derivatives
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.volatility_swap import VolatilitySwap

#: Where the `ψ` integral is truncated and how finely it is sampled, after the `ψ = u²`
#: substitution — so these are limits in `u`, and `u = 200` reaches `ψ = 40,000`.
#:
#: Module constants rather than a config object, per §6: one setting covers the whole parameter
#: range the oracles exercise (measured below 1e-12 relative against an 800/80,000 reference), so a
#: knob would be a speculative extension point. It becomes a value object at the first caller that
#: genuinely needs to vary it — the `_PROTECTION_STEPS` precedent.
#:
#: The truncation is cheap because the `1` in `(1 − F)` is integrated **analytically** beyond the
#: limit: `∫_U^∞ ψ^{−3/2} dψ = 2/√U` exactly, and what remains out there is `∫_U^∞ F·ψ^{−3/2}`,
#: which is negligible because `F` decays. Without that correction the integral converges like
#: `U^{−1/2}` and no affordable truncation would do.
_TRANSFORM_UPPER = 200.0
_TRANSFORM_NODES = 4_000


def integrated_variance_transform(params: HestonParams, psi: float, years: float) -> float:
    """`E[e^{−ψ W_T}]` for `W_T = ∫₀^T v_t dt` — the CIR Laplace transform.

    Transcribed from p. 144 as `A·e^{−ψ v₀ B}` with `φ = √(λ² + 2ψη²)` and

        A = [ 2φ e^{(φ+λ)T/2} / ((φ+λ)(e^{φT} − 1) + 2φ) ]^{2λv̄/η²}
        B = 2(e^{φT} − 1) / ((φ+λ)(e^{φT} − 1) + 2φ)

    **but evaluated in a rearranged form, because the printed one overflows.** `e^{φT}` grows with
    `√ψ`, and the quadrature needs `ψ` up to 40,000: at `η = 1`, `T = 5` the printed expression
    raises before the integrand decays. Dividing numerator and denominator by `e^{φT}` gives

        A = [ 2φ e^{(λ−φ)T/2} / ((φ+λ)(1 − e^{−φT}) + 2φe^{−φT}) ]^{2λv̄/η²}
        B = 2(1 − e^{−φT}) / ((φ+λ)(1 − e^{−φT}) + 2φe^{−φT})

    which is the **same quantity in exact arithmetic** — one division of numerator and denominator
    by a common positive factor — and underflows harmlessly instead (`A → 0`). Agreement with the
    printed form was measured at **2.1e-15 relative**, a few ULP, across every `ψ` where the
    printed form computes at all; that is §4's pure-refactor standard (an exact-arithmetic identity
    plus a ULP band, since the reordering moves the last digits and bit-identity is unavailable).

    `η = 0` is the deterministic case and is handled directly: the exponent `2λv̄/η²` is undefined
    there, while the limit is simply `e^{−ψ E[W_T]}`.
    """
    if years <= 0.0:
        raise ValueError(f"the variance transform needs a positive horizon, got {years}")
    if psi == 0.0:
        return 1.0
    if psi < 0.0:
        raise ValueError(f"the Laplace transform is taken at non-negative psi, got {psi}")
    if params.sigma_v == 0.0:
        # Deterministic total variance: the transform is the exponential of its own mean.
        return math.exp(-psi * params.integrated_variance(years))

    vol_of_vol_squared = params.sigma_v * params.sigma_v
    phi = math.sqrt(params.kappa * params.kappa + 2.0 * psi * vol_of_vol_squared)
    decay = math.exp(-phi * years)
    denominator = (phi + params.kappa) * (1.0 - decay) + 2.0 * phi * decay
    exponent = 2.0 * params.kappa * params.theta / vol_of_vol_squared
    scale = 2.0 * phi * math.exp((params.kappa - phi) * years / 2.0) / denominator
    first = scale**exponent
    second = 2.0 * (1.0 - decay) / denominator
    return first * math.exp(-psi * params.v0 * second)


def fair_volatility(params: HestonParams, years: float) -> float:
    """`K_vol = E[√V_T] = E[√W_T]/√T` — eq 11.6 by Simpson in `u`, with `ψ = u²`.

    The substitution is what makes the integral tractable: `ψ^{−3/2} dψ` becomes `2u^{−2} du`, and
    the integrand `(1 − F(u²))/u²` has a **removable** singularity at `u = 0` whose value is
    `E[W_T]` (since `1 − F(ψ) ≈ ψ E[W_T]` for small `ψ`). Supplying that limit directly is why no
    epsilon offset is needed at the origin.
    """
    if years <= 0.0:
        raise ValueError(f"the fair volatility needs a positive horizon, got {years}")
    mean = params.integrated_variance(years)

    def integrand(u: float) -> float:
        if u <= 0.0:
            return mean
        return (1.0 - integrated_variance_transform(params, u * u, years)) / (u * u)

    step = _TRANSFORM_UPPER / _TRANSFORM_NODES
    total = integrand(0.0) + integrand(_TRANSFORM_UPPER)
    for index in range(1, _TRANSFORM_NODES):
        weight = 4.0 if index % 2 else 2.0
        total += weight * integrand(index * step)
    inner = total * step / 3.0 + 1.0 / _TRANSFORM_UPPER
    expected_root = (2.0 * inner) / (2.0 * math.sqrt(math.pi))
    return expected_root / math.sqrt(years)


def convexity_adjustment(params: HestonParams, years: float) -> float:
    """`√(K_var) − K_vol ≥ 0` — non-negative by Jensen, and zero exactly when `η = 0`.

    Named surface rather than a step inside the pricer because every oracle in this tranche reaches
    for *this* number rather than for a swap's PV (§4 — extract what the oracle must reach). It is
    also **second order** in the vol-of-vol: expanding the square root gives
    `√E[V] − E[√V] ≈ var[V]/(8 E[V]^{3/2})`, and `var[V_T] → v̄η²/(Tλ²)`, so the adjustment scales
    as `(η/λ)²`. Ch 11 p.146 reasons that it "should be of the order of the standard deviation of
    annualized volatility" and so scale as `η/λ`, predicting a factor 3.39 between the
    Bakshi-Cao-Chen and Heston-Nandi parameter sets; the measured factor is **11.49**, which is
    `3.3913²`. The text's heuristic drops a power. Its results are plots, so §4 forbids settling
    this against the figures — the derivation settles it, and the oracle asserts the square.
    """
    root_variance = math.sqrt(params.integrated_variance(years) / years)
    return root_variance - fair_volatility(params, years)


@register(VolatilitySwap, HestonModel)
def price_volatility_swap(
    product: VolatilitySwap, model: HestonModel
) -> PricingResult | PricingFailure:
    """`N * df(T) * (K_vol - K)` — the discounted volatility difference."""
    market = model.market
    currency = product.volatility_notional.currency

    # Invariant 6: a matured swap is the shell's business, PV 0 before any market lookup.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    if market.equities.get(product.asset) is None:
        return PricingFailure(f"no equity underlying for {product.asset.ticker} in the snapshot")
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the volatility swap: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    volatility = fair_volatility(model.params, years)
    payoff = product.volatility_notional.amount * (volatility - product.volatility_strike)
    return PricingResult(pv=Money(discount.df(product.expiry) * payoff, currency))
