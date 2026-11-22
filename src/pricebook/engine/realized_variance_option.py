"""Options on realized variance under Heston, by the normal approximation (L4). KWOK-ZHENG ch 5.

Three things, and the first one's main value is that it anchors the previous tranche.

**The Heston fair variance, and the cross-METHOD anchor it gives tranche 74.** `E[(1/T)∫V dt]` is
the fair strike of a continuous variance swap under Heston, in closed form. Tranche 74 produced the
same number by **static replication** — a `1/K²` strip of vanillas — and every oracle it had was a
transcription, a reduction or a requirement: there was no *independent route to the number itself*.
This is one. Price Heston's European options, imply their Black vols, run them through T74's
`fair_variance` strip, and the two must agree — a replication of a model-generated skew against a
moment of the variance process, different code and different mathematics on each side.

**The two moments, and why one is transcribed and the other derived (§4).** The first is printed:
KWOK-ZHENG §4.1 gives `E[∫V dt] = θT/N + (V₀−θ)(e^{−κt_{k−1}} − e^{−κt_k})/κ` per monitoring
interval, which at the full horizon is `θT + (V₀−θ)(1−e^{−κT})/κ`. The second is **derived**: the
source's printed second moment is per-interval and for the jump-extended model, and a `-layout`
reflow of it is not a thing to transcribe. T19's precedent governs — *derive it and assert it as a
theorem to machine precision, which is stronger than transcription because it is immune to a source
misprint*. From `Cov(V_s,V_t) = e^{−κ(t−s)}Var(V_s)` and the CIR variance
`Var(V_s) = (ε²/κ)[V₀(e^{−κs} − e^{−2κs}) + (θ/2)(1−e^{−κs})²]`, writing
`B = ε²θ/2κ`, `A = ε²V₀/κ`, the inner integral gives `Var(V_s) = B + (A−2B)e^{−κs} + (B−A)e^{−2κs}`
and

    Var[∫₀^T V dt] = 2[ (B/κ)·(T − (1−e^{−κT})/κ)
                      + (A−2B)·(1 − e^{−κT}(1+κT))/κ²
                      + ((B−A)/κ)·((1−e^{−κT})/κ − (1−e^{−2κT})/2κ) ].

Asserted **two independent ways** in the oracle: against a double quadrature of the covariance
integral, and against tranche 48's `integrated_variance_transform` — the CIR Laplace transform,
whose derivatives at zero *are* these moments and which shares no line with this file.

**What is NOT priced here, which is the spec's own reading.** The spec's target says "a call/put on
*discrete* realized variance", while its out-of-scope defers "the discrete-monitoring correction" to
T78 — and §5.2 (Drimus et al.) is explicitly *"the price of its continuous counterpart plus an
adjustment term due to discrete sampling"*. Both cannot hold: the discrete price **is** the
continuous one plus the correction T78 owns. This prices the **continuous** case, which is the
coherent half; T78 adds the adjustment.

Provenance:
  stock:  none — the migration tree carried no realized-variance option
  source: KWOK-ZHENG (docs/references.md) §4.1 (the first moment of integrated variance, printed
          per monitoring interval), ch 5 / §5.2 (the option and the normal approximation to the
          conditional distribution of realized variance, Drimus et al.); the second moment is
          DERIVED from the printed covariance structure rather than transcribed, for the reason in
          the body — the printed form is per-interval and jump-extended
  oracle: tests/L4/test_realized_variance_option.py — the first moment pointwise against the
          printed form; the derived variance against an independent double quadrature of the
          covariance integral AND against tranche 48's Laplace transform by finite differences,
          with the FD's own `O(h)` accuracy argued; the `sigma_v -> 0` reduction with its companion;
          put-call parity, monotonicity in the strike, and the two horizon limits
  crosscheck: quantlib has no realized-variance-option engine under Heston; `scipy` backs nothing
          here — the transform cross-check is in-tree (tranche 48) and is the stronger anchor
  slice:  tranche 75 — options on realized variance under Heston
"""

from __future__ import annotations

import math

from pricebook.engine.registry import register
from pricebook.foundation.distributions import normal_positive_part
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.realized_variance_option import RealizedVarianceOption


def integrated_variance_moments(params: HestonParams, years: float) -> tuple[float, float]:
    """`(E[∫₀^T V dt], Var[∫₀^T V dt])` for the Heston/CIR variance process.

    Returned as a pair because the normal approximation needs both and computing them together is
    where the shared `e^{−κT}` lives; they are not a value object because two numbers with one
    caller are not vocabulary (§6).

    **`sigma_v = 0` gives exactly zero variance**, not a limit approached: with no vol-of-vol the
    path is deterministic, and every term above carries a factor of `ε²`.
    """
    if years < 0.0:
        raise ValueError(f"integrated variance needs a non-negative horizon, got {years}")
    kappa, theta, sigma_v, v0 = params.kappa, params.theta, params.sigma_v, params.v0
    decay = math.exp(-kappa * years)

    mean = theta * years + (v0 - theta) * (1.0 - decay) / kappa

    scaled_v0 = sigma_v * sigma_v * v0 / kappa
    scaled_theta = sigma_v * sigma_v * theta / (2.0 * kappa)
    level = years - (1.0 - decay) / kappa
    ramp = (1.0 - decay * (1.0 + kappa * years)) / (kappa * kappa)
    tail = (1.0 - decay) / kappa - (1.0 - decay * decay) / (2.0 * kappa)
    variance = 2.0 * (
        (scaled_theta / kappa) * level
        + (scaled_v0 - 2.0 * scaled_theta) * ramp
        + ((scaled_theta - scaled_v0) / kappa) * tail
    )
    return mean, variance


def heston_fair_variance(params: HestonParams, years: float) -> float:
    """`E[(1/T)∫₀^T V dt]` — the fair strike of a continuous variance swap under Heston.

    **Annualised**, so it is directly comparable to tranche 74's `fair_variance` (which carries its
    own `2/T`) and to a squared volatility: this is the cross-method anchor's left-hand side.
    """
    if years <= 0.0:
        raise ValueError(f"the fair variance needs a positive horizon, got {years}")
    mean, _ = integrated_variance_moments(params, years)
    return mean / years


def realized_variance_option_value(
    params: HestonParams, years: float, strike: float, kind: CapFloorKind
) -> float:
    """`E[(ω(RV − K))⁺]` under the normal approximation, undiscounted and per unit notional.

    §5.2's scheme: match the **exact** first two moments of annualised realized variance to a
    normal, then price the payoff under that normal. The normal match is the only approximation —
    the moments are closed form — which is what makes the MC companion's measured error an error of
    the *distributional* assumption rather than of the inputs (the T62 pattern).

    The positive part is `normal_positive_part` (L0, tranche 71): a Bachelier call and a normal
    partial expectation are one closed form, so there is nothing to re-derive here.
    """
    if years <= 0.0:
        raise ValueError(f"a realized-variance option needs a positive horizon, got {years}")
    mean, variance = integrated_variance_moments(params, years)
    return normal_positive_part(
        kind.omega * (mean / years - strike), math.sqrt(variance) / years
    )


@register(RealizedVarianceOption, HestonModel)
def price_realized_variance_option(
    product: RealizedVarianceOption, model: HestonModel
) -> PricingResult | PricingFailure:
    """`N · df(T) · E[(ω(RV − K))⁺]` under the normal approximation."""
    market = model.market
    currency = product.variance_notional.currency

    # Invariant 6: a matured option is the shell's business, PV 0 before any market lookup.
    if is_historical(product.expiry, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))

    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the realized-variance option: {exc}")

    years = discount.time_measure.year_fraction(product.expiry)
    value = realized_variance_option_value(
        model.params, years, product.variance_strike, product.kind
    )
    return PricingResult(
        pv=Money(product.variance_notional.amount * discount.df(product.expiry) * value, currency)
    )
