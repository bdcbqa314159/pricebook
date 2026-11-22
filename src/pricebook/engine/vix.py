"""VIX derivatives under Heston (L4). KWOK-ZHENG §1.5.2 eq (1.31), §3.1.2 eqs (3.6)/(3.7a).

**There is almost no new mathematics here, and that is the finding.** Three things the arc already
has compose into the whole tranche:

1. **VIX² IS the variance-swap rate** — eq (1.31), §1.5.2:
   `VIX² = E₀[(1/τ)∫₀^τ σ²dt] = (2e^{rτ}/τ)[∫₀^{F₀} p/K² dK + ∫_{F₀}^∞ c/K² dK]`. The right-hand side
   is tranche 31's strip *verbatim*, and under Heston the left-hand side is tranche 75's
   `heston_fair_variance`. So `vix_squared` **is** that function, renamed for what it means here, and
   the "model-free VIX² strip" is `fair_variance` — nothing to re-derive on either side.
   *(The T76 spec pinned §3.1/§3.1.1 as the load-bearing identity. §3.1.1 opens by saying the
   pure-diffusion identity "we have shown in Sec. 1.5.2 … [see (1.31)]" — §3.1.1 is the **jumps**
   correction, Carr–Wu's account of how VIX² and the variance-swap rate *differ* under jumps, which
   Heston does not have. Sourced from §1.5.2 accordingly.)*

2. **`VIX²_T` is AFFINE in the terminal variance**, with coefficients that fall out of (1) rather
   than being transcribed: `VIX²_T = E_T[(1/τ)∫_T^{T+τ}V ds]` is tranche 75's fair variance started at
   `T` from `V_T`, i.e. `θ + (V_T−θ)(1−e^{−κτ})/(κτ)`. So `a = (1−e^{−κτ})/(κτ)` and `b = θ(1−a)`,
   and `VIX_T = √(a·V_T + b)` is a deterministic function of one random variable.

3. **The law of `V_T` is non-central chi-squared**, so tranche 72's primitive prices these:
   `V_T = c·χ'²_d(λ)`, `c = ε²(1−e^{−κT})/(4κ)`, `d = 4κθ/ε²`, `λ = 4κe^{−κT}V₀/(ε²(1−e^{−κT}))`.

**Integrated BY PARTS, so only the SURVIVAL function is needed.** With `u(x) = √(ax+b) − K`, which
vanishes at the exercise boundary `x_K = max((K²−b)/a, 0)`, and `S(x) = P(V_T > x)`:

    E[(VIX_T − K)⁺] = ∫_{x_K}^∞ (a / 2√(ax+b)) · S(x) dx
    E[VIX_T]        = √b + ∫_0^∞ (a / 2√(ax+b)) · S(x) dx          (since `S(0) = 1`)

Three things this buys. No density is needed, so nothing new lands at L0 (§6). Nothing is
differentiated, so there is no finite-difference noise floor (the T11 trap). And the integrand decays
with the chi-squared **tail**, which is exactly where `noncentral_chisquare_sf` is accurate and where
`1 − cdf` collapses to zero — the defect F-8.6 was about, here turned into the reason this
construction is the right one.

Provenance:
  stock:  none — the migration tree carried no VIX derivative
  source: KWOK-ZHENG (docs/references.md) §1.5.2 eq (1.31) — VIX² as the model-free variance-swap
          rate, transcribed in `-raw` and `-layout` and reconciled; §3.1.2 eq (3.6) the futures
          price as an expectation and eq (3.7a) the discounted European call. The affine
          coefficients and the by-parts integrals are DERIVED from (1.31) and the CIR law rather
          than transcribed — there is no printed closed form for them under Heston in ch 3, and
          T19's precedent governs: derive and assert as a theorem
  oracle: tests/L4/test_vix.py — the VIX²/variance-swap-rate identity as a requirement oracle; the
          affine coefficients against `heston_fair_variance` at every starting variance; the future
          and the option against an independent 300k-path simulation of `V_T`, banded `4*SE + the
          measured Euler bias` with the resolution asserted; Jensen's convexity requirement
          (strict under vol-of-vol, vanishing without it); put-call parity; and the
          translation-invariance premise that the forward window is not the spot one
  crosscheck: none — no standard library prices VIX under Heston; the independent anchors are
          in-tree (tranche 31's strip, tranche 72's chi-squared, the simulation)
  slice:  tranche 76 — VIX derivatives
"""

from __future__ import annotations

import math

from pricebook.engine.realized_variance_option import heston_fair_variance
from pricebook.engine.registry import register
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.foundation.noncentral_chisquare import noncentral_chisquare_sf
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.time_measure import is_historical
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.vix import VixFuture, VixOption

#: Midpoint nodes for the by-parts integral, chosen from a **measured** convergence rather than a
#: round number. Against a 16000-node reference the relative error runs 2.65e-03 at 250 nodes,
#: 7.33e-04 at 500, 1.97e-04 at 1000, **5.22e-05 at 2000** and 1.34e-05 at 4000 for the future (and
#: an order and a half tighter for an out-of-the-money option). The oracle bands are
#: `4*SE + the measured Euler bias`, i.e. ~0.5-1%, so 2000 puts the quadrature two orders below what
#: any green here can resolve — and 4000 bought nothing but time on a file the `slow` gate watches.
_NODES = 2000

#: How far either side of `E[V_T]` the quadrature spans, in standard deviations of `V_T`.
#:
#: **Standard deviations, not multiples of the mean, and the first cut got that wrong.** Spanning
#: `60x` the *mean* looks generous and is not: as the noncentrality grows the distribution's spread
#: shrinks *relative* to its mean, so a fixed multiple of the mean leaves the integrand's whole
#: transition inside a few of 4000 nodes. Measured, that cost ~1e-4 relative at a noncentrality of
#: 1e4 and — the tell — **non-monotonically**, which is a quadrature losing resolution rather than a
#: limit being approached. Tracking the spread keeps the nodes on the mass at every noncentrality.
#:
#: 12 standard deviations above the mean is past anything a chi-squared result carries; below the
#: mass the survival is 1 and that stretch is integrated **analytically** instead (see
#: `_survival_integral`), which is what makes a narrow span safe rather than truncating.
_SPREAD_SPANS = 12.0


def vix_squared(params: HestonParams, window: float) -> float:
    """`VIX²(0; window)` — the variance-swap rate over `[0, window]`, eq (1.31).

    **It is `heston_fair_variance`, and that is the identity rather than a shortcut.** Eq (1.31)
    says VIX² *is* the expected annualised integrated variance; under Heston that is tranche 75's
    closed form. Named here because "the VIX level" and "the fair variance of a variance swap" are
    the same number wearing the two names the market uses, and a caller reaching for one should not
    have to know the other.
    """
    return heston_fair_variance(params, window)


def vix_affine_coefficients(params: HestonParams, window: float) -> tuple[float, float]:
    """`(a, b)` with `VIX²_T = a·V_T + b` — derived from eq (1.31) started at `T`.

    `a = (1−e^{−κτ})/(κτ)`, `b = θ(1−a)`. Returned as a pair because the two are one decomposition
    and every caller needs both; not a value object, since two numbers with three callers inside one
    module are not vocabulary (§6).
    """
    if window <= 0.0:
        raise ValueError(f"a VIX window must be positive, got {window}")
    decay = (1.0 - math.exp(-params.kappa * window)) / (params.kappa * window)
    return decay, params.theta * (1.0 - decay)


def _terminal_variance_law(params: HestonParams, years: float) -> tuple[float, float, float]:
    """`(scale, dof, noncentrality)` with `V_T = scale · χ'²_dof(noncentrality)`."""
    if years <= 0.0:
        raise ValueError(f"the terminal-variance law needs a positive horizon, got {years}")
    kappa, theta, sigma_v, v0 = params.kappa, params.theta, params.sigma_v, params.v0
    if sigma_v <= 0.0:
        raise ValueError("a deterministic variance process has no chi-squared law")
    spread = 1.0 - math.exp(-kappa * years)
    scale = sigma_v * sigma_v * spread / (4.0 * kappa)
    dof = 4.0 * kappa * theta / (sigma_v * sigma_v)
    noncentrality = 4.0 * kappa * math.exp(-kappa * years) * v0 / (sigma_v * sigma_v * spread)
    return scale, dof, noncentrality


def _survival_integral(params: HestonParams, years: float, window: float, lower: float) -> float:
    """`∫_lower^∞ (a / 2√(ax+b)) · P(V_T > x) dx`, by the midpoint rule.

    The one integrand both payoffs reduce to after integrating by parts — see the module docstring.
    """
    a, b = vix_affine_coefficients(params, window)
    scale, dof, noncentrality = _terminal_variance_law(params, years)
    mean = scale * (dof + noncentrality)
    spread = scale * math.sqrt(2.0 * (dof + 2.0 * noncentrality))
    start = max(lower, mean - _SPREAD_SPANS * spread)
    stop = mean + _SPREAD_SPANS * spread
    if stop <= lower:
        return 0.0

    # Below the mass the survival is 1 to every digit, so that stretch is exact:
    # `int (a / 2 sqrt(ax+b)) dx = sqrt(ax+b)`. Doing it analytically is what lets the quadrature
    # span only where the integrand actually varies, instead of spending its nodes on a constant.
    total = math.sqrt(a * start + b) - math.sqrt(a * lower + b) if start > lower else 0.0

    step = (stop - start) / _NODES
    for index in range(_NODES):
        x = start + (index + 0.5) * step
        total += step * (a / (2.0 * math.sqrt(a * x + b))) * noncentral_chisquare_sf(
            x / scale, dof, noncentrality
        )
    return total


#: Noncentrality past which the chi-squared mixture is abandoned for the deterministic limit.
#:
#: Tranche 72's affordability bound, and the same arithmetic: the Poisson mixture costs
#: `O(sqrt(noncentrality))` terms. What is different here is that the fallback is **right**, and
#: that was measured rather than assumed — T72's lesson, where "a large `z` means a negligible
#: variance" was false and returned an intrinsic for a caplet worth 0.00398.
#:
#: Here the noncentrality is `4*kappa*e^{-kappa T}*V0 / (sigma_v^2 (1-e^{-kappa T}))`, so it grows
#: **only** as the vol-of-vol vanishes or the horizon shortens — and `V_T`'s relative dispersion
#: tracks it as `stdev/mean ~ 1.5/sqrt(noncentrality)` in *both* directions, measured: 2.0e-2 at
#: 2.3e3, 2.0e-3 at 2.3e5, 2.0e-6 at 2.3e11 for a vanishing `sigma_v`, and 5.8e-2 at 1.2e3, 5.8e-3
#: at 1.2e5 for a vanishing horizon. So a noncentrality this large *is* a near-deterministic `V_T`,
#: and at the bound the relative dispersion is ~3e-4, which puts the convexity gap the limit drops
#: at ~1e-8 relative — measured in the oracle, not hoped for.
_MAX_NONCENTRALITY = 1.0e7


def _is_numerically_deterministic(params: HestonParams, years: float) -> bool:
    """Whether `V_T`'s law has collapsed to a point as far as this quadrature can tell."""
    if params.sigma_v <= 0.0:
        return True
    _, _, noncentrality = _terminal_variance_law(params, years)
    return noncentrality > _MAX_NONCENTRALITY


def _deterministic_vix(params: HestonParams, years: float, window: float) -> float:
    """`VIX_T` when the variance path is deterministic — no dispersion, so no integral."""
    a, b = vix_affine_coefficients(params, window)
    terminal = params.theta + (params.v0 - params.theta) * math.exp(-params.kappa * years)
    return math.sqrt(a * terminal + b)


def vix_future_value(params: HestonParams, years: float, window: float) -> float:
    """`E₀[VIX_T(window)]` — eq (3.6), undiscounted because a futures price is an expectation.

    `= √b + ∫₀^∞ (a/2√(ax+b))·S(x) dx`, the by-parts form. **Strictly below `√(VIX²)`** by Jensen,
    and the gap is the whole economic content of a VIX future; the oracle asserts it is strict under
    vol-of-vol and vanishes without it.
    """
    if _is_numerically_deterministic(params, years):
        return _deterministic_vix(params, years, window)
    _, b = vix_affine_coefficients(params, window)
    return math.sqrt(b) + _survival_integral(params, years, window, 0.0)


def vix_option_value(
    params: HestonParams, years: float, window: float, strike: float, kind: CapFloorKind
) -> float:
    """`E₀[(ω(VIX_T − K))⁺]`, undiscounted — the bracket of eq (3.7a).

    The call is the by-parts integral from the exercise boundary; the **put comes from parity**
    rather than a second integral, so the two cannot disagree about the quadrature (and parity then
    holds by construction, which is why the oracle checks it against the *future* rather than
    against a second integration).
    """
    if strike < 0.0:
        raise ValueError(f"a VIX strike cannot be negative, got {strike}")
    if _is_numerically_deterministic(params, years):
        return max(
            kind.omega * (vix_future_value(params, years, window) - strike), 0.0
        )
    a, b = vix_affine_coefficients(params, window)
    boundary = max((strike * strike - b) / a, 0.0)
    call = _survival_integral(params, years, window, boundary)
    if kind is CapFloorKind.CAP:
        # A call needs no future, and the first cut computed one anyway — a whole extra quadrature
        # per call, on a file the `slow` gate watches.
        return call
    return call - (vix_future_value(params, years, window) - strike)


@register(VixFuture, HestonModel)
def price_vix_future(
    product: VixFuture, model: HestonModel
) -> PricingResult | PricingFailure:
    """`N · E₀[VIX_T]` — **undiscounted**, because eq (3.6) is an expectation and not a PV."""
    market = model.market
    currency = product.notional.currency
    if is_historical(product.maturity, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the VIX future: {exc}")
    years = discount.time_measure.year_fraction(product.maturity)
    value = vix_future_value(model.params, years, product.window)
    return PricingResult(pv=Money(product.notional.amount * value, currency))


@register(VixOption, HestonModel)
def price_vix_option(
    product: VixOption, model: HestonModel
) -> PricingResult | PricingFailure:
    """`N · df(T) · E₀[(ω(VIX_T − K))⁺]` — eq (3.7a), discounted."""
    market = model.market
    currency = product.notional.currency
    if is_historical(product.maturity, market.valuation_date):
        return PricingResult(pv=Money(0.0, currency))
    try:
        discount = market.curves.discount(currency)
    except KeyError as exc:
        return PricingFailure(f"cannot price the VIX option: {exc}")
    years = discount.time_measure.year_fraction(product.maturity)
    value = vix_option_value(
        model.params, years, product.window, product.strike, product.kind
    )
    return PricingResult(
        pv=Money(product.notional.amount * discount.df(product.maturity) * value, currency)
    )
