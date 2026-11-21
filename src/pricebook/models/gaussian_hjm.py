"""One-factor Gaussian HJM (L3). `MCINERNEY-ZASTAWNIAK` §4.1-4.2.

The framework the short-rate models are special cases of. HJM models the **instantaneous forward
rate** directly, and the whole of chapter 4 turns on one equation.

**The source derives the drift condition rather than quoting it, which hands over two printed forms
of one statement** - the integrated condition and its derivative in `T`:

    df(t,T) = alpha(t,T) dt + sigma(t,T) dW(t)                                      (4.1)
    integral_t^T alpha(t,u) du = 1/2 ( integral_t^T sigma(t,u) du )^2    the integrated condition
    alpha(t,T) = sigma(t,T) integral_t^T sigma(t,u) du                   its T-derivative (Thm 4.1)

**It is a REQUIREMENT, not a source value.** No-arbitrage forces it - the source's argument is that
the discounted bond must be driftless, and the condition is what makes the `dt` term collapse to
`r(t)` - so it holds for **every** `sigma(t,T)`. That is §4's requirement-oracle case in its
cleanest form: a transcription error cannot hide behind a matching printed number, because the
identity is what the finance demands rather than what the book happened to print. The oracle asserts
it in both printed forms *and* at a **humped, non-Hull-White** vol, where nothing about the
exponential structure can be doing the work.

**Gaussian means deterministic `sigma(t,T)`**, which is what makes the forward rate normal, the bond
lognormal, and a Black-type bond option available (§4.2).

**The reduction is the arc's strongest anchor: already-verified code.** At
`sigma(t,T) = sigma e^{-kappa(T-t)}` the bond-option variance

    nu(0,S) = integral_0^S ( integral_S^T sigma(s,u) du )^2 ds

**is** tranche 10's `bond_option_variance` - measured identical to 1.3e-15 - so the one new closed
form here is checked against code that has carried its own oracles for fifty-nine tranches rather
than against a second transcription of itself. That is the pair Cowork named at tranche 68: anchor
by **reducing to what is already proved**, or by **an identity the answer must satisfy**. This
tranche has both, which is why its oracle spine is unusually short.

**The integrals are computed numerically rather than in closed form, deliberately.** `sigma(t,T)` is
an arbitrary deterministic function - that generality is the framework - so there is no closed form
to transcribe for the general case, and writing one for the Hull-White case only would be building
the special case the tranche exists to generalise. Quadrature of the printed integrand is the honest
route (§5: the source prints the integral, not its evaluation), and the Hull-White reduction then
checks the quadrature against a closed form the tree already holds.

Provenance:
  stock:  none - the archived predecessor had no HJM
  source: MCINERNEY-ZASTAWNIAK (docs/references.md) §4.1 (eq 4.1, the no-arbitrage drift condition
          in both its integrated and differentiated forms, Theorem 4.1) and §4.2 (Gaussian models -
          deterministic forward-rate volatility, hence a lognormal bond and Black-type bond
          options); transcribed in both renderings and reconciled
  oracle: tests/L3/test_gaussian_hjm.py - the drift condition asserted against the source's OTHER
          printed form (the integrated statement) and transcribed pointwise besides (T12(b) - a
          requirement never replaces a transcription), at a Hull-White vol AND at a humped one so
          the requirement is seen to hold for every sigma; the bond-option variance REDUCING to
          tranche 10's `bond_option_variance` at the Hull-White vol (1e-13, a cross-check against
          already-oracled code, flagged a consistency guard not independence); a MANDATORY humped
          companion measured 2%+ away so the general integral is doing real work (T39); both
          degenerate directions of the double integral; monotonicity in expiry and span; and
          put-call parity on the bond option, which the variance cancels out of
  crosscheck: quantlib GaussianHJM / its Hull-White engine, which coincides on the reduction -
          deferred to the optional QuantLib tier
  slice:  tranche 69 - the Gaussian HJM framework; generalised to `n` factors at
          tranche 70, where the one-factor model was shown to BE the `n`-factor one
          at `n = 1` rather than to resemble it
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from scipy.integrate import quad

from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.hull_white import HullWhiteParams

#: A deterministic forward-rate volatility `sigma(t, T)`. A plain callable rather than a type, and
#: that is the §6 call: the framework's content is that `sigma` may be *any* deterministic function,
#: so wrapping it would add vocabulary without adding meaning - and the two named constructors below
#: are what a caller actually reaches for. Each refuses an inverted interval, which is where a
#: silently-wrong sign would otherwise produce a plausible number.
ForwardVolatility = Callable[[float, float], float]


def hull_white_volatility(params: HullWhiteParams) -> ForwardVolatility:
    """`sigma(t,T) = sigma e^{-kappa(T-t)}` - the vol under which Gaussian HJM *is* Hull-White.

    Time-homogeneous: it depends only on the span `T - t`, which is what makes the short-rate model
    a special case rather than merely a possible one. Named rather than written inline at the call
    site, because it is the reduction anchor and a test needs to name the same object the model does.
    """
    def volatility(t: float, maturity: float) -> float:
        if maturity < t:
            raise ValueError(
                f"a forward volatility cannot be asked for a maturity ({maturity}) before the "
                f"observation time ({t})"
            )
        return params.sigma * math.exp(-params.kappa * (maturity - t))

    return volatility


def humped_volatility(level: float, slope: float, decay: float) -> ForwardVolatility:
    """`sigma(t,T) = (level + slope*(T-t)) e^{-decay(T-t)}` - a **humped** term structure.

    Deliberately **not** Hull-White: it rises then falls where an exponential only decays, so it is
    the T39 companion for every reduction in this module. A humped forward-rate vol is also the
    standard empirical shape, so this is the realistic case rather than a contrived one.
    """
    if level < 0.0:
        raise ValueError(f"the volatility level cannot be negative, got {level}")
    if decay <= 0.0:
        raise ValueError(f"the decay must be positive, got {decay}")
    if not all(math.isfinite(x) for x in (level, slope, decay)):
        raise ValueError(f"the parameters must be finite, got {level}, {slope}, {decay}")

    def volatility(t: float, maturity: float) -> float:
        if maturity < t:
            raise ValueError(
                f"a forward volatility cannot be asked for a maturity ({maturity}) before the "
                f"observation time ({t})"
            )
        span = maturity - t
        return (level + slope * span) * math.exp(-decay * span)

    return volatility


@dataclass(frozen=True)
class GaussianHjmModel:
    """The `n` deterministic forward-rate volatilities, carrying the snapshot they belong to (§3).

    **One type for any `n`, which is a ruling rather than a convenience** (tranche 70). A one-factor
    HJM is not a different model that resembles the multi-factor one - it *is* the multi-factor model
    at `n = 1`, as eq (4.11) makes plain by collapsing to eq (4.3). The argument that would normally
    keep two types apart is that each one's reduction to the other is a self-verifying oracle; here
    that reduction is **bit-identical by construction** (a one-element sum is its summand), so it is
    an identity rather than an independent cross-check, and two types would have been separated to
    preserve a tautology. *(Ruled by Cowork at the tranche-70 review, on Code's finding that the
    reduction is an identity; the one-factor behaviour stays oracled because this tranche's own
    transcription and Hull-White-reduction tests exercise the model at `n = 1`.)*

    An **ordered** tuple: the quantities are symmetric in the factors (a tested property - a sum of
    squares is order-independent where a cross term would not be), but a caller's bookkeeping is not.

    The initial forward curve is **not** a field: `f(0,T)` is `-d ln P(0,T)/dT`, so the snapshot's
    discount curve already carries it and a second copy could disagree with the first. That is the
    same reasoning tranche 67 used to leave G2++'s `phi` out - the curve is the fit.
    """

    market: MarketSnapshot
    volatilities: tuple[ForwardVolatility, ...]

    def __post_init__(self) -> None:
        if not self.volatilities:
            raise ValueError("a Gaussian HJM model needs at least one factor")


def hjm_drift(volatility: ForwardVolatility, t: float, maturity: float) -> float:
    """`alpha(t,T) = sigma(t,T) integral_t^T sigma(t,u) du` - Theorem 4.1's no-arbitrage drift.

    Forced by no-arbitrage rather than chosen, which is why the oracle asserts it as an identity for
    *any* `sigma` rather than checking it against a printed number for one.

    Zero at `T = t` because the inner integral is empty - a requirement companion, and the boundary
    where a sign error in either factor shows.
    """
    if maturity < t:
        raise ValueError(
            f"a drift cannot be asked for a maturity ({maturity}) before the observation time ({t})"
        )
    if maturity == t:
        return 0.0
    accumulated = quad(lambda u: volatility(t, u), t, maturity, epsabs=1e-15)[0]
    return volatility(t, maturity) * accumulated


def hjm_bond_option_variance(
    volatility: ForwardVolatility, option_expiry: float, bond_maturity: float
) -> float:
    """`nu(0,S) = integral_0^S ( integral_S^T sigma(s,u) du )^2 ds` - the variance of `ln B(S,T)`.

    The inner integral is the bond's exposure to the one factor as seen from `s`; the outer
    accumulates its square over the option's life, which is the Ito isometry applied to a
    deterministic integrand.

    **Computed by quadrature, and the reduction is what validates it:** at the Hull-White vol this
    is tranche 10's `bond_option_variance` to 1e-13, so the double integral is checked against a
    closed form the tree already proved rather than against a second reading of itself. A general
    `sigma(t,T)` has no closed form to transcribe, which is exactly the generality the framework is
    for.
    """
    if bond_maturity < option_expiry:
        raise ValueError(
            f"a bond cannot mature ({bond_maturity}) before the option expires ({option_expiry})"
        )
    if bond_maturity == option_expiry or option_expiry == 0.0:
        return 0.0

    def exposure(s: float) -> float:
        return quad(
            lambda u: volatility(s, u), option_expiry, bond_maturity, epsabs=1e-15
        )[0]

    return quad(lambda s: exposure(s) ** 2, 0.0, option_expiry, epsabs=1e-16)[0]
