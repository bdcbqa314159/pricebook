"""Dupire local volatility from a parametrized implied surface (L3).

    sigma_loc^2(K,T) = [ dC/dT + (r-q)*K*dC/dK + q*C ] / [ 0.5 * K^2 * d2C/dK2 ]

**The derivatives are ANALYTIC, by the chain rule, and that is a correctness decision rather than a
performance one.** `d2C/dK2` by finite differences subtracts near-equal numbers twice: measured on
our own surface it is wrong by 1% at `h = 1e-5`, best near `h = 1e-2`, and degrading again beyond —
a noise-dominated reference with a step-size sweet spot that would have to be argued for every
regime it is used in (§4's "a reference is code too", the shape that bit tranche 20a three times).
Our surface is closed-form and Black is closed-form, so there is no reason to accept that noise.
With the derivatives exact, the oracle's band is **pure Monte-Carlo standard error** and a Dupire
transcription error has nowhere to hide in it.

**The chain rule is the whole subtlety**, because `sigma` itself depends on `K` and `T`:

    dC/dK   = C_K + C_v * v_K
    d2C/dK2 = C_KK + 2*C_Kv*v_K + C_vv*v_K^2 + C_v*v_KK
    dC/dT   = C_T + C_v * v_T

Differentiating Black at *fixed* vol and stopping there is the error this formulation exists to make
impossible: on a skewed surface the `C_v * v_K` term is first-order, and the three `d2C/dK2`
cross-terms are what distinguish a local volatility from an implied one at all.

**Scope: a continuous dividend yield only.** Discrete cash and proportional dividends make `F(T)`
non-smooth in `T` (it jumps at each ex-date), so `dC/dT` is not defined there and the formula above
does not apply unaddressed. Rather than silently differentiating through a jump, this refuses the
input: the discrete-dividend Dupire is a different problem and is deferred, not approximated.

Provenance:
  stock:  none
  source: BROCKHAUS-2016 (docs/references.md) ch 6.1 - the Dupire local-volatility equation, read in
          `-raw` and `-layout` and reconciled; the Black partials are standard and are DERIVED here
          by the chain rule rather than transcribed, which §4 rates higher
  oracle: tests/L4/test_local_volatility.py - every analytic partial against the test's **own**
          finite differences on its own grid; the flat-surface reduction `sigma_loc = sigma` (exact)
          with the mandatory skewed companion where local skew is visibly steeper than implied
          (Dupire's rule of two); and the inversion anchor - the local-vol MC repricing the
          parametrized surface within Monte-Carlo standard error
  crosscheck: quantlib LocalVolSurface / Dupire calibration - deferred to the optional QuantLib tier
  slice:  tranche 32 - the implied-vol surface and Dupire local volatility
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from pricebook.foundation.distributions import norm_cdf
from pricebook.market.curve import CurveHandle
from pricebook.market.equity import EquityUnderlying, dividend_forward
from pricebook.market.equity_skew import ImpliedVolSkew

_SQRT_TWO_PI = math.sqrt(2.0 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_TWO_PI


@dataclass(frozen=True)
class ForwardState:
    """What the market fixes at one expiry: the forward, the horizon, discounting and the carry.

    Five quantities that always travel together and are always read from the same curve and the same
    underlying, so they are vocabulary rather than an argument list (§7). Bundling them also removes
    the failure a long signature invites, which is passing `rate` and `yield_rate` in the wrong
    order - both are small positive floats and the result would be plausible and wrong.
    """

    forward: float
    years: float
    discount_factor: float
    rate: float
    yield_rate: float

    @property
    def carry(self) -> float:
        """`r - q` - the forward's own drift, `dlog F/dT`."""
        return self.rate - self.yield_rate

    def moneyness_terms(self, strike: float, volatility: float) -> tuple[float, float]:
        """`(d1, d2)` - computed once here rather than re-derived in every partial below."""
        w = volatility * math.sqrt(self.years)
        d1 = (math.log(self.forward / strike) + 0.5 * w * w) / w
        return d1, d1 - w


@dataclass(frozen=True)
class BlackPartials:
    """The Black call's partials at fixed volatility - the pieces the chain rule composes.

    Grouped because they are computed from one set of `d1`/`d2` and are only ever used together;
    five loose return values would be §7's un-bundled vocabulary, and the order would be the kind of
    thing a caller gets silently wrong.
    """

    price: float
    d_strike: float
    d2_strike2: float
    d_vol: float
    d_strike_d_vol: float


def black_partials(state: ForwardState, strike: float, volatility: float) -> BlackPartials:
    """The call price and its partials in `K` and `sigma`, at **fixed** `sigma`.

    Standard Black-76 results, written out rather than finite-differenced:
    `C_K = -df*N(d2)`, `C_KK = df*n(d2)/(K*w)`, `C_v = df*F*n(d1)*sqrt(T)`,
    `C_Kv = df*n(d2)*d1/sigma`, with `w = sigma*sqrt(T)`.
    """
    d1, d2 = state.moneyness_terms(strike, volatility)
    root_t = math.sqrt(state.years)
    pdf2 = _norm_pdf(d2)
    df = state.discount_factor
    return BlackPartials(
        price=df * (state.forward * norm_cdf(d1) - strike * norm_cdf(d2)),
        d_strike=-df * norm_cdf(d2),
        d2_strike2=df * pdf2 / (strike * volatility * root_t),
        d_vol=df * state.forward * _norm_pdf(d1) * root_t,
        d_strike_d_vol=df * pdf2 * d1 / volatility,
    )


def _volga(
    partials: BlackPartials, state: ForwardState, strike: float, volatility: float
) -> float:
    """`C_vv = C_v * d1 * d2 / sigma` - the vol convexity the `v_K^2` chain term needs."""
    d1, d2 = state.moneyness_terms(strike, volatility)
    return partials.d_vol * d1 * d2 / volatility



@dataclass(frozen=True)
class SurfacePartials:
    """The call price and its `K`/`T` partials **along the surface** - after the chain rule.

    Named surface rather than locals inside `dupire_local_variance`, because these are exactly what
    the oracle has to reach: each one is checked against the test's own finite difference of the
    surface's own price function (§4 - extract the buried step so a test can see it). A Dupire that
    only exposed its final answer would hide which of the three partials was wrong.
    """

    price: float
    d_strike: float
    d2_strike2: float
    d_time: float


def surface_partials(
    surface: ImpliedVolSkew, strike: float, state: ForwardState
) -> SurfacePartials:
    """Compose Black's fixed-vol partials with the surface's own, by the chain rule.

    The three compositions in the module docstring, in one place. `sigma` depends on both `K` and
    `T`, so every `K`-derivative picks up a `C_v * v_K` term and the second picks up three.
    """
    forward, years = state.forward, state.years
    volatility = surface.volatility(strike, forward, years)
    v_k = surface.d_volatility_d_strike(strike, forward, years)
    v_kk = surface.d2_volatility_d_strike2(strike, forward, years)
    v_t = surface.d_volatility_d_time(strike, forward, years, state.carry)

    partials = black_partials(state, strike, volatility)
    volga = _volga(partials, state, strike, volatility)
    fixed_vol_d_time = _d_price_d_time(partials, state, strike, volatility)
    return SurfacePartials(
        price=partials.price,
        d_strike=partials.d_strike + partials.d_vol * v_k,
        d2_strike2=(
            partials.d2_strike2
            + 2.0 * partials.d_strike_d_vol * v_k
            + volga * v_k * v_k
            + partials.d_vol * v_kk
        ),
        d_time=fixed_vol_d_time + partials.d_vol * v_t,
    )


def dupire_local_variance(
    surface: ImpliedVolSkew,
    underlying: EquityUnderlying,
    strike: float,
    expiry: date,
    discount: CurveHandle,
) -> float:
    """`sigma_loc^2(K,T)` - the local **variance**, returned squared.

    Squared rather than rooted because that is what the equation produces and what a diffusion
    consumes; taking the root here would hide the one failure mode worth surfacing, which is a
    *negative* numerator or denominator (a butterfly or calendar arbitrage in the surface). The
    caller decides what a negative variance means - §3.4 says that is a value, not an exception.
    """
    schedule = underlying.dividends
    if schedule.cash or schedule.proportional:
        raise ValueError(
            "Dupire here supports a continuous dividend yield only: discrete dividends make the "
            "forward non-smooth in T, so dC/dT is undefined at an ex-date (deferred, not approximated)"
        )
    years = discount.time_measure.year_fraction(expiry)
    if years <= 0.0:
        raise ValueError(f"the local volatility needs a positive horizon, got {years}")

    df = discount.df(expiry)
    forward = dividend_forward(underlying, expiry, discount)
    rate = -math.log(df) / years
    yield_rate = schedule.continuous_yield + underlying.repo_rate
    carry = rate - yield_rate

    state = ForwardState(forward, years, df, rate, yield_rate)
    composed = surface_partials(surface, strike, state)

    numerator = composed.d_time + carry * strike * composed.d_strike + yield_rate * composed.price
    denominator = 0.5 * strike * strike * composed.d2_strike2
    if denominator == 0.0:
        raise ValueError("the local variance is undefined where d2C/dK2 vanishes")
    return numerator / denominator


def _d_price_d_time(
    partials: BlackPartials, state: ForwardState, strike: float, volatility: float
) -> float:
    """`dC/dT` at **fixed** volatility.

    `C = S*e^{-qT}N(d1) - K*e^{-rT}N(d2)`, so three terms: the two discount drifts and the one from
    `d1 - d2 = sigma*sqrt(T)` widening with maturity. Written in forward terms, `S*e^{-qT} = df*F`.
    """
    d1, d2 = state.moneyness_terms(strike, volatility)
    df = state.discount_factor
    return (
        -state.yield_rate * df * state.forward * norm_cdf(d1)
        + state.rate * df * strike * norm_cdf(d2)
        + df * strike * _norm_pdf(d2) * volatility / (2.0 * math.sqrt(state.years))
    )
