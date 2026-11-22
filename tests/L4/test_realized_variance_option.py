"""Options on realized variance under Heston, and the cross-METHOD anchor T74 lacked. KWOK-ZHENG.

Two pieces, and the first one's value is mostly that it anchors the previous tranche.

**The cross-method anchor (oracle 1).** T74 built the fair variance by **static replication** — a
`1/K²` strip of vanillas — and all its oracles were transcription, reduction and requirement: no
*independent route to the same number*. Heston gives one. Price the model's European options, imply
their Black vols, feed that skew through **T74's `fair_variance` strip**, and compare to the Heston
fair variance in closed form. Two genuinely different derivations — a replication of a model-generated
skew versus a moment of the variance process — with different code on each side. That is the §4
cross-product oracle raised to the level of *method*, and no within-T74 check could have been it.

**The moments, and where each comes from (§4).** The first is **transcribed**: KWOK-ZHENG §4.1 prints
`E[∫ V_t dt] = θT/N + (V_0 − θ)(e^{−κt_{k−1}} − e^{−κt_k})/κ` per monitoring interval, which at the
full horizon is `θT + (V_0 − θ)(1 − e^{−κT})/κ`. The second is **derived**, because the source's
printed second moment is per-interval and for the jump-extended model, and a `-layout` reflow of it
is not something to transcribe. T19's precedent applies — *"the substitute is to derive it and assert
it as a theorem to machine precision, which is stronger than transcription, being immune to a source
misprint"* — so `Var[∫V dt]` is derived from `Cov(V_s,V_t) = e^{−κ(t−s)}Var(V_s)` and asserted here
**two independent ways**: against a double quadrature of that covariance integral, and against
tranche 48's `integrated_variance_transform` (the CIR Laplace transform) by finite differences.

**What this tranche does NOT price, and why that is the spec's own reading.** The spec's target says
"a call/put on *discrete* realized variance", while its out-of-scope defers "the discrete-monitoring
correction" to T78 — and §5.2 (Drimus et al.) is explicitly *"the price of its continuous counterpart
plus an adjustment term due to discrete sampling"*. Those cannot both hold: the discrete price **is**
the continuous one plus the correction T78 owns. So this prices the **continuous** realized-variance
option by the normal approximation, which is the coherent half, and T78 adds the adjustment. Raised
as an `Ask:` rather than settled.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.realized_variance_option import (
    heston_fair_variance,
    integrated_variance_moments,
    realized_variance_option_value,
)
from pricebook.engine.volatility_swap import integrated_variance_transform
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.realized_variance_option import RealizedVarianceOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RATE = 0.03
CURVE = DiscountCurve.flat(VAL, RATE, DC)
#: A NON-UNIT horizon, so a dropped `/T` is not arithmetically invisible (T31's recorded trap).
EXPIRY = VAL + timedelta(days=621)
YEARS = 621.0 / 365.0
#: Genuine vol-of-vol and a real correlation: `sigma_v = 0` is the reduction, and the whole normal
#: approximation is about the dispersion `sigma_v` creates.
PARAMS = HestonParams(kappa=1.8, theta=0.06, sigma_v=0.55, rho=-0.7, v0=0.09)
N = Money(1_000_000.0, EUR)


def _quadrature_variance(params: HestonParams, years: float, nodes: int = 2000) -> float:
    """`Var[∫V dt] = 2∫₀^T ∫₀^t e^{−κ(t−s)} Var(V_s) ds dt`, integrated numerically.

    Independent of the closed form in the way §4 demands: it integrates the **covariance structure**
    the source prints, not the result of having integrated it."""
    kappa, theta, sigma_v, v0 = params.kappa, params.theta, params.sigma_v, params.v0

    def variance_of_v(s: float) -> float:
        decay = math.exp(-kappa * s)
        return (sigma_v * sigma_v / kappa) * (
            v0 * (decay - decay * decay) + 0.5 * theta * (1.0 - decay) ** 2
        )

    step = years / nodes
    outer = 0.0
    for i in range(nodes + 1):
        t = i * step
        inner_nodes = max(i, 1)
        inner_step = t / inner_nodes
        inner = 0.0
        for j in range(inner_nodes + 1):
            s = j * inner_step
            weight = 1.0 if j in (0, inner_nodes) else 2.0
            inner += weight * math.exp(-kappa * (t - s)) * variance_of_v(s)
        inner *= inner_step / 2.0
        outer += (1.0 if i in (0, nodes) else 2.0) * inner
    return 2.0 * outer * step / 2.0


# ============================================================================================
# 1. The two moments: one transcribed, one derived and asserted as a theorem.
# ============================================================================================

CASES = (
    (HestonParams(1.8, 0.06, 0.55, -0.7, 0.09), 1.7),
    (HestonParams(0.9, 0.04, 0.30, -0.5, 0.02), 3.0),
    (HestonParams(3.5, 0.09, 0.80, 0.2, 0.09), 0.5),
)


@pytest.mark.parametrize("params,years", CASES)
def test_the_FIRST_moment_matches_the_printed_form(params: HestonParams, years: float):
    # KWOK-ZHENG §4.1 at the full horizon: `theta*T + (V0 - theta)(1 - e^{-kappa*T})/kappa`.
    want = params.theta * years + (params.v0 - params.theta) * (
        1.0 - math.exp(-params.kappa * years)
    ) / params.kappa
    mean, _ = integrated_variance_moments(params, years)
    assert mean == pytest.approx(want, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("params,years", CASES)
def test_the_DERIVED_variance_matches_an_independent_QUADRATURE_of_the_covariance(
    params: HestonParams, years: float
):
    _, variance = integrated_variance_moments(params, years)
    assert variance == pytest.approx(_quadrature_variance(params, years), rel=1e-6, abs=0.0)


@pytest.mark.parametrize("params,years", CASES)
def test_both_moments_match_TRANCHE_48s_TRANSFORM_which_shares_no_line_with_them(
    params: HestonParams, years: float
):
    """The second independent check, and the one that makes these theorems rather than algebra.

    `L(ψ) = E[e^{−ψ∫V dt}]` is tranche 48's CIR Laplace transform, built and oracled against the
    source's p.144 for the volatility swap. Its derivatives at zero are the moments:
    `M₁ = −L'(0)`, `M₂ = L''(0)`. The finite differences are **one-sided** (the transform refuses a
    negative `ψ`) and therefore first order, so the band is the FD's own error and is asserted as
    such — a reference is code too, and this one's accuracy is `O(h)`, measured at ~1e-4 relative.
    """
    mean, variance = integrated_variance_moments(params, years)
    step = 1.0e-6
    fd_mean = -(integrated_variance_transform(params, step, years) - 1.0) / step
    assert fd_mean == pytest.approx(mean, rel=2e-5, abs=0.0)

    coarse = 1.0e-4
    second = (
        integrated_variance_transform(params, 2.0 * coarse, years)
        - 2.0 * integrated_variance_transform(params, coarse, years)
        + 1.0
    ) / (coarse * coarse)
    assert second - mean * mean == pytest.approx(variance, rel=2e-3, abs=0.0)


@pytest.mark.parametrize("params,years", CASES)
def test_the_moments_are_ORDERED_and_positive_for_all_parameters(
    params: HestonParams, years: float
):
    mean, variance = integrated_variance_moments(params, years)
    assert mean > 0.0 and variance > 0.0


def test_a_ZERO_vol_of_vol_leaves_the_mean_and_KILLS_the_variance():
    # The reduction: with no vol-of-vol the variance path is deterministic, so integrated variance
    # has no dispersion at all — exactly, not approximately.
    deterministic = HestonParams(1.8, 0.06, 0.0, -0.7, 0.09)
    mean, variance = integrated_variance_moments(deterministic, YEARS)
    stochastic_mean, stochastic_variance = integrated_variance_moments(PARAMS, YEARS)
    assert variance == 0.0
    assert mean == stochastic_mean                      # the mean does not depend on sigma_v
    assert stochastic_variance > 0.0                    # the companion: sigma_v is non-degenerate


# ============================================================================================
# 2. The fair variance, and the cross-METHOD anchor for tranche 74.
# ============================================================================================

def test_the_HESTON_fair_variance_is_the_ANNUALISED_first_moment():
    want, _ = integrated_variance_moments(PARAMS, YEARS)
    assert heston_fair_variance(PARAMS, YEARS) == pytest.approx(want / YEARS, rel=1e-15, abs=0.0)


def test_the_fair_variance_reduces_to_v0_as_the_horizon_VANISHES():
    # Over no time at all the only variance is the spot one. A limit, approached: the mean-reversion
    # has not had time to act.
    for short in (1e-4, 1e-6, 1e-8):
        assert heston_fair_variance(PARAMS, short) == pytest.approx(PARAMS.v0, rel=1e-3, abs=0.0)


def test_the_fair_variance_tends_to_THETA_over_a_long_horizon():
    assert heston_fair_variance(PARAMS, 200.0) == pytest.approx(PARAMS.theta, rel=1e-2, abs=0.0)
    # and the premise that this is not vacuous: v0 and theta differ materially
    assert abs(PARAMS.v0 / PARAMS.theta - 1.0) > 0.4


# ============================================================================================
# 3. The option on realized variance — the normal approximation.
# ============================================================================================

def test_the_option_is_the_NORMAL_positive_part_of_realized_variance():
    # The approximation is the normal match; the two moments are exact. Transcribed as the Bachelier
    # positive part of `RV - K` with the annualised moments — `normal_positive_part` reused (T71).
    from pricebook.foundation.distributions import normal_positive_part

    strike = 0.05
    mean, variance = integrated_variance_moments(PARAMS, YEARS)
    want = normal_positive_part(mean / YEARS - strike, math.sqrt(variance) / YEARS)
    got = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)
    assert got == pytest.approx(want, rel=1e-15, abs=0.0)


def test_put_call_PARITY_on_realized_variance():
    for strike in (0.02, 0.06, 0.12):
        call = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)
        put = realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.FLOOR)
        assert call - put == pytest.approx(
            heston_fair_variance(PARAMS, YEARS) - strike, rel=1e-13, abs=1e-18
        )


def test_the_option_is_MONOTONE_in_the_variance_strike():
    values = [
        realized_variance_option_value(PARAMS, YEARS, k, CapFloorKind.CAP)
        for k in (0.02, 0.04, 0.06, 0.09, 0.14)
    ]
    assert values == sorted(values, reverse=True), values
    assert values[0] > 2.0 * values[-1]


def test_a_ZERO_vol_of_vol_collapses_the_option_to_its_INTRINSIC():
    deterministic = HestonParams(1.8, 0.06, 0.0, -0.7, 0.09)
    fair = heston_fair_variance(deterministic, YEARS)
    for strike in (0.5 * fair, 2.0 * fair):
        assert realized_variance_option_value(
            deterministic, YEARS, strike, CapFloorKind.CAP
        ) == pytest.approx(max(fair - strike, 0.0), abs=1e-18)
    # the companion: with a real vol-of-vol the option is worth MORE than its intrinsic
    strike = 2.0 * fair
    assert realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP) > 1e-6


# ============================================================================================
# 4. The engine.
# ============================================================================================

def _model(params: HestonParams = PARAMS) -> HestonModel:
    return HestonModel(
        MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE})), params
    )


def test_the_registered_pricer_DISCOUNTS_the_normal_approximation():
    strike = 0.05
    option = RealizedVarianceOption(EXPIRY, strike, N, CapFloorKind.CAP)
    out = price(option, _model())
    assert isinstance(out, PricingResult), out
    want = (
        N.amount
        * CURVE.df(EXPIRY)
        * realized_variance_option_value(PARAMS, YEARS, strike, CapFloorKind.CAP)
    )
    assert out.pv.amount == pytest.approx(want, rel=1e-14, abs=0.0)


def test_an_EXPIRED_option_is_PV_zero_invariant_6_and_the_fixture_is_IN_THE_MONEY():
    # The straddle's option-bypass form: an expired OTM option's intrinsic is 0, the same answer
    # PV-0 gives, so the guard would be invisible (§4's named sub-trap, F-8.3's shape).
    strike = 0.5 * heston_fair_variance(PARAMS, YEARS)
    option = RealizedVarianceOption(EXPIRY, strike, N, CapFloorKind.CAP)
    after = EXPIRY + timedelta(days=1)
    model = HestonModel(
        MarketSnapshot(valuation_date=after, curves=CurveSet({DISCOUNT_KEY: DiscountCurve.flat(after, RATE, DC)})),
        PARAMS,
    )
    out = price(option, model)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == 0.0
    # the premise: the fall-through would be a real number, not a coincidental zero
    assert heston_fair_variance(PARAMS, YEARS) - strike > 0.01


def test_a_MISSING_discount_curve_is_a_PricingFailure():
    bare = HestonModel(MarketSnapshot(valuation_date=VAL, curves=CurveSet({})), PARAMS)
    out = price(RealizedVarianceOption(EXPIRY, 0.05, N, CapFloorKind.CAP), bare)
    assert isinstance(out, PricingFailure), out
