"""VIX derivatives under Heston: VIX² as the variance-swap rate, VIX futures and options (L4).

**The source section is §1.5.2 / eq (1.31), not §3.1.1 — finding one.** The T76 spec pins
"§3.1 (and §3.1.1 for the jump correction) … **this is the load-bearing identity**". §3.1.1 opens by
saying the opposite: *"Under the assumption of pure diffusion process of the underlying index value
process, we have shown **in Sec. 1.5.2** that VIX² equals the risk neutral expected cumulative
diffusion variance **[see (1.31)]**."* §3.1.1 is the **jumps** correction — Carr–Wu's quantification
of how VIX² and the variance-swap rate *differ* when the index jumps — and Heston has no jumps, so
it is out of scope for this tranche rather than load-bearing in it. The identity is eq (1.31):

    VIX² = E₀[(1/T)∫₀^T σ_t² dt] = (2e^{rT}/T)·[∫₀^{F₀} p(K)/K² dK + ∫_{F₀}^∞ c(K)/K² dK]

**Finding two: that is tranche 31's strip, verbatim** — the same `1/K²` weight, the same `2/T`, the
same OTM split at the forward. So "the model-free VIX² strip" the spec asks for as a cross-check is
literally `fair_variance`, and the Heston VIX² level is literally T75's `heston_fair_variance`. There
is no forward-window arrangement to build for the *level*: VIX²_t reads `[t, t+τ]`, which at `t = 0`
is a spot-starting window. The forward window appears only in the **futures**, which is where the
affine structure below comes from.

**Finding three: the affine coefficients fall out of T75's own formula, so there is no new
mathematics here either.** `VIX²_T = E_T[(1/τ)∫_T^{T+τ}V ds]` is T75's fair variance started at `T`
with initial variance `V_T`, i.e. `θ + (V_T−θ)(1−e^{−κτ})/(κτ)`. So

    VIX²_T = a·V_T + b,   a = (1−e^{−κτ})/(κτ),   b = θ(1−a)

measured equal to `heston_fair_variance` at 4.4e-16 across `V_T`.

**Finding four: the law of `V_T` is non-central chi-squared, so tranche 72's primitive prices these.**
`V_T = c·χ'²_d(λ)` with `c = ε²(1−e^{−κT})/(4κ)`, `d = 4κθ/ε²`,
`λ = 4κe^{−κT}V₀/(ε²(1−e^{−κT}))` — verified against a 400k-path simulation at the 10th, 50th and
90th percentiles (0.24–0.65%, the Euler bias). The spec mentioned "non-central χ² / the transform"
parenthetically; this makes it the construction.

**And integrating BY PARTS turns both payoffs into one integrand against the SURVIVAL function**,
which is the thing tranche 72 built for tail accuracy (F-8.6's whole subject). With
`u = √(ax+b) − K` vanishing at the exercise boundary and `S = P(V_T > x)`:

    E[(VIX_T − K)⁺] = ∫_{x_K}^∞ (a / 2√(ax+b)) · S(x) dx ,   x_K = max((K²−b)/a, 0)
    E[VIX_T]        = √b + ∫_0^∞ (a / 2√(ax+b)) · S(x) dx

No density is needed, nothing is differentiated, and the integrand decays with the chi-squared tail
where `noncentral_chisquare_sf` is accurate rather than where `1 − cdf` collapses to zero.
"""

import math
from datetime import date, timedelta

import numpy
import pytest

from pricebook.engine import price
from pricebook.engine.realized_variance_option import heston_fair_variance
from pricebook.engine.vix import (
    vix_affine_coefficients,
    vix_future_value,
    vix_option_value,
    vix_squared,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.inflation import CapFloorKind
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.vix import VixFuture, VixOption

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RATE = 0.03
CURVE = DiscountCurve.flat(VAL, RATE, DC)
#: The index's own window: 30 calendar days (§1.5.2).
WINDOW = 30.0 / 365.0
#: A NON-UNIT maturity, so a dropped horizon is not arithmetically invisible.
MATURITY = VAL + timedelta(days=274)
MATURITY_YEARS = 274.0 / 365.0
#: **`v0 != theta` is load-bearing**, not decoration: with the spot variance already at the long-run
#: level, `[0,T]` and `[T,T+tau]` carry the *same* expected variance, so a mutation that drops the
#: forward offset and reads the spot window survives — §4's translation-invariance trap (T17), here
#: in the time dimension. Mean-reverting away from `theta` is what separates the two windows.
PARAMS = HestonParams(kappa=1.8, theta=0.06, sigma_v=0.55, rho=-0.7, v0=0.09)
N = Money(1_000.0, EUR)


def _simulate_terminal_variance(
    params: HestonParams, years: float, paths: int = 300_000, steps: int = 1200
) -> "numpy.ndarray":
    """`V_T` by full-truncation Euler — the independent law the closed forms are checked against."""
    rng = numpy.random.default_rng(20261123)
    step = years / steps
    root = math.sqrt(step)
    variance = numpy.full(paths, params.v0)
    for _ in range(steps):
        floored = numpy.maximum(variance, 0.0)
        variance = (
            variance
            + params.kappa * (params.theta - variance) * step
            + params.sigma_v * numpy.sqrt(floored) * rng.standard_normal(paths) * root
        )
    return numpy.maximum(variance, 0.0)


@pytest.fixture(scope="module")
def terminal_variance():
    """One simulation, shared — the draws do not depend on the strike. Lossless, and the `slow`
    gate's lesson applied before CI has to find it (F-8.9, T74/T75)."""
    return _simulate_terminal_variance(PARAMS, MATURITY_YEARS)


# ============================================================================================
# 1. The identity: VIX squared IS the variance-swap rate (eq 1.31) — a REQUIREMENT oracle.
# ============================================================================================

def test_VIX_SQUARED_is_the_variance_swap_rate_which_is_the_whole_identity():
    # Forced by the definition, so it survives a transcription error a source-value oracle would
    # share (the F-6.1 reasoning). Under Heston the variance-swap rate over [0, tau] is T75's fair
    # variance, and eq (1.31) says VIX squared IS that rate.
    assert vix_squared(PARAMS, WINDOW) == heston_fair_variance(PARAMS, WINDOW)


def test_the_AFFINE_coefficients_reproduce_the_fair_variance_at_every_starting_variance():
    # VIX^2_T = a*V_T + b, with a and b from T75's formula started at T — no new mathematics.
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    for level in (0.005, 0.02, 0.09, 0.25, 0.80):
        started = HestonParams(PARAMS.kappa, PARAMS.theta, PARAMS.sigma_v, PARAMS.rho, level)
        assert a * level + b == pytest.approx(
            heston_fair_variance(started, WINDOW), rel=1e-14, abs=0.0
        )


def test_the_affine_coefficients_are_the_PRINTED_forms():
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    decay = 1.0 - math.exp(-PARAMS.kappa * WINDOW)
    assert a == pytest.approx(decay / (PARAMS.kappa * WINDOW), rel=1e-15, abs=0.0)
    assert b == pytest.approx(PARAMS.theta * (1.0 - a), rel=1e-15, abs=0.0)
    # the premise that `a` is not degenerate: over a 30-day window it is near but not equal to 1
    assert 0.9 < a < 1.0


# ============================================================================================
# 2. The futures and options against the simulated law of V_T.
# ============================================================================================

def test_the_FUTURE_matches_the_simulated_expectation(terminal_variance):
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    simulated_vix = numpy.sqrt(a * terminal_variance + b)
    mean = float(simulated_vix.mean())
    standard_error = float(simulated_vix.std(ddof=1)) / math.sqrt(simulated_vix.size)
    got = vix_future_value(PARAMS, MATURITY_YEARS, WINDOW)
    # Banded `4*SE + the Euler scheme's measured bias`: the simulation runs ~1.5 SE high across the
    # future and every strike, consistently, which is discretisation and not sampling.
    assert abs(got - mean) < 4.0 * standard_error + 0.005 * mean, (got, mean, standard_error)


@pytest.mark.parametrize("strike", [0.20, 0.28, 0.40])
def test_the_OPTION_matches_the_simulated_expectation(strike: float, terminal_variance):
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    payoff = numpy.maximum(numpy.sqrt(a * terminal_variance + b) - strike, 0.0)
    mean = float(payoff.mean())
    standard_error = float(payoff.std(ddof=1)) / math.sqrt(payoff.size)
    got = vix_option_value(PARAMS, MATURITY_YEARS, WINDOW, strike, CapFloorKind.CAP)
    assert abs(got - mean) < 4.0 * standard_error + 0.01 * mean, (got, mean, standard_error)


def test_the_simulation_RESOLVES_the_quantities_it_is_judging():
    # The band-resolution premise: at 300k paths the SE must be small against the value, else the
    # agreement above asserts only that both numbers are noisy.
    sample = _simulate_terminal_variance(PARAMS, MATURITY_YEARS, paths=300_000, steps=1200)
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    vix = numpy.sqrt(a * sample + b)
    relative_error = float(vix.std(ddof=1)) / math.sqrt(vix.size) / float(vix.mean())
    assert relative_error < 0.002, relative_error


# ============================================================================================
# 3. Requirement oracles — convexity, monotonicity, parity.
# ============================================================================================

def test_the_FUTURE_sits_BELOW_the_square_root_of_VIX_SQUARED_by_JENSEN():
    # `E[sqrt(X)] <= sqrt(E[X])`, forced by concavity — and STRICT under genuine vol-of-vol. The
    # forward VIX^2 is `a*E[V_T] + b`, which is the fair variance over the forward window.
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    forward_vix_squared = a * _expected_terminal_variance(PARAMS, MATURITY_YEARS) + b
    future = vix_future_value(PARAMS, MATURITY_YEARS, WINDOW)
    assert future < math.sqrt(forward_vix_squared)
    # materially, not on rounding noise — the convexity gap is the whole content of a VIX future
    assert future < 0.995 * math.sqrt(forward_vix_squared)


def _expected_terminal_variance(params: HestonParams, years: float) -> float:
    """`E[V_T] = theta + (v0 - theta)e^{-kappa T}` — the CIR mean, formed here independently."""
    return params.theta + (params.v0 - params.theta) * math.exp(-params.kappa * years)


def test_the_convexity_gap_VANISHES_as_the_vol_of_vol_does():
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    gaps = []
    for sigma_v in (0.55, 0.2, 0.05, 1e-6):
        params = HestonParams(PARAMS.kappa, PARAMS.theta, sigma_v, PARAMS.rho, PARAMS.v0)
        level = math.sqrt(a * _expected_terminal_variance(params, MATURITY_YEARS) + b)
        gaps.append(level - vix_future_value(params, MATURITY_YEARS, WINDOW))
    assert gaps == sorted(gaps, reverse=True), gaps
    assert gaps[-1] < 1e-6 < gaps[0]


def test_the_OPTION_is_monotone_in_the_strike_and_obeys_PARITY():
    values = [
        vix_option_value(PARAMS, MATURITY_YEARS, WINDOW, k, CapFloorKind.CAP)
        for k in (0.15, 0.22, 0.30, 0.45)
    ]
    assert values == sorted(values, reverse=True), values
    for strike in (0.18, 0.28, 0.40):
        call = vix_option_value(PARAMS, MATURITY_YEARS, WINDOW, strike, CapFloorKind.CAP)
        put = vix_option_value(PARAMS, MATURITY_YEARS, WINDOW, strike, CapFloorKind.FLOOR)
        future = vix_future_value(PARAMS, MATURITY_YEARS, WINDOW)
        assert call - put == pytest.approx(future - strike, rel=1e-6, abs=1e-12)


# ============================================================================================
# 4. The forward window must really be FORWARD — the translation-invariance trap (T17).
# ============================================================================================

def test_the_forward_window_is_NOT_the_spot_window_which_needs_v0_away_from_theta():
    # If `v0 == theta` the two windows carry identical expected variance and a mutation reading the
    # spot window survives. The fixture mean-reverts away from theta, so they differ — asserted.
    spot = vix_squared(PARAMS, WINDOW)
    a, b = vix_affine_coefficients(PARAMS, WINDOW)
    forward = a * _expected_terminal_variance(PARAMS, MATURITY_YEARS) + b
    assert abs(forward / spot - 1.0) > 0.2, (spot, forward)
    # and the premise behind the premise: at v0 == theta they would coincide
    flat = HestonParams(PARAMS.kappa, PARAMS.theta, PARAMS.sigma_v, PARAMS.rho, PARAMS.theta)
    assert vix_squared(flat, WINDOW) == pytest.approx(
        a * _expected_terminal_variance(flat, MATURITY_YEARS) + b, rel=1e-12, abs=0.0
    )


# ============================================================================================
# 5. The engines.
# ============================================================================================

def _model(params: HestonParams = PARAMS) -> HestonModel:
    return HestonModel(
        MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE})), params
    )


def test_the_FUTURE_prices_UNDISCOUNTED_because_a_futures_price_is_an_expectation():
    # eq (3.6): `F_t = E_t[VIX_T]`, with no discount factor — a futures *price*, not a forward's PV.
    out = price(VixFuture(MATURITY, WINDOW, N), _model())
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == pytest.approx(
        N.amount * vix_future_value(PARAMS, MATURITY_YEARS, WINDOW), rel=1e-14, abs=0.0
    )


def test_the_OPTION_prices_DISCOUNTED_because_eq_3_7a_carries_the_discount():
    strike = 0.25
    out = price(VixOption(MATURITY, WINDOW, strike, N, CapFloorKind.CAP), _model())
    assert isinstance(out, PricingResult), out
    want = (
        N.amount
        * CURVE.df(MATURITY)
        * vix_option_value(PARAMS, MATURITY_YEARS, WINDOW, strike, CapFloorKind.CAP)
    )
    assert out.pv.amount == pytest.approx(want, rel=1e-14, abs=0.0)


def test_an_EXPIRED_future_and_an_expired_IN_THE_MONEY_option_are_PV_zero_invariant_6():
    after = MATURITY + timedelta(days=1)
    model = HestonModel(
        MarketSnapshot(
            valuation_date=after,
            curves=CurveSet({DISCOUNT_KEY: DiscountCurve.flat(after, RATE, DC)}),
        ),
        PARAMS,
    )
    deep = 0.5 * vix_future_value(PARAMS, MATURITY_YEARS, WINDOW)
    for product in (
        VixFuture(MATURITY, WINDOW, N),
        VixOption(MATURITY, WINDOW, deep, N, CapFloorKind.CAP),
    ):
        out = price(product, model)
        assert isinstance(out, PricingResult), (product, out)
        assert out.pv.amount == 0.0
    # the premise: the expired option is IN the money, so the fall-through is not a coincidental 0
    assert vix_future_value(PARAMS, MATURITY_YEARS, WINDOW) - deep > 0.05


def test_a_MISSING_discount_curve_is_a_PricingFailure_for_the_option():
    bare = HestonModel(MarketSnapshot(valuation_date=VAL, curves=CurveSet({})), PARAMS)
    out = price(VixOption(MATURITY, WINDOW, 0.25, N, CapFloorKind.CAP), bare)
    assert isinstance(out, PricingFailure), out
