"""The 3/2-model variance swap (L3/L4). KWOK-ZHENG §2.75d, §4.2.2 eq (4.35); ZHAO-CUI-FLORESCU-2018 eq (19).

**Two routes to one number, which is the point of the tranche.** Production prices the fair variance
by the **reciprocal-CIR Poisson ladder**: `Y = 1/V` follows a CIR process (eq 4.35), so
`E[V_t] = E[1/Y_t]` is the **inverse first moment of a non-central chi-squared**, and that is a
Poisson-weighted sum of *elementary* terms `1/(dof + 2j − 2)`. No confluent hypergeometric, no scipy
in `src/`. The oracle is the other route: **eq (19)** of the held paper — the canonical Carr–Sun
Laplace transform, which *is* hypergeometric — differentiated at `λ = 0`, with scipy's `hyp1f1`
**in the test only** (the tranche-20a arrangement). Measured, the two agree to **8.1e-10**: an
elementary ladder against a hypergeometric transform, different derivations and different code.

**Why eq (19) and not the book's eq (3.30) — the arc's third source defect, and the first repaired by
acquiring the attributed source.** The book prints the transform as `₁F₁(−2/σ²U; α, γ)` —
**argument first** — and read that way it **diverges**: `L(0)` comes out 4.32 or 4.19 depending on
which rendering's grouping of `α` you take, and the renderings disagree. A Laplace transform of a
non-negative quantity must give `L(0) = 1`, so the printed form is not transcribable. Eq (3.30)
attributes itself to Zhao et al. (2018); that paper's **Proposition 3 / eq (19)** writes
`M(α, γ, −2/(σ²y_T))` — **argument last** — and then `L(0) = 1` **exactly**, because α's grouping
makes `α(0) = 0`, whence `M(0,γ,z) = 1`, `(·)^0 = 1` and `Γ(γ)/Γ(γ) = 1`. So the requirement oracle
does not merely *check* the form, it **pins α's grouping**, which is what the two renderings could
not settle between them.

**The parameter map is confirmed here, not trusted.** The paper's `dV = (pV + qV²)dt + σV^{3/2}dW`
is the book's and the spec's `dv = κv(θ−v)dt + εv^{3/2}dW` under `p = κθ`, `q = −κ`, `σ = ε`; the
book's §4.2.2 generalisation uses `+q` for the same coefficient, which is why eq (4.35)'s CIR drift
reads `[q + ε² − pY]` with a positive `q = κ`. Asserted below rather than assumed.
"""

import math
from datetime import date, timedelta

import numpy
import pytest

from pricebook.engine import price
from pricebook.engine.three_halves import (
    expected_variance,
    reciprocal_cir_law,
    stationary_variance,
    three_halves_fair_variance,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import EquityAsset
from pricebook.models.three_halves import ThreeHalvesModel, ThreeHalvesParams
from pricebook.products import VarianceSwap

scipy_special = pytest.importorskip("scipy.special")

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
RATE = 0.03
CURVE = DiscountCurve.flat(VAL, RATE, DC)
#: A NON-UNIT horizon, so a dropped `/T` in the annualisation is not arithmetically invisible.
EXPIRY = VAL + timedelta(days=475)
YEARS = 475.0 / 365.0
#: **`v0 != theta` is load-bearing.** At `v0 = theta` the process starts at its long-run level and
#: the mean-reversion terms collapse — several errors become invisible (the τ=1 shape, in the initial
#: condition). And a MATERIAL `epsilon`: the `epsilon -> 0` reduction is blind to the whole
#: stochastic term, so its companion must exercise one (T39).
PARAMS = ThreeHalvesParams(v0=0.09, kappa=2.0, theta=0.06, epsilon=0.4)
N = Money(1_000_000.0, EUR)


def _paper_transform(params: ThreeHalvesParams, years: float, decay: float) -> float:
    """Eq (19), transcribed from `ZHAO-CUI-FLORESCU-2018` Proposition 3 — the OTHER route.

    `E[e^{−λ∫V ds}] = Γ(γ−α)/Γ(γ) · (2/(σ²y_T))^α · M(α, γ, −2/(σ²y_T))`, **argument last**, with
    `y_t = V_t(e^{pT}−1)/p`, `α = −A + √(A² + 2λ/σ²)` for `A = ½ − q/σ²`, and `γ = 2(α + 1 − q/σ²)`.
    Reconciled across `-raw` and `-layout`. `scipy.special.hyp1f1` is used **here and never in
    `src/`** — the tranche-20a arrangement, which is exactly what lets production stay elementary.
    """
    p, q, sigma_squared = (
        params.kappa * params.theta,
        -params.kappa,
        params.epsilon * params.epsilon,
    )
    y = params.v0 * (math.exp(p * years) - 1.0) / p
    a_term = 0.5 - q / sigma_squared
    alpha = -a_term + math.sqrt(a_term * a_term + 2.0 * decay / sigma_squared)
    gamma = 2.0 * (alpha + 1.0 - q / sigma_squared)
    argument = -2.0 / (sigma_squared * y)
    return (
        math.gamma(gamma - alpha)
        / math.gamma(gamma)
        * (2.0 / (sigma_squared * y)) ** alpha
        * scipy_special.hyp1f1(alpha, gamma, argument)
    )


def _logistic_integral(params: ThreeHalvesParams, years: float) -> float:
    """`∫₀^T v dt` for the DETERMINISTIC `ε = 0` process, derived here independently.

    `dv/dt = κv(θ−v)` is the logistic equation — the mean-reversion *speed scales with `v`*, which is
    what makes it logistic rather than exponential — with solution
    `v(t) = θv₀ / (v₀ + (θ−v₀)e^{−κθt})`, whence
    `∫₀^T v dt = (1/κ)·ln[(v₀ + (θ−v₀)e^{−κθT}) / θ] + θT`. A genuine closed form, so the reduction
    is exact rather than a numerical limit."""
    inner = params.v0 + (params.theta - params.v0) * math.exp(
        -params.kappa * params.theta * years
    )
    return math.log(inner / params.theta) / params.kappa + params.theta * years


# ============================================================================================
# 1. The parameter map and the reciprocal-CIR law — confirmed, not trusted.
# ============================================================================================

def test_the_PARAMETER_MAP_between_the_paper_and_the_spec_is_what_it_claims():
    # `kappa*v*(theta - v) == p*v + q*v^2` with `p = kappa*theta`, `q = -kappa`, for every `v`.
    p, q = PARAMS.kappa * PARAMS.theta, -PARAMS.kappa
    for level in (0.001, 0.02, 0.06, 0.25, 1.0):
        assert PARAMS.kappa * level * (PARAMS.theta - level) == pytest.approx(
            p * level + q * level * level, rel=1e-15, abs=0.0
        )


def test_the_RECIPROCAL_process_is_the_CIR_of_eq_4_35():
    # eq (4.35): `dY = [q_book + eps^2 - p*Y]dt - eps*sqrt(Y)dW` with the BOOK's `q_book = +kappa`.
    # As a CIR that is speed `p = kappa*theta` and level `(kappa + eps^2)/(kappa*theta)`.
    speed, level, _ = reciprocal_cir_law(PARAMS)
    assert speed == pytest.approx(PARAMS.kappa * PARAMS.theta, rel=1e-15, abs=0.0)
    assert level == pytest.approx(
        (PARAMS.kappa + PARAMS.epsilon**2) / (PARAMS.kappa * PARAMS.theta), rel=1e-15, abs=0.0
    )


def test_the_INVERSE_MOMENT_always_exists_for_a_positive_kappa():
    # `E[1/Y]` needs `dof > 2`, and `dof = 4*speed*level/eps^2 = 4(kappa + eps^2)/eps^2 > 4` for any
    # positive kappa — so the Feller-type condition the source discusses is automatic here, and that
    # is worth asserting rather than guarding for.
    for params in (
        PARAMS,
        ThreeHalvesParams(0.01, 0.001, 0.9, 3.0),
        ThreeHalvesParams(2.0, 50.0, 0.01, 0.05),
    ):
        _, _, dof = reciprocal_cir_law(params)
        assert dof > 4.0, params
        assert dof == pytest.approx(
            4.0 * (params.kappa + params.epsilon**2) / params.epsilon**2, rel=1e-14, abs=0.0
        )


# ============================================================================================
# 2. The cross-ROUTE anchor: the elementary ladder against the hypergeometric transform.
# ============================================================================================

def test_the_paper_transform_satisfies_L_of_zero_EQUALS_ONE_exactly():
    # The requirement oracle that pins alpha's grouping: alpha(0) = 0 makes every factor exactly 1.
    # The book's argument-FIRST reading diverges instead, which is the defect (see the docstring).
    assert _paper_transform(PARAMS, YEARS, 0.0) == 1.0


def test_the_transform_is_a_DECREASING_function_of_the_decay():
    values = [_paper_transform(PARAMS, YEARS, d) for d in (0.0, 0.5, 2.0, 8.0)]
    assert values == sorted(values, reverse=True), values
    assert values[-1] < 0.5


@pytest.mark.parametrize(
    "params,years",
    [
        (PARAMS, YEARS),
        (ThreeHalvesParams(0.02, 1.2, 0.05, 0.3), 0.8),
        (ThreeHalvesParams(0.15, 3.0, 0.04, 0.6), 2.5),
    ],
)
def test_the_LADDER_agrees_with_the_TRANSFORM_two_routes_one_number(params, years):
    """The cross-route anchor. Production is an elementary Poisson ladder of inverse-central
    chi-squared moments; the oracle is a confluent-hypergeometric Laplace transform differentiated at
    zero. Different mathematics, different code, and scipy on one side only."""
    step = 1.0e-6
    from_transform = -(
        _paper_transform(params, years, step) - _paper_transform(params, years, 0.0)
    ) / step
    from_ladder = three_halves_fair_variance(params, years) * years
    # The band is the one-sided finite difference's own `O(h)` error, not the ladder's: measured at
    # ~8e-10 relative, so this is a tight cross-check rather than a loose agreement.
    assert from_ladder == pytest.approx(from_transform, rel=1e-7, abs=0.0)


# ============================================================================================
# 3. The epsilon -> 0 logistic reduction, and its mandatory companion.
# ============================================================================================

def test_a_VANISHING_vol_of_vol_reduces_to_the_LOGISTIC_integral():
    # The mean-reversion speed scales with `v`, so the deterministic limit is logistic, not
    # exponential — and its integral is closed form, derived independently in `_logistic_integral`.
    for epsilon in (1e-3, 1e-4, 1e-5):
        params = ThreeHalvesParams(PARAMS.v0, PARAMS.kappa, PARAMS.theta, epsilon)
        assert three_halves_fair_variance(params, YEARS) * YEARS == pytest.approx(
            _logistic_integral(params, YEARS), rel=1e-5, abs=0.0
        )


def test_the_reduction_CONVERGES_at_first_order_in_the_vol_of_vol():
    # Asserted as a RATE, since the reduction is approached rather than reached: the stochastic term
    # contributes at O(epsilon^2) to the variance but the fair variance picks it up at O(epsilon^2)
    # too — measured, and the rate is what makes the limit a statement rather than a coincidence.
    errors = []
    for epsilon in (0.2, 0.1, 0.05):
        params = ThreeHalvesParams(PARAMS.v0, PARAMS.kappa, PARAMS.theta, epsilon)
        exact = _logistic_integral(params, YEARS)
        errors.append(abs(three_halves_fair_variance(params, YEARS) * YEARS / exact - 1.0))
    assert errors == sorted(errors, reverse=True), errors
    for coarse, fine in zip(errors, errors[1:]):
        assert 2.5 < coarse / fine < 6.0, errors      # second order in epsilon, halving by halving


def test_the_COMPANION_a_material_vol_of_vol_moves_the_answer_away_from_the_logistic():
    """§4/T39's mandatory companion — and it needed its **own** fixture, which is the finding.

    The `ε → 0` reduction zeroes exactly the parameter the stochastic term lives in, so it is
    structurally blind to it and the companion must exercise a vol-of-vol where it bites. **The main
    fixture does not**: measured, `ε = 0.4` over this horizon separates the stochastic fair variance
    from the logistic one by only **5.8e-04** — a companion built on it would be nearly degenerate in
    the very parameter it exists to exercise, which is the T39 trap wearing a plausible fixture's
    clothes. The separation grows with both the vol-of-vol and the horizon: 2.4e-03 at `ε = 0.8`,
    5.7e-03 at 1.2, **1.9e-02 at 2.0**; and at `ε = 0.4` alone, 5.0e-03 by `T = 5` and 2.0e-02 by
    `T = 20`. So the companion uses a vol-of-vol of 2.0, and both halves are asserted — that it
    separates there, and that the main fixture is the weak case.
    """
    loud = ThreeHalvesParams(PARAMS.v0, PARAMS.kappa, PARAMS.theta, 2.0)
    separation = abs(
        three_halves_fair_variance(loud, YEARS) * YEARS / _logistic_integral(loud, YEARS) - 1.0
    )
    assert separation > 0.01, separation

    quiet = abs(
        three_halves_fair_variance(PARAMS, YEARS) * YEARS
        / _logistic_integral(PARAMS, YEARS)
        - 1.0
    )
    assert 1e-4 < quiet < 1e-3, quiet          # the main fixture: real, but nearly degenerate
    assert separation > 20.0 * quiet, (separation, quiet)


# ============================================================================================
# 4. Requirement oracles — structural, so they survive a transcription error (F-6.1).
# ============================================================================================

def test_the_fair_variance_is_POSITIVE_and_MONOTONE_in_the_initial_variance():
    values = [
        three_halves_fair_variance(
            ThreeHalvesParams(v0, PARAMS.kappa, PARAMS.theta, PARAMS.epsilon), YEARS
        )
        for v0 in (0.01, 0.04, 0.09, 0.20, 0.40)
    ]
    assert all(v > 0.0 for v in values)
    assert values == sorted(values), values
    assert values[-1] > 2.0 * values[0]


def test_the_fair_RATE_tends_to_the_STATIONARY_mean_which_is_NOT_theta():
    """**The spec's oracle 5 is wrong for a positive vol-of-vol, and this is the corrected form.**

    It asserts the long-horizon limit is `θ`. It is not: the 3/2 variance's stationary mean is
    `θ/(1 + ε²/(2κ))` — the vol-of-vol pulls it **below** the reversion level, by 3.8% at this
    fixture and **26%** at `ε = 1.2`. It falls out of the reciprocal-CIR law (`Y`'s stationary law is
    a central chi-squared, whose inverse mean is `1/(scale·(dof−2))`) and is verified to machine
    precision against the ladder. `θ` is the `ε → 0` case, which is asserted below as the reduction.
    """
    assert three_halves_fair_variance(PARAMS, 600.0) == pytest.approx(
        stationary_variance(PARAMS), rel=0.02, abs=0.0
    )
    # the premise that makes this a correction rather than a restatement: it differs from theta
    assert abs(stationary_variance(PARAMS) / PARAMS.theta - 1.0) > 0.03
    assert abs(PARAMS.v0 / PARAMS.theta - 1.0) > 0.4


def test_the_stationary_mean_REDUCES_to_theta_as_the_vol_of_vol_vanishes():
    # Which is why the spec's claim is the limiting case rather than simply wrong.
    for epsilon in (0.4, 0.1, 0.01):
        params = ThreeHalvesParams(PARAMS.v0, PARAMS.kappa, PARAMS.theta, epsilon)
        assert stationary_variance(params) < PARAMS.theta
    assert stationary_variance(
        ThreeHalvesParams(PARAMS.v0, PARAMS.kappa, PARAMS.theta, 1e-6)
    ) == pytest.approx(PARAMS.theta, rel=1e-12, abs=0.0)


def test_the_SPOT_limit_is_v0_as_the_horizon_vanishes():
    for short in (1e-4, 1e-6):
        assert three_halves_fair_variance(PARAMS, short) == pytest.approx(
            PARAMS.v0, rel=1e-3, abs=0.0
        )


def test_the_expected_variance_at_a_single_time_is_MONOTONE_toward_theta():
    # `E[V_t]` must move from `v0` toward `theta` monotonically — here downward, since `v0 > theta`.
    levels = [expected_variance(PARAMS, t) for t in (0.01, 0.2, 0.8, 2.0, 8.0)]
    assert levels == sorted(levels, reverse=True), levels
    assert levels[0] < PARAMS.v0 and levels[-1] > PARAMS.theta


# ============================================================================================
# 5. The engine.
# ============================================================================================

def _model(params: ThreeHalvesParams = PARAMS) -> ThreeHalvesModel:
    return ThreeHalvesModel(
        MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE})), params
    )


def test_the_EXISTING_variance_swap_prices_against_the_NEW_model():
    # No new product: `VarianceSwap` (T31) priced model-based rather than model-free. The registry's
    # `(product, model)` key is what makes that expressible — the third product to carry several.
    strike = 0.05
    swap = VarianceSwap(ACME, EXPIRY, strike, N)
    out = price(swap, _model())
    assert isinstance(out, PricingResult), out
    want = (
        N.amount
        * CURVE.df(EXPIRY)
        * (three_halves_fair_variance(PARAMS, YEARS) - strike)
    )
    assert out.pv.amount == pytest.approx(want, rel=1e-14, abs=0.0)


def test_a_MATURED_swap_is_PV_zero_invariant_6_and_the_fixture_is_IN_THE_MONEY():
    strike = 0.5 * three_halves_fair_variance(PARAMS, YEARS)
    after = EXPIRY + timedelta(days=1)
    model = ThreeHalvesModel(
        MarketSnapshot(
            valuation_date=after,
            curves=CurveSet({DISCOUNT_KEY: DiscountCurve.flat(after, RATE, DC)}),
        ),
        PARAMS,
    )
    out = price(VarianceSwap(ACME, EXPIRY, strike, N), model)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == 0.0
    # the premise: the fall-through would be a real number, not a coincidental zero
    assert three_halves_fair_variance(PARAMS, YEARS) - strike > 0.01


def test_a_MISSING_discount_curve_is_a_PricingFailure():
    bare = ThreeHalvesModel(MarketSnapshot(valuation_date=VAL, curves=CurveSet({})), PARAMS)
    assert isinstance(price(VarianceSwap(ACME, EXPIRY, 0.05, N), bare), PricingFailure)
