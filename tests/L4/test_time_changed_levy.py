"""T78 — variance swaps under time-changed Lévy processes, and the discrete-monitoring adjustment.

`KWOK-ZHENG` §2.4 (framework), §4.4.1 (Prop 4.1/4.2 — the multiplier), §4.4.2 (Prop 4.3 — the
discrete decomposition), §4.4.4 (Prop 4.8 — the `O(1/N)` rate), §1.5.3.

**Two corrections to the spec are asserted here rather than assumed** (thread 0417):

1. The spec's oracle 2 anchors on *"the adjustment VANISHES in the pure-diffusion limit"*. **It does
   not.** Prop 4.3's first correction term survives with no jumps at all, and the pure-diffusion
   adjustment is exactly `(r − q − σ²/2)²·T/N`. The adjustment vanishes as `N → ∞`, not as the jumps
   vanish — and the replacement anchor is **stronger** than the spec's, being an exact nonzero closed
   form rather than a zero (which a no-op adjustment would also satisfy, §4's vacuous-reduction trap).

2. **Prop 4.8 contradicts Prop 4.2 + Prop 4.1.** It prints `Kv(∞) = −(1/T)φ⁽²⁾(0)E[Z_T]`, which under
   the book's own (4.84) is `E[L₁²]`; the multiplier route gives `var(L₁)`. They differ by `(E[L₁])²`
   unless `L` is centred. We take `var(L₁)` — quadratic variation is **drift-free**, so a strike
   cannot depend on `E[L₁]` — and the discrepancy is pinned below rather than left to the reader (§4).
"""

import math
from datetime import date, timedelta
from decimal import Decimal, getcontext

import numpy
import pytest

from pricebook.engine import price
from pricebook.engine.time_changed_levy import (
    clock_increment_variance,
    discrete_monitoring_adjustment,
    levy_cumulant,
    levy_fair_variance,
    levy_mean,
    levy_multiplier,
    levy_variance,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import EquityAsset, EquityUnderlying
from pricebook.models.time_changed_levy import (
    ActivityRateParams,
    LevyParams,
    TimeChangedLevyModel,
)
from pricebook.products import VarianceSwap
from pricebook.products.discrete_variance_swap import DiscreteVarianceSwap

VAL = date(2026, 1, 15)
ACME = EquityAsset("ACME")
YEARS = 2.0
EXPIRY = VAL + timedelta(days=730)
USD = Currency("USD")

# A genuine-jump fixture: C is material, and G != M so the jumps are SKEWED (a symmetric Levy
# measure would make the third cumulant vanish and hide a sign error -- the degeneracy catalogue).
JUMPS = LevyParams(sigma=0.20, c=2.0, g=5.0, m=10.0)
# The pure-diffusion reduction: no jumps at all.
DIFFUSION = LevyParams(sigma=0.25, c=0.0, g=5.0, m=10.0)
CLOCK = ActivityRateParams(kappa=2.0, vol_of_vol=0.5)
# A DETERMINISTIC clock -- the reduction that kills the time change (vol_of_vol = 0).
FLAT_CLOCK = ActivityRateParams(kappa=2.0, vol_of_vol=0.0)

RATE, DIVIDEND = 0.03, 0.01


def _market(valuation: date = VAL) -> MarketSnapshot:
    curve = DiscountCurve.flat(valuation, RATE, DayCountConvention.ACT_365_FIXED)
    return MarketSnapshot(
        valuation_date=valuation,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): curve}),
        # The repo/borrow rate enters the forward exactly as a yield does, so it IS the `q` of the
        # `r − q` carry (see `market/equity.py`); a non-zero one keeps the carry from collapsing to
        # the discount rate, where a dropped dividend leg would be invisible.
        equities={ACME: EquityUnderlying(spot=100.0, repo_rate=DIVIDEND)},
    )


def _model(levy: LevyParams = JUMPS, clock: ActivityRateParams = CLOCK) -> TimeChangedLevyModel:
    return TimeChangedLevyModel(market=_market(), levy=levy, clock=clock)


# ============================================================================================
# 1. The cumulant exponent and the multiplier — transcribed POINTWISE against Table 4.3.
# ============================================================================================

def _table_4_3_cgmy_multiplier(g: float, m: float) -> float:
    """Table 4.3's CGMY row, transcribed independently of the implementation.

    `(G⁻² + M⁻²) / (G⁻¹ − ln(1+G⁻¹) − M⁻¹ − ln(1−M⁻¹))`. The row carries **no `Y` and no `Γ`**
    while its footnote b permits `Y < 2`; derived against the Lévy density it is the **`Y = 0`**
    (Variance Gamma) case exactly, and is off by 2.8% at `Y = 0.5`. The formula is right; the row's
    generality claim is not (thread 0417, finding 4). `C` cancels, which is why it is absent.
    """
    return (g**-2 + m**-2) / (
        1.0 / g - math.log(1.0 + 1.0 / g) - 1.0 / m - math.log(1.0 - 1.0 / m)
    )


def test_the_cumulant_exponent_matches_an_independent_transcription():
    # kappa_L(u) = sigma^2 u^2 / 2 + C[ln(G/(G+u)) + ln(M/(M-u))], read from the VG Levy density.
    for u in (-0.5, 0.25, 1.0, 1.5):
        expected = JUMPS.sigma**2 * u * u / 2.0 + JUMPS.c * (
            math.log(JUMPS.g / (JUMPS.g + u)) + math.log(JUMPS.m / (JUMPS.m - u))
        )
        assert levy_cumulant(JUMPS, u) == pytest.approx(expected, rel=1e-14, abs=0.0)


def test_the_variance_is_the_SECOND_CUMULANT_derived_not_restated():
    # var(L1) = kappa''(0), checked against a central difference of the cumulant exponent itself --
    # a different computation, not the same closed form twice.
    step = 1e-4
    numerical = (
        levy_cumulant(JUMPS, step) - 2.0 * levy_cumulant(JUMPS, 0.0) + levy_cumulant(JUMPS, -step)
    ) / (step * step)
    assert levy_variance(JUMPS) == pytest.approx(numerical, rel=1e-7, abs=0.0)
    # and the premise that this fixture's jumps actually carry the variance
    assert JUMPS.c * (JUMPS.g**-2 + JUMPS.m**-2) > 0.3 * levy_variance(JUMPS)


def test_the_multiplier_matches_TABLE_4_3s_cgmy_row_when_the_diffusion_is_off():
    pure_jump = LevyParams(sigma=0.0, c=JUMPS.c, g=JUMPS.g, m=JUMPS.m)
    assert levy_multiplier(pure_jump) == pytest.approx(
        _table_4_3_cgmy_multiplier(JUMPS.g, JUMPS.m), rel=1e-13, abs=0.0
    )
    # Table 4.3's own premise: the multiplier EXCEEDS 2 under jumps (Carr et al., negatively
    # skewed jump risk). Without this the row below is satisfied by a no-op.
    assert levy_multiplier(pure_jump) > 2.0


def test_the_multiplier_REDUCES_to_TABLE_4_3s_brownian_row_as_the_jumps_vanish():
    # The Brownian row of Table 4.3 is the integer 2 -- an exact reduction, not a limit.
    assert levy_multiplier(DIFFUSION) == pytest.approx(2.0, rel=1e-15, abs=0.0)
    # T39: the companion must be non-degenerate in the very parameter the reduction zeroes.
    assert abs(levy_multiplier(JUMPS) / 2.0 - 1.0) > 0.02


def test_the_multiplier_route_is_an_IDENTITY_so_it_buys_no_independence():
    """`Q_L·(κ(1) − κ′(0)) = κ″(0)` by construction — the same computation, not two routes.

    Asserted so the oracle set is not credited with independence it does not have (T70, and thread
    0417 finding 3). The continuous strike's independent check is the MC companion below.
    """
    for params in (JUMPS, DIFFUSION):
        gap = levy_cumulant(params, 1.0) - levy_mean(params)
        # exact, not approximately: the identity holds in the arithmetic, which is the claim
        assert levy_multiplier(params) * gap == pytest.approx(
            levy_variance(params), rel=1e-15, abs=0.0
        )


def test_the_MEAN_is_the_first_cumulant_derived_not_restated():
    # `levy_mean` is a closed form; this checks it against a different computation (a central
    # difference of the cumulant exponent), which is what the identity above cannot do.
    assert levy_mean(JUMPS) == pytest.approx(_mean(JUMPS), rel=1e-7, abs=0.0)
    assert levy_mean(DIFFUSION) == pytest.approx(0.0, abs=1e-15)


def _mean(params: LevyParams) -> float:
    """`E[L₁] = κ′(0)`, by central difference — the test's own reference."""
    step = 1e-6
    return (levy_cumulant(params, step) - levy_cumulant(params, -step)) / (2.0 * step)


# ============================================================================================
# 2. Prop 4.8 vs Prop 4.2 — the source contradicts itself; the choice is pinned, not deferred.
# ============================================================================================

def test_PROP_4_8s_PRINTED_form_disagrees_with_PROP_4_2_and_we_take_prop_4_2():
    """§4.4 contradicts itself and the discrepancy is load-bearing (§4's rule-which-is-right).

    Prop 4.8 prints `Kv(∞) = −(1/T)φ⁽²⁾(0)E[Z_T]`; under the book's own (4.84) that is `E[L₁²]`.
    Prop 4.2 + Prop 4.1 give `var(L₁)`. **We take `var(L₁)`**: quadratic variation is drift-free
    (`[L,L]` ignores the drift), Prop 4.1's own proof says `E[L,L]₁ = var(L₁)`, and the Brownian
    case agrees both ways — which is exactly why a centred fixture cannot see the difference.
    """
    second_moment = levy_variance(JUMPS) + levy_mean(JUMPS) ** 2
    # what we compute
    assert levy_fair_variance(_model().levy, _model().clock, YEARS) == pytest.approx(
        levy_variance(JUMPS), rel=1e-13, abs=0.0
    )
    # and the premise: this fixture SEPARATES the two readings, so the choice is being tested
    assert abs(second_moment / levy_variance(JUMPS) - 1.0) > 0.1


def test_a_CENTRED_levy_process_cannot_see_the_prop_4_8_discrepancy():
    """The degeneracy premise for the test above: at `E[L₁] = 0` the two readings COINCIDE.

    So a fixture built with a centred Lévy process would assert the choice vacuously — the
    catalogue's refrain, here on a source-discrepancy oracle (T13's shape).
    """
    # sigma-only is centred: E[L1] = 0 exactly.
    assert levy_mean(DIFFUSION) == 0.0
    assert levy_variance(DIFFUSION) + levy_mean(DIFFUSION) ** 2 == pytest.approx(
        levy_variance(DIFFUSION), rel=1e-12, abs=0.0
    )


# ============================================================================================
# 3. The clock — `var(ΔZ)` in closed form, its small-step order, and its STABILITY.
# ============================================================================================

def test_the_clock_increment_variance_matches_an_independent_transcription():
    # var(int_0^D v) = 2 s^2 (kD - 1 + e^{-kD})/k^2 with s^2 = vol_of_vol^2/(2k), from the
    # stationary CIR autocovariance cov(v_u,v_w) = s^2 e^{-k|u-w|}. Written here with the naive
    # bracket; production uses expm1, which is the point of the stability test below.
    for step in (0.1, 0.25, 1.0):
        s2 = CLOCK.vol_of_vol**2 / (2.0 * CLOCK.kappa)
        expected = (
            2.0 * s2 * (CLOCK.kappa * step - 1.0 + math.exp(-CLOCK.kappa * step)) / CLOCK.kappa**2
        )
        assert clock_increment_variance(CLOCK, step) == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_the_clock_increment_variance_is_ORDER_step_SQUARED_which_is_4_4_4s_condition():
    """§4.4.4 assumes `E[(ΔZ)²] ~ O(N⁻²)` — which an absolutely continuous clock satisfies.

    Measured as a RATE, not asserted: `var/D² → s²`, the stationary variance of the activity rate.
    A jump clock (a gamma subordinator) would give `O(D)` and break the convergence the whole
    tranche rests on, which is why the clock is an integrated rate and not a subordinator.
    """
    s2 = CLOCK.vol_of_vol**2 / (2.0 * CLOCK.kappa)
    ratios = [clock_increment_variance(CLOCK, step) / step**2 for step in (0.1, 0.01, 1e-3)]
    assert ratios == sorted(ratios), ratios          # monotone toward the limit
    assert ratios[-1] == pytest.approx(s2, rel=1e-3, abs=0.0)
    assert abs(ratios[0] / s2 - 1.0) > 0.05          # the premise: the coarse end is NOT converged


def _exact_decay_bracket(x: float) -> float:
    """`x − 1 + e^{−x}` at 60 significant digits, via `decimal` — the test's own reference.

    A genuinely independent high-precision reference for a cancelling expression (T20a's move,
    reached for with the standard library rather than scipy). Both of production's branches are
    measured against *this*, not against each other and not against a series that is itself the
    weaker side above a few thousandths — which is the trap that hid the branch below.
    """
    getcontext().prec = 60
    value = Decimal(repr(x))
    return float(value - 1 + (-value).exp())


@pytest.mark.parametrize(
    "x, tolerance",
    [
        (1e-9, 1e-14),       # deep in the series branch, where `expm1` itself is 1.5e-07 wrong
        (1e-6, 1e-14),
        (1e-4, 1e-14),
        # The point that discriminates the CROSSOVER itself: here the series is 70x more accurate
        # than `expm1` (2.78e-15 against 1.95e-13), so a tolerance between the two proves which
        # branch production actually took. Reverting the threshold to the first draft's 1e-3 sends
        # this point through `expm1` and fails -- which is how that mutation is killed.
        (1e-3, 2e-14),
        (1.9e-3, 1e-13),     # the last point BELOW the crossover
        (3e-3, 1e-13),       # the first point above it -- the `expm1` branch, naive is 6.1e-12
        (0.05, 1e-14),
        (1.0, 1e-15),
    ],
)
def test_the_decay_BRACKET_is_accurate_in_BOTH_branches_against_60_digits(x, tolerance):
    """The oracle must reach into **both** branches, which is what a first version did not.

    Every `κ` the first version probed put `x = κ·Δ` **below** the crossover, so the whole
    `expm1` branch was unexercised and substituting the naive cancelling form **survived the
    sweep**. The branch boundary is a guard like any other: it has to be tested on both sides
    (§4's reachability rule, here on an arithmetic branch rather than a refusal).
    """
    clock = ActivityRateParams(kappa=x / 0.25, vol_of_vol=0.5)
    stationary = 0.5**2 / (2.0 * clock.kappa)
    expected = 2.0 * stationary * _exact_decay_bracket(x) / (clock.kappa * clock.kappa)
    assert clock_increment_variance(clock, 0.25) == pytest.approx(expected, rel=tolerance, abs=0.0)


def test_the_NAIVE_bracket_would_FAIL_the_oracle_above_the_crossover():
    """The companion that proves the test above has power — the T15 shape.

    Asserting that production is accurate says nothing unless the rejected form is shown to be
    **in**accurate at the same point. The naive bracket is 6.1e-12 off at `x = 3e-3` (200× worse),
    8.9e-05 off at `1e-6`, and **100% wrong** by `1e-9` where it returns a value of the wrong
    order entirely.
    """
    for x, floor in ((3e-3, 1e-12), (1e-6, 1e-5), (1e-9, 0.5)):
        naive = x - 1.0 + math.exp(-x)
        exact = _exact_decay_bracket(x)
        assert abs(naive / exact - 1.0) > floor, (x, naive, exact)
    # and at 1e-9 it is not merely inaccurate -- it has lost every significant digit
    assert 1e-9 - 1.0 + math.exp(-1e-9) == 0.0


def test_the_CROSSOVER_sits_where_the_two_errors_actually_cross():
    """`2e-3`, measured — and a first draft had `1e-3`, where the series is 70× better.

    **This is the PREMISE, not the oracle, and the distinction cost a mutation survivor.** A first
    version of this test compared the two *forms* to each other and never called production at
    all, so it held whatever threshold the code used — a tautology with respect to the code under
    test. It is kept because it documents *why* the boundary is where it is; what pins production
    to it is the `1e-3` row of the parametrised test above, where the two forms are separable.
    """
    for x in (1e-4, 1e-3):
        exact = _exact_decay_bracket(x)
        series = x * x * (1 / 2 + x * (-1 / 6 + x * (1 / 24 - x / 120)))
        assert abs(series / exact - 1.0) < abs((x + math.expm1(-x)) / exact - 1.0)
    for x in (3e-3, 1e-2):
        exact = _exact_decay_bracket(x)
        series = x * x * (1 / 2 + x * (-1 / 6 + x * (1 / 24 - x / 120)))
        assert abs((x + math.expm1(-x)) / exact - 1.0) < abs(series / exact - 1.0)
    # and production agrees with the BETTER form on each side -- the part that touches the code
    for x, use_series in ((1e-3, True), (3e-3, False)):
        better = (
            x * x * (1 / 2 + x * (-1 / 6 + x * (1 / 24 - x / 120)))
            if use_series
            else x + math.expm1(-x)
        )
        clock = ActivityRateParams(kappa=x / 0.25, vol_of_vol=0.5)
        stationary = 0.5**2 / (2.0 * clock.kappa)
        assert clock_increment_variance(clock, 0.25) == pytest.approx(
            2.0 * stationary * better / (clock.kappa * clock.kappa), rel=1e-15, abs=0.0
        )


def test_a_deterministic_clock_has_ZERO_increment_variance():
    assert clock_increment_variance(FLAT_CLOCK, 0.25) == 0.0


# ============================================================================================
# 4. The discrete adjustment (Prop 4.3) — the spec's anchor corrected, and the O(1/N) rate.
# ============================================================================================

def test_the_PURE_DIFFUSION_adjustment_is_an_EXACT_CLOSED_FORM_not_zero():
    """**The spec's oracle 2 is wrong and this is the corrected anchor** (thread 0417, finding 1).

    With no jumps and a deterministic clock the adjustment is `(r − q − σ²/2)²·T/N` **exactly** —
    the classic discrete-variance result — because Prop 4.3's first term `(1/T)Σ(E[Δln S])²`
    survives the vanishing of the jumps entirely. Exact, not a limit: asserted at `rel=1e-14`.
    """
    drift = RATE - DIVIDEND - DIFFUSION.sigma**2 / 2.0
    for count in (4, 12, 52, 252):
        expected = drift * drift * YEARS / count
        assert discrete_monitoring_adjustment(
            DIFFUSION, FLAT_CLOCK, YEARS, count, RATE - DIVIDEND
        ) == pytest.approx(expected, rel=1e-14, abs=0.0)
    # The premise that makes this an anchor rather than a tautology: it is NOT zero.
    assert discrete_monitoring_adjustment(DIFFUSION, FLAT_CLOCK, YEARS, 12, RATE - DIVIDEND) > 1e-7


def test_the_adjustment_with_GENUINE_JUMPS_is_the_T39_companion():
    # The reduction above zeroes BOTH the jumps and the clock; the companion must exercise both,
    # or the anchor is blind to exactly the terms the tranche is about.
    reduced = discrete_monitoring_adjustment(DIFFUSION, FLAT_CLOCK, YEARS, 52, RATE - DIVIDEND)
    full = discrete_monitoring_adjustment(JUMPS, CLOCK, YEARS, 52, RATE - DIVIDEND)
    assert full > 10.0 * reduced, (full, reduced)


def test_the_adjustment_is_POSITIVE_which_is_4_4_2s_printed_requirement():
    """§4.4.2: *"the fair strike of the discrete variance swap is always greater than that of its
    continuous counterpart since the correction term is seen to be always positive."*

    A requirement oracle — it survives a transcription error in either term, since both of Prop
    4.3's terms are manifestly non-negative (a square and a variance).
    """
    for levy in (JUMPS, DIFFUSION):
        for clock in (CLOCK, FLAT_CLOCK):
            for count in (4, 52, 252):
                assert discrete_monitoring_adjustment(levy, clock, YEARS, count, RATE - DIVIDEND) > 0.0


def test_the_discrete_strike_converges_to_the_continuous_one_at_RATE_one_over_N():
    """§4.4.4 / Prop 4.8: `Kv(N) − Kv(∞) ~ O(N⁻¹)`, asserted as a RATE (T25), not a flat band.

    The sharp form: `adjustment × N` tends to a constant. A PLATEAU instead of convergence is the
    detector's red — which is what a quadrature or reference error would produce.
    """
    scaled = [
        discrete_monitoring_adjustment(JUMPS, CLOCK, YEARS, count, RATE - DIVIDEND) * count
        for count in (50, 100, 200, 400, 800, 1600)
    ]
    # successive ratios -> 1 (O(1/N)); a surviving O(1) term would hold them away from 1
    for coarse, fine in zip(scaled, scaled[1:]):
        assert fine / coarse == pytest.approx(1.0, rel=0.05)
    assert scaled[-1] / scaled[0] == pytest.approx(1.0, rel=0.02)
    # the premise: the adjustment itself moves by a factor of ~32 over that range, so "constant
    # x N" is a real claim about the rate and not an artefact of everything being tiny
    raw = discrete_monitoring_adjustment(JUMPS, CLOCK, YEARS, 50, RATE - DIVIDEND)
    assert raw / discrete_monitoring_adjustment(JUMPS, CLOCK, YEARS, 1600, RATE - DIVIDEND) > 25.0


def test_the_CLOCK_term_contributes_at_the_same_order_and_is_not_dead_code():
    # var(dZ) is O(D^2) so the clock term is O(1/N) too -- same order as the squared-mean term.
    # Without this, a fixture with a deterministic clock would leave Prop 4.3's second term
    # unexercised (the duplicated-guard shape of the degeneracy catalogue, one term over).
    with_clock = discrete_monitoring_adjustment(JUMPS, CLOCK, YEARS, 52, RATE - DIVIDEND)
    without = discrete_monitoring_adjustment(JUMPS, FLAT_CLOCK, YEARS, 52, RATE - DIVIDEND)
    assert with_clock > without
    assert abs(with_clock / without - 1.0) > 0.1, (with_clock, without)


# ============================================================================================
# 5. The MC companion — the only INDEPENDENT check of the continuous strike (see §1's identity).
# ============================================================================================

def _simulate_realized_variance(
    levy: LevyParams,
    clock: ActivityRateParams,
    years: float,
    count: int,
    paths: int,
    seed: int,
    sub_steps: int = 20,
) -> tuple[float, float]:
    """Discrete realized variance from EXACT Lévy increments under a simulated clock.

    The Lévy increments carry **no discretisation error**: a `Y = 0` CGMY process is the difference
    of two gamma processes (`Γ(Ct, 1/M) − Γ(Ct, 1/G)`), so given the elapsed clock time the
    increment is drawn exactly. Only the clock integral is discretised, which is why the activity
    rate is sub-stepped. Returns `(mean, standard error)`.
    """
    rng = numpy.random.default_rng(seed)
    step = years / count
    dt = step / sub_steps
    drift = RATE - DIVIDEND
    log_moment = levy_cumulant(levy, 1.0)

    rate = numpy.full(paths, 1.0)
    if clock.vol_of_vol > 0.0:                    # burn in to the STATIONARY law (relaxation 1/kappa)
        burn_steps, burn_dt = 400, 10.0 / clock.kappa / 400
        for _ in range(burn_steps):
            rate = numpy.abs(
                rate
                + clock.kappa * (1.0 - rate) * burn_dt
                + clock.vol_of_vol * numpy.sqrt(numpy.maximum(rate, 0.0) * burn_dt)
                * rng.standard_normal(paths)
            )

    total = numpy.zeros(paths)
    for _ in range(count):
        elapsed = numpy.zeros(paths)
        for _ in range(sub_steps):
            elapsed += rate * dt
            if clock.vol_of_vol > 0.0:
                rate = numpy.abs(
                    rate
                    + clock.kappa * (1.0 - rate) * dt
                    + clock.vol_of_vol * numpy.sqrt(numpy.maximum(rate, 0.0) * dt)
                    * rng.standard_normal(paths)
                )
        shape = levy.c * elapsed
        jump = numpy.zeros(paths)
        if levy.c > 0.0:
            jump = rng.gamma(shape, 1.0 / levy.m) - rng.gamma(shape, 1.0 / levy.g)
        diffusion = levy.sigma * numpy.sqrt(elapsed) * rng.standard_normal(paths)
        log_return = drift * step + jump + diffusion - log_moment * elapsed
        total += log_return * log_return

    realized = total / years
    return float(realized.mean()), float(realized.std(ddof=1) / math.sqrt(paths))


def test_the_closed_form_discrete_strike_matches_SIMULATION():
    """The genuinely independent check — the identity in §1 means the closed forms do not provide one.

    Banded `4·SE + the clock's own discretisation error`, the two measured separately rather than
    lumped into one percentage (§4): the sub-stepped activity rate is the only approximation on the
    simulation side, and it is measured by halving the sub-steps.
    """
    count, paths = 12, 60_000
    predicted = levy_fair_variance(JUMPS, CLOCK, YEARS) + discrete_monitoring_adjustment(
        JUMPS, CLOCK, YEARS, count, RATE - DIVIDEND
    )
    mean, standard_error = _simulate_realized_variance(JUMPS, CLOCK, YEARS, count, paths, seed=11)
    coarse, _ = _simulate_realized_variance(
        JUMPS, CLOCK, YEARS, count, paths, seed=11, sub_steps=10
    )
    clock_error = abs(mean - coarse)
    assert abs(mean - predicted) < 4.0 * standard_error + clock_error, (
        mean, predicted, standard_error, clock_error
    )
    # resolution: state what this band actually rules out at this path count
    assert 4.0 * standard_error / predicted < 0.05


def test_simulation_RESOLVES_the_adjustment_it_is_asked_to_confirm():
    """The band-resolution premise (§4): at a coarse `N` the adjustment must exceed the noise.

    Otherwise "discrete ≈ continuous" asserts only that both are noisy — the spec's own degeneracy
    premise, and the reason the fixture uses `N = 4` here rather than the `N = 12` above.
    """
    count, paths = 4, 250_000
    adjustment = discrete_monitoring_adjustment(JUMPS, CLOCK, YEARS, count, RATE - DIVIDEND)
    mean, standard_error = _simulate_realized_variance(JUMPS, CLOCK, YEARS, count, paths, seed=5)
    # Measured: 3.86 SE here. At 60k paths it is 1.92 SE -- NOT resolvable, so the path count is
    # part of the claim and not a default (the band-resolution rule: state what a green rules out).
    assert adjustment > 3.0 * standard_error, (adjustment, standard_error)
    continuous = levy_fair_variance(JUMPS, CLOCK, YEARS)
    assert abs(mean - continuous - adjustment) < abs(mean - continuous)


# ============================================================================================
# 6. The engines.
# ============================================================================================

def _swap(strike: float = 0.05) -> VarianceSwap:
    return VarianceSwap(
        asset=ACME,
        expiry=EXPIRY,
        variance_strike=strike,
        variance_notional=Money(1_000_000.0, USD),
    )


def _discrete_swap(strike: float = 0.05, count: int = 52) -> DiscreteVarianceSwap:
    return DiscreteVarianceSwap(
        asset=ACME,
        expiry=EXPIRY,
        variance_strike=strike,
        variance_notional=Money(1_000_000.0, USD),
        monitoring_count=count,
    )


def test_the_continuous_swap_prices_through_the_registry():
    result = price(_swap(), _model())
    assert isinstance(result, PricingResult)
    market = _model().market
    years = market.curves.discount(USD).time_measure.year_fraction(EXPIRY)
    fair = levy_fair_variance(JUMPS, CLOCK, years)
    expected = market.curves.discount(USD).df(EXPIRY) * 1_000_000.0 * (fair - 0.05)
    assert result.pv.amount == pytest.approx(expected, rel=1e-13, abs=0.0)


def test_the_DISCRETE_swap_is_the_continuous_one_PLUS_the_adjustment():
    model = _model()
    years = model.market.curves.discount(USD).time_measure.year_fraction(EXPIRY)
    continuous = price(_swap(), model)
    discrete = price(_discrete_swap(), model)
    assert isinstance(continuous, PricingResult) and isinstance(discrete, PricingResult)
    discount = model.market.curves.discount(USD).df(EXPIRY)
    adjustment = discrete_monitoring_adjustment(JUMPS, CLOCK, years, 52, RATE - DIVIDEND)
    assert discrete.pv.amount - continuous.pv.amount == pytest.approx(
        discount * 1_000_000.0 * adjustment, rel=1e-12, abs=0.0
    )
    # §4.4.2's requirement, at the level of the PRICE: discrete is worth more to the payer.
    assert discrete.pv.amount > continuous.pv.amount


def test_the_discrete_swap_converges_to_the_continuous_PRICE_as_monitoring_densifies():
    model = _model()
    continuous = price(_swap(), model)
    assert isinstance(continuous, PricingResult)
    gaps = []
    for count in (4, 52, 2520):
        result = price(_discrete_swap(count=count), model)
        assert isinstance(result, PricingResult)
        gaps.append(abs(result.pv.amount - continuous.pv.amount))
    assert gaps == sorted(gaps, reverse=True), gaps
    assert gaps[-1] < gaps[0] / 100.0


def test_a_MATURED_swap_is_PV_ZERO_under_invariant_6():
    # The fixture STRADDLES: the valuation date sits after the expiry, so the filter is load-bearing
    # (the cross-tranche fixture-convention degeneracy, tenth shape -- asserted, not believed).
    after = EXPIRY + timedelta(days=30)
    late = MarketSnapshot(
        valuation_date=after,
        curves=CurveSet(
            {
                CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(
                    after, RATE, DayCountConvention.ACT_365_FIXED
                )
            }
        ),
        equities={ACME: EquityUnderlying(spot=100.0, repo_rate=DIVIDEND)},
    )
    model = TimeChangedLevyModel(market=late, levy=JUMPS, clock=CLOCK)
    for product in (_swap(), _discrete_swap()):
        result = price(product, model)
        assert isinstance(result, PricingResult)
        assert result.pv.amount == 0.0
    # the premise: with a LIVE valuation date the same swaps are worth something non-zero
    assert abs(price(_swap(), _model()).pv.amount) > 1.0


def test_a_missing_curve_is_a_PRICING_FAILURE_not_an_exception():
    bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
    model = TimeChangedLevyModel(market=bare, levy=JUMPS, clock=CLOCK)
    for product in (_swap(), _discrete_swap()):
        assert isinstance(price(product, model), PricingFailure)


# ============================================================================================
# 7. Refusals — supplying the input each guard exists to reject (T33b).
# ============================================================================================

@pytest.mark.parametrize(
    "sigma, c, g, m",
    [
        (-0.1, 2.0, 5.0, 10.0),      # a negative diffusion volatility
        (0.2, -1.0, 5.0, 10.0),      # a negative jump arrival rate
        (0.2, 2.0, 0.0, 10.0),       # G must be positive
        (0.2, 2.0, 5.0, 1.0),        # M > 1, or kappa_L(1) diverges (Table 4.3 footnote b)
        (0.2, 2.0, 5.0, 0.5),
    ],
)
def test_the_levy_parameters_REFUSE_values_outside_their_domain(sigma, c, g, m):
    with pytest.raises(ValueError):
        LevyParams(sigma=sigma, c=c, g=g, m=m)


def test_M_GREATER_THAN_ONE_is_exactly_the_condition_for_a_FINITE_log_moment():
    """The guard is not arbitrary: `κ_L(1)` contains `ln(M/(M−1))`, which diverges at `M = 1`.

    Asserted rather than asserted-by-fiat — the refusal's reason is checked by watching the
    quantity blow up as the boundary is approached.
    """
    moments = [
        levy_cumulant(LevyParams(0.2, 2.0, 5.0, m), 1.0)
        for m in (1.5, 1.1, 1.001, 1.0 + 1e-9, 1.0 + 1e-15)
    ]
    assert moments == sorted(moments), moments
    # it diverges LOGARITHMICALLY (`C·ln(M/(M−1))`), so the right claim is unboundedness over a
    # wide range of M, not a growth factor over a narrow one -- 1e-15 from the boundary is 20x 1.5's
    assert moments[-1] > 20.0 * moments[0]
    # and the RATE is the log -- the exact gap, not an approximation of it: the jump part of
    # `kappa_L(1)` is `C*ln(M/(M-1))`, so moving M from 1.001 to 1+1e-9 adds exactly this.
    near, far = 1.001, 1.0 + 1e-9
    exact_gap = 2.0 * (
        math.log(far / (far - 1.0)) - math.log(near / (near - 1.0))
    )  # `far - 1.0` DERIVED, not the literal 1e-9: `1.0 + 1e-9` is not representable, so the
    # value the code sees is 1.0000000827e-09 and a literal reference would be 6e-9 off (F-7.1)
    assert moments[-2] - moments[-3] == pytest.approx(exact_gap, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("kappa, vol_of_vol", [(0.0, 0.5), (-1.0, 0.5), (2.0, -0.1)])
def test_the_clock_parameters_REFUSE_values_outside_their_domain(kappa, vol_of_vol):
    with pytest.raises(ValueError):
        ActivityRateParams(kappa=kappa, vol_of_vol=vol_of_vol)


@pytest.mark.parametrize("count", [0, -1])
def test_a_discrete_swap_REFUSES_a_non_positive_monitoring_count(count):
    with pytest.raises(ValueError):
        _discrete_swap(count=count)


def test_the_fair_variance_REFUSES_a_non_positive_horizon():
    with pytest.raises(ValueError):
        levy_fair_variance(JUMPS, CLOCK, 0.0)
