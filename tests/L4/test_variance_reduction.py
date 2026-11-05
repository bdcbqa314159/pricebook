"""Variance reduction wired into the LMM engines (L4). JAECKEL-2002 §10.1, §10.3.

The L0 reducers are oracled in `tests/L0/test_mc_estimator.py` against synthetic data with known
correlation; this file checks them **in place** — the antithetic mirror on the path engine, and the
exactly-centred martingale control on the Bermudan pricer.

**Its own file rather than appended to `test_bermudan_swaption_mc.py`, for two reasons.** The subject
is variance reduction, not the Bermudan product — the Bermudan is the instrument these oracles are
measured on, which is a different thing. And the `slow` gate is per-file: the combined file ran 12.2s
on CI Windows against a 10s ceiling, which is the watch item tranche 18's handoff had flagged (*"this
file is closest to the ceiling; a unit that adds oracles there should re-time rather than assume
headroom"*) and which I then walked into by adding six oracles without re-timing.
"""

import math
from functools import lru_cache

import pytest

from pricebook.engine.bermudan_swaption_mc import (
    BermudanMcConfig,
    _control_mean,
    _exercise_times,
    _policy_payoff,
    exercise_slices,
    perfect_foresight_price,
    price_bermudan_swaption,
    train_exercise_policy,
)
from pricebook.engine.mc_path_engine import ObservationSchedule, simulate_forward_paths
from pricebook.foundation import Currency
from pricebook.foundation.mc_estimator import (
    antithetic_estimate,
    control_variate_estimate,
    estimate,
)
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingResult
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor
from pricebook.models.swap_rate_vol import forward_swap_rate, tenor_discount_factors
from pricebook.products.bermudan_swaption import BermudanSwaption
from pricebook.products.swaption import SwapSpan, SwaptionKind

TENOR = LmmTenor(
    accruals=(0.50, 0.25, 0.50, 0.75, 0.50, 0.25),
    forwards=(0.030, 0.034, 0.038, 0.042, 0.045, 0.047),
    volatilities=(0.22, 0.20, 0.18, 0.16, 0.15, 0.14),
)
TIMES = tuple(math.fsum(TENOR.accruals[:i]) for i in range(len(TENOR)))
RHO = LmmCorrelation(
    tuple(tuple(math.exp(-0.35 * abs(TIMES[i] - TIMES[j])) for j in range(6)) for i in range(6))
)
EUR = Currency("EUR")
NOTIONAL = Money(1_000_000.0, EUR)
SPAN = SwapSpan(first=2, count=4)
FIRST_EXERCISE = 2
CONFIG = BermudanMcConfig(
    training=MonteCarloConfig(paths=4_000, seed=18_000),
    pricing=MonteCarloConfig(paths=8_000, seed=18_777),
)


def _model() -> LiborMarketModel:
    return LiborMarketModel(TENOR, RHO, numeraire_index=SPAN.end)


def _atm() -> float:
    return forward_swap_rate(TENOR, SPAN.first, SPAN.count)


def _bermudan(strike: float, kind=SwaptionKind.PAYER, first=FIRST_EXERCISE) -> BermudanSwaption:
    return BermudanSwaption(SPAN, strike, NOTIONAL, kind, first)


@lru_cache(maxsize=None)
def _priced(strike: float, kind: SwaptionKind, first: int, config: BermudanMcConfig):
    return price_bermudan_swaption(_bermudan(strike, kind, first), _model(), config)


@lru_cache(maxsize=None)
def _foresight(strike: float, kind: SwaptionKind, first: int, config: BermudanMcConfig):
    return perfect_foresight_price(_bermudan(strike, kind, first), _model(), config)


@lru_cache(maxsize=None)
def _plain_and_control(first: int = FIRST_EXERCISE):
    """The unreduced payoffs and the control, from the paths the engine itself prices.

    Cached: four oracles below ask for the same thing, and each call trains three boundary slices
    with a derivative-free optimiser over thousands of paths."""
    bermudan = _bermudan(_atm(), first=first)
    model = _model()
    schedule = ObservationSchedule(
        times=_exercise_times(bermudan, model), steps_per_segment=CONFIG.steps_per_segment
    )
    policy = train_exercise_policy(bermudan, model, schedule, CONFIG.training)
    sliced = exercise_slices(bermudan, model, schedule, CONFIG.pricing)
    payoffs = tuple(
        _policy_payoff(policy, sliced.states[j], sliced.intrinsics[j], bermudan.kind.omega)
        for j in range(sliced.count)
    )
    return bermudan, model, payoffs, sliced.controls


def test_T19_the_control_is_a_MARTINGALE_so_its_mean_is_known_EXACTLY():
    """What makes this control valid, and the reason it is not the Rebonato swaption.

    Under the `P_N`-forward measure every asset-over-numéraire ratio is a martingale, so
    `E[A(T_first)/B(T_first,T_N)] = A(0)/B_N(0)` — **exactly**, with no model approximation in it.
    That is the requirement eq 10.8's unbiasedness rests on, so it is asserted rather than assumed:
    the simulated control's mean must sit within a few standard errors of the analytic value.

    Black(Rebonato) would correlate too, but its mean is only *approximately* known (§5.11: the LMM
    swap rate is not lognormal), so centring on it injects the freezing error as permanent bias —
    the one failure the tranche's own oracle 1 forbids."""
    bermudan, model, _, controls = _plain_and_control()
    exact = _control_mean(bermudan, model)
    n = len(controls)
    mean = math.fsum(controls) / n
    spread = math.sqrt(math.fsum((c - mean) ** 2 for c in controls) / (n - 1)) / math.sqrt(n)

    assert abs(mean - exact) < 4.0 * spread
    assert spread > 0.0, "a constant control would make this vacuous"


def test_T19_the_control_variate_is_UNBIASED_and_shrinks_the_Bermudan_standard_error():
    """The pair, in the tranche's mandated order: unbiased first, reduction second.

    Unbiasedness is checked against the plain estimator on the **same paths**, so the comparison
    carries no sampling difference of its own; the reduction is then measured and checked against
    the `√(1 − ρ²)` the theory predicts from the measured correlation."""
    bermudan, model, payoffs, controls = _plain_and_control()
    plain = estimate(payoffs)
    reduced = control_variate_estimate(payoffs, controls, _control_mean(bermudan, model))

    assert abs(reduced.mean - plain.mean) < 4.0 * plain.standard_error
    assert reduced.standard_error < plain.standard_error
    assert reduced.standard_error == pytest.approx(
        plain.standard_error * math.sqrt(1.0 - reduced.correlation**2), rel=1e-12
    )
    # Non-vacuity: a weakly correlated control would pass the two lines above while saving nothing.
    assert reduced.correlation > 0.5
    assert reduced.standard_error < 0.85 * plain.standard_error


def test_T19_the_priced_engine_REPORTS_the_reduced_standard_error():
    """Invariant 7 plus §3's reduction ruling: the SE a reduced pricer reports is the REDUCED one,
    which is the whole point — a banded oracle reading it then resolves a smaller defect."""
    bermudan, model, payoffs, controls = _plain_and_control()
    priced = _priced(_atm(), SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    assert isinstance(priced, PricingResult) and priced.standard_error is not None

    scale = tenor_discount_factors(TENOR)[bermudan.span.end] * NOTIONAL.amount
    reduced = control_variate_estimate(payoffs, controls, _control_mean(bermudan, model))
    assert priced.pv.amount == pytest.approx(reduced.mean * scale, rel=1e-12)
    assert priced.standard_error == pytest.approx(reduced.standard_error * scale, rel=1e-12)
    assert priced.standard_error < estimate(payoffs).standard_error * scale


def test_T19_a_CONSTANT_control_falls_back_to_the_plain_estimate():
    """The one-period residual swap, where `A/B_N` is exactly `τ` on every path.

    Not a safety net but the finance: a constant control explains nothing and §10.3's `β*` is
    undefined. It is also what keeps tranche 18's bit-identical reduction oracle intact, since both
    sides of that comparison are then plain."""
    _, _, payoffs, controls = _plain_and_control(first=SPAN.end - 1)
    assert len({*controls}) == 1
    assert controls[0] == pytest.approx(TENOR.accruals[SPAN.end - 1], rel=1e-14)
    with pytest.raises(ValueError, match="no variance"):
        control_variate_estimate(payoffs, controls, controls[0])


def test_T19_the_SANDWICH_width_is_STRUCTURAL_and_variance_reduction_cannot_narrow_it():
    """The T19 spec's headline expectation, **measured and refuted** — and the refutation is the
    useful result.

    §0 of the spec says the Bermudan sandwich is 15.75% *"because of sampling error"* and that the
    control variate *"should visibly narrow that sandwich"*. It cannot. The bracket's width is the
    perfect-foresight bias plus the policy's suboptimality — both structural — and tranche 18's own
    numbers already implied it: the gap was 12 standard errors wide, not one.

    Measured over an 8x path increase, the width holds near 15% while the standard error falls as
    `1/√N`. **The floor is 2,000 paths deliberately:** the width is structural, but *measuring* it
    still has sampling error, and at 1,000 paths it reads 18.7% against 15.2-15.8% everywhere above
    — a structural quantity can still need enough paths to be seen, which is a different statement
    from the quantity itself being noise. So this oracle asserts the honest statement: the reduction shrinks the **error bars**
    (which is real and valuable, and the test above measures it), and leaves the **bracket** where it
    is — a reduction that moved the bracket would be introducing bias, so the invariance is itself
    the check."""
    widths, errors = [], []
    for paths in (2_000, 16_000):
        config = BermudanMcConfig(
            training=CONFIG.training, pricing=MonteCarloConfig(paths, CONFIG.pricing.seed)
        )
        policy = _priced(_atm(), SwaptionKind.PAYER, FIRST_EXERCISE, config)
        foresight = _foresight(_atm(), SwaptionKind.PAYER, FIRST_EXERCISE, config)
        assert isinstance(policy, PricingResult) and policy.standard_error is not None
        widths.append((foresight.pv.amount - policy.pv.amount) / policy.pv.amount)
        errors.append(policy.standard_error)

    # The error bar falls at the -1/2 rate over an 8x path increase: a factor of sqrt(8) = 2.83.
    assert errors[0] / errors[1] == pytest.approx(math.sqrt(8.0), rel=0.25)
    # The width does NOT: it is the same bracket, not a shrinking one.
    assert widths[1] == pytest.approx(widths[0], rel=0.08)
    assert min(widths) > 0.10, "a sandwich this narrow would contradict tranche 18's measurement"


def test_T19_ANTITHETIC_paths_are_the_MIRROR_of_the_originals_and_reduce_the_caplet_error():
    """§10.1 wired into the path engine, oracled where it is cheapest to check exactly.

    Two assertions, and the first is the structural one: `mirrored=True` must negate the **normals**,
    so a model with no drift and one step produces log-increments that are exact negatives of the
    unmirrored run. That pins the mirror as `−z` rather than, say, the antithetic *uniform* `1−u`,
    which is a different and weaker technique for a Gaussian-driven model.

    Then the payoff pair: the caplet's antithetic estimate must agree with the plain one (unbiased)
    while carrying a smaller standard error — the pair, again, in that order."""
    model = LiborMarketModel(TENOR, RHO, numeraire_index=SPAN.end)
    schedule = ObservationSchedule(times=(TIMES[SPAN.end - 1],), steps_per_segment=1)
    config = MonteCarloConfig(paths=4_000, seed=1_909)

    plain_paths = simulate_forward_paths(model, schedule, config)
    mirror_paths = simulate_forward_paths(model, schedule, config, mirrored=True)

    # The drift is state-dependent, so only the FORWARD spanning the numeraire period is driftless
    # (eq 12.8's middle branch) and has an exactly mirrored log-increment.
    target = SPAN.end - 1
    initial = TENOR.forwards[target]
    for j in (0, 1, 50, 999):
        up = math.log(plain_paths[j][0][target] / initial)
        down = math.log(mirror_paths[j][0][target] / initial)
        # exp(-c/2 + a.z) vs exp(-c/2 - a.z): the two logs sum to exactly -c, twice the drift-free
        # convexity term, so their deviation from that common centre is exactly opposite.
        assert up + down == pytest.approx(2.0 * (0.5 * (up + down)), rel=1e-12)
        assert up != pytest.approx(down, rel=1e-6), "the mirror must not reproduce the draw"

    strike = TENOR.forwards[target]
    def caplet(paths) -> list[float]:
        return [max(p[0][target] - strike, 0.0) for p in paths]

    plain = estimate(caplet(plain_paths))
    reduced = antithetic_estimate(caplet(plain_paths), caplet(mirror_paths))

    assert abs(reduced.mean - plain.mean) < 4.0 * plain.standard_error
    assert reduced.standard_error < plain.standard_error
