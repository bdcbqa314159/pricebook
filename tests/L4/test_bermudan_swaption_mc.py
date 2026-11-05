"""The Bermudan swaption under the LMM by Monte Carlo (L4). JAECKEL-2002 §12.6-12.8.

Jäckel's headline instrument, and the one with **no closed form and no tree here** — so it is priced
inside a *sandwich* rather than reported as a single number:

    trained policy on FRESH paths   <=   true Bermudan   <=   perfect foresight on the same paths

The lower bound is a real, achievable exercise strategy, so it under-prices; the upper bound exercises
with hindsight, so it over-prices. **The bracket's width IS this tranche's resolution** — what the
oracles can and cannot rule out — and it is stated rather than assumed tight, per §4's banded-
resolution rule. A tight sandwich says the parametrisation is adequate for the trade; a loose one says
it is not, which is the diagnostic job Jäckel's non-recombining tree did for him (§0 of the spec
rules the tree out of scope, so the sandwich has to carry it).

**Train on one path set, price on a fresh one.** Jäckel: *"The final value U_0(p_0) gives then an
estimate of the value of the Bermudan swaption with a slight upward bias. Therefore, we finally re-run
the simulation with a new set of N_Sampling yield curve evolutions."* Optimising and pricing on the
same paths lets the policy peek at the realisations it is scored on; the split is a correctness
requirement, and oracle 5 asserts the bias is real rather than trusting the instruction.
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
from pricebook.engine.mc_path_engine import (
    ObservationSchedule,
    PathGrid,
    simulate_forward_paths,
)
from pricebook.foundation.mc_estimator import (
    antithetic_estimate,
    control_variate_estimate,
    estimate,
)
from pricebook.models.exercise_boundary import (
    BoundaryParams,
    BoundaryState,
    TrainingSlice,
    roll_continuations,
    train_slice,
)
from pricebook.engine.swaption_mc import price_european_swaption
from pricebook.foundation import Currency
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor
from pricebook.models.swap_rate_vol import forward_swap_rate, tenor_discount_factors
from pricebook.products.bermudan_swaption import BermudanSwaption
from pricebook.products.swaption import EuropeanSwaption, SwapSpan, SwaptionKind

# Non-uniform accruals and a correlation decaying in the actual TIMES, not index distance — the
# tranche-17 lesson: a translation-invariant ρ and a flat τ both hide a slice offset.
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

#: The coterminal swap runs to the tenor's end; exercise opportunities start at index 2 — a
#: "6-non-call-2" in Jäckel's notation, his own worked example's shape.
SPAN = SwapSpan(first=2, count=4)
FIRST_EXERCISE = 2
CONFIG = BermudanMcConfig(
    training=MonteCarloConfig(paths=4_000, seed=18_000),
    pricing=MonteCarloConfig(paths=8_000, seed=18_777),
)


def _model() -> LiborMarketModel:
    return LiborMarketModel(TENOR, RHO, numeraire_index=SPAN.end)


def _bermudan(strike: float, kind=SwaptionKind.PAYER, first=FIRST_EXERCISE) -> BermudanSwaption:
    return BermudanSwaption(SPAN, strike, NOTIONAL, kind, first)


def _atm() -> float:
    return forward_swap_rate(TENOR, SPAN.first, SPAN.count)


# **Prices are CACHED, and the `slow` gate is why.** `price_bermudan_swaption` is a deterministic
# function of its arguments, so caching changes nothing about what is asserted — while six of the
# oracles below ask for the SAME priced Bermudan (the ATM payer at the full exercise schedule), each
# of which retrains three slices with a derivative-free optimiser over thousands of paths. Caching
# took the file from 2.35s to well under half that, against a ceiling that has to hold on the slowest
# CI runner. Path counts are NOT reduced: every oracle here states its margin at the committed count.
@lru_cache(maxsize=None)
def _priced(strike: float, kind: SwaptionKind, first: int, config: BermudanMcConfig):
    return price_bermudan_swaption(_bermudan(strike, kind, first), _model(), config)


@lru_cache(maxsize=None)
def _foresight(strike: float, kind: SwaptionKind, first: int, config: BermudanMcConfig):
    return perfect_foresight_price(_bermudan(strike, kind, first), _model(), config)


# ============================================================================================
# ORACLE 1 — THE EXACT REDUCTION: a ONE-exercise-date Bermudan IS the T17 European swaption.
# ============================================================================================


@pytest.mark.parametrize("kind", [SwaptionKind.PAYER, SwaptionKind.RECEIVER])
def test_O1_a_SINGLE_exercise_date_Bermudan_equals_the_T17_EUROPEAN(kind):
    """The anchor, and it inherits tranche 17's exact caplet->Black anchor beneath it.

    With only the last exercise date available, eq 12.39 pins the policy exactly — exercise iff the
    residual swap is in the money — and that is precisely the European payoff. So this must agree
    with `price_european_swaption` to Monte Carlo error on a shared seed, and any error in the
    exercise machinery that survives into the single-date case shows up here."""
    strike = _atm()
    bermudan = _bermudan(strike, kind, first=SPAN.end - 1)
    european = EuropeanSwaption(SwapSpan(SPAN.end - 1, 1), strike, NOTIONAL, kind)

    priced = _priced(strike, kind, SPAN.end - 1, CONFIG)
    assert isinstance(priced, PricingResult)
    assert priced.standard_error is not None

    reference = price_european_swaption(
        european,
        LiborMarketModel(TENOR, RHO, numeraire_index=SPAN.end),
        PathGrid(horizon=TIMES[SPAN.end - 1], steps=CONFIG.steps_per_segment),
        CONFIG.pricing,
    )
    assert isinstance(reference, PricingResult)
    assert reference.standard_error is not None

    # **BIT-IDENTICAL, not banded** — and the first version of this oracle got that wrong.
    # It compared inside `4 x combined SE`, which at the committed path counts resolves only 8-11%
    # of the price; a mutation dropping the annuity's numeraire division (worth ~1.2%) survived
    # inside it. The comparison can be exact: with one exercise date the Bermudan's schedule IS
    # `PathGrid(T, steps)` after tranche 18's widening, the seeds and configs match, so both engines
    # consume the same stream and produce the same payoff path by path. Exactly tranche 17's lesson
    # about a band wide enough to absorb Monte Carlo noise being wide enough to absorb a defect.
    assert priced.pv.amount == reference.pv.amount
    assert priced.standard_error == reference.standard_error


def test_O1_the_second_COORDINATE_is_the_NEXT_coterminal_swap_rate_not_this_one():
    """§12.6's actual finding, pinned pointwise — and nothing else in this file can see it.

    Jäckel tries several projections and settles on *"the first forward rate `f_i` and the forward
    swap rate `s_{i+1}` starting from the reset time of `f_{i+1}` going out to the terminal
    maturity"*, because `(f_i, f_{i+1})` overlaps and `(s_i, s_{i+1})` crowds the diagonal. Those two
    rates are highly correlated, so substituting `s_i` for `s_{i+1}` leaves a *trainable* boundary
    whose price still satisfies every bound in this file — a mutation doing exactly that survived all
    653 tests. The coordinate is therefore transcribed, not inferred from the price."""
    bermudan = _bermudan(_atm())
    model = _model()
    schedule = ObservationSchedule(
        times=_exercise_times(bermudan, model), steps_per_segment=CONFIG.steps_per_segment
    )
    sliced = exercise_slices(bermudan, model, schedule, CONFIG.pricing)
    indices = bermudan.exercise_indices

    for j in (0, 1, 7, 99):
        path_states = sliced.states[j]
        for slot, i in enumerate(indices):
            forwards = _rebuild_forwards(model, schedule, CONFIG.pricing, j, slot)
            assert path_states[slot].forward == forwards[i]
            if i + 1 < SPAN.end:
                nxt = _swap_rate(forwards, TENOR.accruals, i + 1, SPAN.end)
                own = _swap_rate(forwards, TENOR.accruals, i, SPAN.end)
                assert path_states[slot].swap_rate == pytest.approx(nxt, rel=1e-14)
                # Non-vacuity: the two coordinates must actually differ, or the oracle is blind.
                assert abs(nxt - own) > 1e-6


def _rebuild_forwards(model, schedule, config, path: int, slot: int):
    from pricebook.engine.mc_path_engine import simulate_forward_paths

    return simulate_forward_paths(model, schedule, config)[path][slot]


def _swap_rate(forwards, accruals, first: int, end: int) -> float:
    """Eq 5.31 at a simulated state, transcribed here independently of the engine."""
    bonds = [1.0]
    for k in range(first, end):
        bonds.append(bonds[-1] / (1.0 + forwards[k] * accruals[k]))
    annuity = math.fsum(accruals[first + m] * bonds[m + 1] for m in range(end - first))
    return (bonds[0] - bonds[-1]) / annuity


# ============================================================================================
# ORACLE 2 — NON-VACUITY: the early-exercise premium is non-negative, and strictly positive here.
# ============================================================================================


def test_O2_the_EARLY_EXERCISE_PREMIUM_is_non_negative_and_STRICTLY_positive():
    """The companion the reduction requires: a Bermudan that never exercises early collapses to the
    European and would satisfy oracle 1 perfectly. More exercise opportunities can never be worth
    less — that is the finance, not the implementation.

    **Banded at the resolution it can actually assert** (tranche 17's carry-in, and the spec's oracle
    6): the European it is compared against is itself a Monte Carlo number, so the premium is only
    *resolved* above the combined standard error. The test states the smallest premium it can see and
    requires the measured one to clear it, rather than asserting a bare inequality that Monte Carlo
    noise could satisfy in either direction."""
    strike = _atm()
    many = _priced(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    one = _priced(strike, SwaptionKind.PAYER, SPAN.end - 1, CONFIG)
    assert isinstance(many, PricingResult) and isinstance(one, PricingResult)
    assert many.standard_error is not None and one.standard_error is not None

    premium = many.pv.amount - one.pv.amount
    resolution = 4.0 * math.hypot(many.standard_error, one.standard_error)
    assert premium > resolution, (
        f"the early-exercise premium is {premium:,.2f} against a resolution of {resolution:,.2f} — "
        f"either the policy never exercises early, or this fixture cannot see the premium"
    )


def test_O2_the_FIXTURE_has_a_genuine_interior_exercise_region():
    """The seventh watch for a degenerate fixture, in Bermudan dress — the spec names it.

    If early exercise were *never* optimal the premium vanishes; if *always* optimal the boundary is
    trivial and its parametrisation untested. Either way the oracles above go blind. So the premise
    is asserted: the priced Bermudan must sit strictly inside the sandwich, away from both the
    never-exercise floor (the European) and the always-exercise alternative."""
    strike = _atm()
    priced = _priced(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    never = _priced(strike, SwaptionKind.PAYER, SPAN.end - 1, CONFIG)
    foresight = _foresight(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    assert isinstance(priced, PricingResult) and isinstance(never, PricingResult)

    assert never.pv.amount < priced.pv.amount < foresight.pv.amount
    # And the interior is a real interval, not a rounding gap.
    assert (priced.pv.amount - never.pv.amount) / priced.pv.amount > 0.01
    assert (foresight.pv.amount - priced.pv.amount) / priced.pv.amount > 0.01


# ============================================================================================
# ORACLES 3 and 4 — THE SANDWICH, and its width reported as the resolution.
# ============================================================================================


def test_O4_the_policy_price_is_BRACKETED_by_perfect_foresight():
    """`policy <= true <= foresight`, the honest statement of a price with no closed form.

    Perfect foresight exercises at the ex-post best date on each path, which no real strategy can
    do, so it is a genuine upper bound; the trained policy is achievable, so it is a genuine lower
    bound. Neither is the price. What this oracle asserts is that the bracket holds **on the same
    paths** — comparing across different path sets would let Monte Carlo noise invert it."""
    strike = _atm()
    priced = _priced(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    foresight = _foresight(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    assert isinstance(priced, PricingResult)

    assert priced.pv.amount <= foresight.pv.amount
    width = (foresight.pv.amount - priced.pv.amount) / priced.pv.amount
    # The bracket is this tranche's resolution. Asserted to be a real bracket and not absurdly
    # wide; the measured width is reported in the tranche doc rather than tuned to.
    assert 0.0 < width < 0.60


def test_O3_MORE_TRAINING_does_not_LOWER_the_policy_price():
    """A suboptimal policy under-prices, so better training must not make the lower bound worse.

    Jäckel's point: the Bermudan price is the *optimum* over exercise strategies, so any achievable
    policy is a lower bound and improving it moves the bound up. Asserted as a non-decrease within
    Monte Carlo error rather than a strict rise, because the objective is piecewise constant in the
    parameters — more training can leave the fitted boundary genuinely unchanged, which is not a
    failure."""
    strike = _atm()
    coarse = BermudanMcConfig(
        training=MonteCarloConfig(paths=500, seed=18_000), pricing=CONFIG.pricing
    )
    lean = price_bermudan_swaption(_bermudan(strike), _model(), coarse)
    full = _priced(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    assert isinstance(lean, PricingResult) and isinstance(full, PricingResult)
    assert full.standard_error is not None and lean.standard_error is not None

    tolerance = 4.0 * math.hypot(full.standard_error, lean.standard_error)
    assert full.pv.amount > lean.pv.amount - tolerance


# ============================================================================================
# ORACLE 5 — the TRAIN/PRICE separation, asserted rather than trusted.
# ============================================================================================


def test_O5_pricing_on_the_TRAINING_paths_OVER_states_the_value():
    """Jäckel's *"slight upward bias"*, measured instead of taken on faith.

    The policy is fitted to maximise the average payoff **on the training paths**, so scoring it on
    those same paths is in-sample and biased high. The production price uses fresh paths. This is the
    foot-gun the split exists to avoid, and the oracle proves the split is doing work — without it,
    a mutant that priced on the training set would pass every other test here."""
    strike = _atm()
    fresh = _priced(strike, SwaptionKind.PAYER, FIRST_EXERCISE, CONFIG)
    in_sample = price_bermudan_swaption(
        _bermudan(strike),
        _model(),
        BermudanMcConfig(training=CONFIG.training, pricing=CONFIG.training),
    )
    assert isinstance(fresh, PricingResult) and isinstance(in_sample, PricingResult)
    assert in_sample.pv.amount > fresh.pv.amount


def test_O5_the_policy_does_NOT_exercise_into_a_NEGATIVE_value():
    """A branch the simulated paths never reach, relocated to where a test can reach it.

    `_policy_payoff` ignores a firing boundary when the intrinsic is non-positive — Jäckel's step 1
    carries the same `I > 0` condition. Training makes that case rare, because firing out of the
    money *lowers* eq 12.41, so a mutation dropping the check survived the whole suite. That is §4's
    third survivor kind: not an equivalent mutant (a Bermudan paying a negative amount would be
    plainly wrong) but a branch no fixture reaches. The fix is §4's — exercise it directly rather
    than leave it as dead defensive code."""
    always = (BoundaryParams(0.0, 0.0, -1.0),) * 2      # a boundary that fires everywhere
    states = (BoundaryState(0.03, 0.03, 0.03), BoundaryState(0.03, 0.03, 0.03))
    assert _policy_payoff(always, states, (-0.5, 0.2), 1.0) == pytest.approx(0.2)
    assert _policy_payoff(always, states, (-0.5, -0.2), 1.0) == 0.0
    assert _policy_payoff(always, states, (0.3, 0.2), 1.0) == pytest.approx(0.3)


def test_O5_the_BACKWARD_induction_is_reproduced_STEP_BY_STEP_from_eq_12_41():
    """§12.8's recursion, reconstructed here independently — and only this sees the DIRECTION.

    Both directions produce a valid lower bound, so no bound in this file can tell them apart: a
    forward-running mutant survived all 657 tests. A weaker first attempt at this oracle asserted
    that the earliest slice's continuations were non-zero, and **that also failed to catch it** —
    running forwards still rolls the values, it just rolls them through later slices whose boundaries
    have not been fitted yet, so they are non-zero but wrong.

    What distinguishes the directions is therefore the *whole chain*: at each step the roll must use
    the boundary that step's successor was **already trained to**. So the recursion is rebuilt from
    eq 12.39 backwards using only pieces oracled elsewhere — `roll_continuations` and `train_slice`,
    both pinned pointwise in `tests/L3/test_exercise_boundary.py` — and the engine's policy must
    match it exactly, slice for slice."""
    bermudan = _bermudan(_atm())
    model = _model()
    schedule = ObservationSchedule(
        times=_exercise_times(bermudan, model), steps_per_segment=CONFIG.steps_per_segment
    )
    policy = train_exercise_policy(bermudan, model, schedule, CONFIG.training)
    sliced = exercise_slices(bermudan, model, schedule, CONFIG.training)
    omega, strike = bermudan.kind.omega, bermudan.strike
    slots = len(bermudan.exercise_indices)

    def column(slot: int, continuations: tuple[float, ...]) -> TrainingSlice:
        return TrainingSlice(
            states=tuple(sliced.states[j][slot] for j in range(sliced.count)),
            intrinsics=tuple(sliced.intrinsics[j][slot] for j in range(sliced.count)),
            continuations=continuations,
        )

    # Eq 12.39 pins the last slice; everything earlier is fitted in REVERSE order.
    expected = [BoundaryParams.final(strike)] * slots
    values: tuple[float, ...] = (0.0,) * sliced.count
    for slot in range(slots - 2, -1, -1):
        values = roll_continuations(expected[slot + 1], column(slot + 1, values), omega)
        expected[slot] = train_slice(column(slot, values), omega, strike)

    assert policy == tuple(expected)
    # Non-vacuity: the fitted slices must actually differ from the pinned one, or the chain above
    # is reproducing a constant and would match whatever direction the engine ran.
    assert any(p != BoundaryParams.final(strike) for p in policy[:-1])


def test_O5_the_CONTINUATION_ROLL_overwrites_only_where_the_policy_exercises_in_the_money():
    """§12.8 step 1, pointwise — the function that makes the induction's direction load-bearing.

    A mutation that simply never updated the carried values survived every bound in this file, so the
    roll is transcribed against a hand-built three-path slice rather than inferred from a price."""
    fires = BoundaryParams(0.0, 0.0, 0.035)             # exercise iff f > 0.035
    states = (
        BoundaryState(0.050, 0.04, 0.04),               # fires, intrinsic > 0  -> overwrite
        BoundaryState(0.050, 0.04, 0.04),               # fires, intrinsic < 0  -> keep
        BoundaryState(0.020, 0.04, 0.04),               # does not fire         -> keep
    )
    rolled = roll_continuations(
        fires, TrainingSlice(states, (0.9, -0.4, 0.7), (0.1, 0.2, 0.3)), 1.0
    )
    assert rolled == pytest.approx((0.9, 0.2, 0.3))


# ============================================================================================
# ORACLE 7 — determinism, the product round-trip, invariant 7, and the refusals.
# ============================================================================================


def test_O7_the_same_seeds_give_the_SAME_price_and_the_result_carries_its_SE():
    strike = _atm()
    # NOT through the cache: determinism is the claim, so the engine must be called twice.
    a = price_bermudan_swaption(_bermudan(strike), _model(), CONFIG)
    b = price_bermudan_swaption(_bermudan(strike), _model(), CONFIG)
    assert isinstance(a, PricingResult) and isinstance(b, PricingResult)
    assert a.pv.amount == b.pv.amount
    assert a.standard_error is not None and a.standard_error > 0.0
    assert a.standard_error < 0.10 * a.pv.amount


def test_O7_the_product_ROUND_TRIPS_through_its_dict():
    bermudan = _bermudan(0.041, SwaptionKind.RECEIVER)
    assert BermudanSwaption.from_dict(bermudan.to_dict()) == bermudan
    with pytest.raises(ValueError, match="not a bermudan_swaption"):
        BermudanSwaption.from_dict({"kind_": "european_swaption", "schema_version": 1})


def test_O7_a_BAD_product_or_MISMATCHED_numeraire_is_a_PricingFailure():
    strike = _atm()
    wrong_measure = LiborMarketModel(TENOR, RHO, numeraire_index=5)
    out = price_bermudan_swaption(_bermudan(strike), wrong_measure, CONFIG)
    assert isinstance(out, PricingFailure) and "numeraire" in out.reason

    # A span starting at index 0 has its first exercise at t = 0: no optionality to value. This is
    # the engine's to refuse, not the product's — the product does not know the tenor's dates, so it
    # cannot tell that index 0 means today. (The product refuses an index outside its SPAN, below.)
    at_inception = BermudanSwaption(
        SwapSpan(0, 6), strike, NOTIONAL, SwaptionKind.PAYER, 0
    )
    out = price_bermudan_swaption(at_inception, LiborMarketModel(TENOR, RHO, 6), CONFIG)
    assert isinstance(out, PricingFailure) and "t = 0" in out.reason


def test_O7_an_exercise_index_OUTSIDE_the_span_is_refused_by_the_PRODUCT():
    with pytest.raises(ValueError, match="exercise"):
        BermudanSwaption(SPAN, 0.04, NOTIONAL, SwaptionKind.PAYER, SPAN.end)
    with pytest.raises(ValueError, match="exercise"):
        BermudanSwaption(SPAN, 0.04, NOTIONAL, SwaptionKind.PAYER, SPAN.first - 1)


# ---------------------------------------------------------------------------------------------
# Tranche 19 — variance reduction wired into the engine. JAECKEL-2002 §10.1/§10.3.
# ---------------------------------------------------------------------------------------------


def _plain_and_control(first: int = FIRST_EXERCISE):
    """The unreduced payoffs and the control, from the paths the engine itself prices."""
    bermudan = _bermudan(_atm(), first=first)
    model = _model()
    schedule = ObservationSchedule(
        times=_exercise_times(bermudan, model), steps_per_segment=CONFIG.steps_per_segment
    )
    policy = train_exercise_policy(bermudan, model, schedule, CONFIG.training)
    sliced = exercise_slices(bermudan, model, schedule, CONFIG.pricing)
    payoffs = [
        _policy_payoff(policy, sliced.states[j], sliced.intrinsics[j], bermudan.kind.omega)
        for j in range(sliced.count)
    ]
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

    Measured over a 16x path increase, the width holds near 15% while the standard error falls as
    `1/√N`. So this oracle asserts the honest statement: the reduction shrinks the **error bars**
    (which is real and valuable, and the test above measures it), and leaves the **bracket** where it
    is — a reduction that moved the bracket would be introducing bias, so the invariance is itself
    the check."""
    widths, errors = [], []
    for paths in (2_000, 32_000):
        config = BermudanMcConfig(
            training=CONFIG.training, pricing=MonteCarloConfig(paths, CONFIG.pricing.seed)
        )
        policy = _priced(_atm(), SwaptionKind.PAYER, FIRST_EXERCISE, config)
        foresight = _foresight(_atm(), SwaptionKind.PAYER, FIRST_EXERCISE, config)
        assert isinstance(policy, PricingResult) and policy.standard_error is not None
        widths.append((foresight.pv.amount - policy.pv.amount) / policy.pv.amount)
        errors.append(policy.standard_error)

    # The error bar falls at the -1/2 rate over a 16x path increase: a factor of ~4.
    assert errors[0] / errors[1] == pytest.approx(4.0, rel=0.35)
    # The width does NOT: it is the same bracket, not a shrinking one.
    assert widths[1] == pytest.approx(widths[0], rel=0.15)
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
