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

import pytest

from pricebook.engine.bermudan_swaption_mc import (
    BermudanMcConfig,
    perfect_foresight_price,
    price_bermudan_swaption,
)
from pricebook.engine.mc_path_engine import PathGrid
from pricebook.engine.swaption_mc import price_european_swaption
from pricebook.foundation import Currency
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor
from pricebook.models.swap_rate_vol import forward_swap_rate
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

    priced = price_bermudan_swaption(bermudan, _model(), CONFIG)
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

    combined = math.hypot(priced.standard_error, reference.standard_error)
    assert abs(priced.pv.amount - reference.pv.amount) < 4.0 * combined


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
    many = price_bermudan_swaption(_bermudan(strike, first=FIRST_EXERCISE), _model(), CONFIG)
    one = price_bermudan_swaption(_bermudan(strike, first=SPAN.end - 1), _model(), CONFIG)
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
    priced = price_bermudan_swaption(_bermudan(strike), _model(), CONFIG)
    never = price_bermudan_swaption(_bermudan(strike, first=SPAN.end - 1), _model(), CONFIG)
    foresight = perfect_foresight_price(_bermudan(strike), _model(), CONFIG)
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
    priced = price_bermudan_swaption(_bermudan(strike), _model(), CONFIG)
    foresight = perfect_foresight_price(_bermudan(strike), _model(), CONFIG)
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
    full = price_bermudan_swaption(_bermudan(strike), _model(), CONFIG)
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
    fresh = price_bermudan_swaption(_bermudan(strike), _model(), CONFIG)
    in_sample = price_bermudan_swaption(
        _bermudan(strike),
        _model(),
        BermudanMcConfig(training=CONFIG.training, pricing=CONFIG.training),
    )
    assert isinstance(fresh, PricingResult) and isinstance(in_sample, PricingResult)
    assert in_sample.pv.amount > fresh.pv.amount


# ============================================================================================
# ORACLE 7 — determinism, the product round-trip, invariant 7, and the refusals.
# ============================================================================================


def test_O7_the_same_seeds_give_the_SAME_price_and_the_result_carries_its_SE():
    strike = _atm()
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

    # An exercise index at the span's start would expire at t=0: no optionality to value.
    out = price_bermudan_swaption(_bermudan(strike, first=0), _model(), CONFIG)
    assert isinstance(out, PricingFailure)


def test_O7_an_exercise_index_OUTSIDE_the_span_is_refused_by_the_PRODUCT():
    with pytest.raises(ValueError, match="exercise"):
        BermudanSwaption(SPAN, 0.04, NOTIONAL, SwaptionKind.PAYER, SPAN.end)
    with pytest.raises(ValueError, match="exercise"):
        BermudanSwaption(SPAN, 0.04, NOTIONAL, SwaptionKind.PAYER, SPAN.first - 1)
