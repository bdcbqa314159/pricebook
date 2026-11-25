"""T79 — perpetual timer options, the vol-derivatives arc capstone.

`KWOK-ZHENG` §1.3.3 (payoff), §6.2.1 (Theorem 10 / eq 6.10 — the Black–Scholes-type conditional
form), §6.2.3 (the perturbation approximation, this suite's second non-MC pin).

**Production is the MC first-passage pricer** (scope ruled (C) at 0424): the §6.2.2 closed form needs
a complex-argument Bessel, a Bromwich density-inversion and a 2-D quadrature, deferred to the
numerical track. So the MC is the quoted price — and **MC-not-sole**: it is pinned by *two
independent non-MC oracles*, which is what keeps it in-method (§1's oracle gate, the T72 lens):

1. the **exact reduction** at `ρ=0, r=q=0`, where the drift collapses to the deterministic `−B/2`
   and eq (6.10) is exactly Black–Scholes at total variance `B`;
2. the **§6.2.3 perturbation** at small vol-of-vol, whose own error is *measured* rather than
   assumed, and which is non-degenerate in `ρ` — so it covers exactly what oracle 1 is blind to.
"""

# slow-test-exempt: the first-passage simulation is irreducibly the cost here -- every oracle needs
# paths run until their variance clock reaches the budget, and the two non-MC pins (the exact
# rho=0/zero-carry reduction and the §6.2.3 perturbation) are both comparisons AGAINST that
# simulation, so there is no closed form to substitute while §6.2.2 is deferred. Measured 142.7s
# locally after three lossless passes (222.1 -> 175.3 -> 142.7). Ledgered as OI-9 with the removal
# trigger: live-path COMPACTION in `simulate_timer_first_passage`, raised as its own unit per OI-5's
# precedent rather than smuggled into a correctness tranche.

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine import price
from pricebook.engine.timer_option import (
    simulate_timer_first_passage,
    timer_option_estimate,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import EquityAsset, EquityUnderlying, dividend_forward
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.heston import HestonModel, HestonParams
from pricebook.products.equity_option import EquityOptionKind
from pricebook.products.timer_option import TimerOption

scipy_special = pytest.importorskip("scipy.special")

VAL = date(2026, 1, 15)
USD = Currency("USD")
ACME = EquityAsset("ACME")
DC = DayCountConvention.ACT_365_FIXED
SPOT, STRIKE, BUDGET = 100.0, 100.0, 0.08

#: `v0 != theta` is load-bearing: starting at the long-run level collapses the mean-reversion terms
#: and makes the first-passage law nearly deterministic (the spec's degeneracy premise).
PARAMS = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=0.2, rho=-0.5)
#: The reduction's parameters: `rho = 0` AND a zero carry, which oracle 1 needs *both* of.
FLAT = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=0.2, rho=0.0)


def _market(rate: float, dividend: float) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(VAL, rate, DC)}),
        # the repo/borrow rate enters the forward exactly as a yield (C17), so it IS the `q`
        equities={ACME: EquityUnderlying(spot=SPOT, repo_rate=dividend)},
    )


def _model(params: HestonParams, rate: float = 0.03, dividend: float = 0.03) -> HestonModel:
    return HestonModel(market=_market(rate, dividend), params=params)


def _option(
    strike: float = STRIKE,
    budget: float = BUDGET,
    kind: EquityOptionKind = EquityOptionKind.CALL,
) -> TimerOption:
    return TimerOption(
        asset=ACME,
        strike=strike,
        variance_budget=budget,
        notional=Money(1.0, USD),
        kind=kind,
    )


# ============================================================================================
# 1. The first-passage simulation itself — its OWN bias measured before it is trusted.
# ============================================================================================

def test_every_path_REACHES_the_budget_so_nothing_is_scored_as_a_silent_zero():
    """A perpetual timer exercises almost surely; a truncated horizon is a simulation artefact.

    Measured, not assumed: a prototype with a horizon of `3·B/θ` left **5.5% of paths unhit** at
    `σ_v = 0.4` and scored them zero — a downward bias of the worst kind, since it looks like a
    price. The engine runs until every path hits and **refuses** rather than truncating, so this
    test asserts the refusal never fires on a sane fixture rather than asserting a hit *fraction*.
    """
    model = _model(PARAMS)
    tau, spots = simulate_timer_first_passage(
        PARAMS, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=4000, seed=11), 200)
    )
    assert np.all(np.isfinite(tau)) and np.all(tau > 0.0)
    assert np.all(np.isfinite(spots)) and np.all(spots > 0.0)
    # and the fixture is not degenerate: the hitting time genuinely SPREADS
    assert tau.std(ddof=1) / tau.mean() > 0.05, tau.std(ddof=1) / tau.mean()


def test_the_mean_hitting_time_CONVERGES_to_T0_at_rate_sigma_v_SQUARED():
    """`E[τ_B] → T₀` as `σ_v → 0`, at `O(σ_v²)` — a RATE, because a flat band would be wrong.

    **`T₀` is not `E[τ_B]`, and the reason is structural:** eq (6.53)'s PDE for `T₀` carries **no
    second-derivative term**, so `T₀` is the hitting time of the *deterministic* variance path
    `dV = κ(θ−V)dt` — the `σ_v → 0` limit of the mean hitting time, not the mean itself. A first
    version of this test banded them at 5% and failed at 6.8% (`σ_v = 0.2`), which was the band
    being wrong rather than either quantity.

    So the oracle is the T25 rate-detector: the gap must **quarter** when `σ_v` halves. Measured
    0.4168 → 0.1190 → 0.0317 over `σ_v = 0.4, 0.2, 0.1`, i.e. ratios 3.50 and 3.76 against the 4
    that `O(σ_v²)` predicts. Below `σ_v ≈ 0.05` the gap sinks under its own MC noise, which is why
    the sweep stops where it does rather than chasing the limit.
    """
    reference = _maturity_function(PARAMS.v0, PARAMS.kappa, PARAMS.theta, BUDGET)
    gaps = []
    for sigma_v in (0.4, 0.2, 0.1):
        params = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=sigma_v, rho=-0.5)
        times, _ = simulate_timer_first_passage(
            params,
            _market(0.0, 0.0).equities[ACME],
            BUDGET,
            (MonteCarloConfig(paths=20000, seed=5), 400),
        )
        gap = times.mean() - reference
        # the gap is POSITIVE: a stochastic variance hits the budget later on average (Jensen)
        assert gap > 0.0, (sigma_v, gap)
        gaps.append(gap)
    for coarse, fine in zip(gaps, gaps[1:]):
        assert 3.0 < coarse / fine < 5.0, gaps
    # and the premise: the gap is RESOLVABLE at every point of the sweep, so the rate is measured
    # rather than fitted through noise
    assert gaps[-1] > 0.01, gaps


def test_the_hitting_time_is_EXACTLY_B_over_theta_when_the_variance_cannot_move():
    # sigma_v = 0 makes the variance deterministic, so tau_B is a fixed number -- and the timer
    # degenerates into a fixed-horizon option. Asserted because it is the premise the OTHER
    # fixtures must avoid (the spec's degeneracy note), and because it pins the accumulator.
    frozen = HestonParams(v0=0.04, kappa=2.0, theta=0.04, sigma_v=0.0, rho=0.0)
    model = _model(frozen)
    tau, _ = simulate_timer_first_passage(
        frozen, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=64, seed=1), 400)
    )
    assert tau == pytest.approx(BUDGET / frozen.theta, rel=1e-3, abs=0.0)
    # A TOLERANCE, not `== 0.0`: every path follows the identical deterministic variance, so the
    # spread is pure floating-point lane noise (measured 2.2e-16, about one ULP of 2.0), and
    # committing an exact zero would promote an in-process bit-identity to a cross-platform claim
    # (T49).
    assert tau.std(ddof=1) < 1e-12


def test_a_DETERMINISTIC_variance_hits_the_budget_at_EXACTLY_T0_mid_step():
    """`σ_v = 0` with `v₀ ≠ θ` ⇒ `τ_B = T₀` **exactly**, and the hit lands **between** grid steps.

    This is the oracle that pins the within-step **interpolation**, and it took two attempts to
    find — the first one could not work, for a structural reason worth recording. With `v₀ = θ` the
    variance is constant, and the step size is *defined* as `(B/θ)/steps`, so the budget is reached
    at exactly `steps` steps **however** `B` and `θ` are chosen: the hit always lands on a grid
    boundary, the crossing fraction is always 1.0, and rounding to the boundary is **identical** to
    interpolating. Choosing an "off-grid" budget does not help, because the grid is built from the
    budget. Only `v₀ ≠ θ` — a variance that is deterministic but not constant — puts the hit
    mid-step.

    And then the reference is exact rather than approximate: with no vol-of-vol, eq (6.53)'s `T₀`
    *is* the hitting time (that PDE has no second-derivative term, which is the same fact that
    makes `T₀` the `σ_v → 0` limit elsewhere in this file). Measured: the hit falls at grid index
    **527.206** at 400 steps, `τ` agrees with `T₀` to **4.9e-05**, and rounding up to the boundary
    would give **1.46e-03** — a 30× separation, which is what gives this test its power.
    """
    deterministic = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=0.0, rho=-0.5)
    reference = _maturity_function(0.06, 2.0, 0.04, BUDGET)
    underlying = _market(0.0, 0.0).equities[ACME]

    times, _ = simulate_timer_first_passage(
        deterministic, underlying, BUDGET, (MonteCarloConfig(paths=16, seed=1), 400)
    )
    # deterministic: every path is the same path
    assert times.std(ddof=1) == 0.0
    assert times[0] == pytest.approx(reference, rel=2e-4, abs=0.0)

    # the premise that makes this an interpolation test: the hit is NOT on a grid boundary
    step = (BUDGET / max(deterministic.theta, deterministic.v0)) / 400
    position = times[0] / step
    assert 0.1 < position % 1.0 < 0.9, position

    # and it converges as O(dt): halving the step must shrink the error, which rounding to the
    # boundary would NOT do (the rounded value is pinned to the grid, not to the truth)
    coarse, _ = simulate_timer_first_passage(
        deterministic, underlying, BUDGET, (MonteCarloConfig(paths=16, seed=1), 100)
    )
    assert abs(coarse[0] - reference) > 2.0 * abs(times[0] - reference)


def test_the_TOTAL_realised_variance_is_the_BUDGET_even_on_a_coarse_grid():
    """`var(ln S_τ) = B` exactly — which pins the **final partial step's diffusion scaling**.

    `ln S` accumulates variance `∫V dt`, and the contract stops that integral at `B`, so the
    terminal log-spot variance **is** the budget, with no model dependence at all. Run on a
    deliberately **coarse** grid (4 steps) because that is where the final partial step is a large
    share of the total: its crossing fraction is 0.253 here, so a diffusion scaled by the fraction
    instead of its square root loses ~3.2% of the total variance, against a 0.70% SE. On a fine
    grid the same error is invisible — which is why the coarse grid is the oracle and not a
    shortcut.
    """
    deterministic = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=0.0, rho=-0.5)
    _, spots = simulate_timer_first_passage(
        deterministic,
        _market(0.0, 0.0).equities[ACME],
        BUDGET,
        (MonteCarloConfig(paths=40000, seed=21), 4),
    )
    realised = float(np.log(spots).var(ddof=1))
    assert realised == pytest.approx(BUDGET, rel=0.02, abs=0.0), realised


def test_an_EXACT_boundary_hit_is_handled_identically_either_way():
    """The budget reached **exactly** on a step boundary — a *proved equivalent mutant*, recorded.

    Flipping `advanced >= budget` to `advanced > budget` survived the sweep, and it is not a hole:
    it is **behaviour-identical even on the boundary**, which is stronger than unreachable. With
    `>=` the crossing fires now and `fraction = (B−clocks)/(advanced−clocks) = 1.0`. With `>` it
    defers one step, `clocks` becomes exactly `B`, and the next step fires with
    `fraction = (B−B)/(V·dt) = 0.0` — so `hitting = (i+1)·step` either way, and the spot likewise
    takes one full step and then a zero-width one. §4 requires the exception to be **proved**:
    measured **bit-identical** outputs (hitting times and spots) under the mutation, on this
    fixture and on a 500-path stochastic one.

    The fixture makes the boundary genuinely exact rather than nearly so: `θ = v₀ = 0.5`,
    `B = 2.0`, 8 steps gives `step = 0.5` and `V·step = 0.25`, all power-of-two multiples, so the
    clock accumulates to `2.0 == B` with no rounding at all (verified: `clock == budget` is True).
    """
    exact = HestonParams(v0=0.5, kappa=1.0, theta=0.5, sigma_v=0.0, rho=0.0)
    budget, steps = 2.0, 8
    # the premise: the clock really does land on the budget with NO floating-point slack
    step = (budget / max(exact.theta, exact.v0)) / steps
    clock = 0.0
    for _ in range(steps):
        clock = clock + exact.theta * step
    assert clock == budget

    times, _ = simulate_timer_first_passage(
        exact, _market(0.0, 0.0).equities[ACME], budget, (MonteCarloConfig(paths=8, seed=1), steps)
    )
    assert times[0] == budget / exact.theta        # 4.0, exactly -- not approximately
    assert times.std(ddof=1) == 0.0


# ============================================================================================
# 2b. The CARRY — every anchor above needs `r = q`, so the carry needs its own fixture.
# ============================================================================================

def test_the_CARRY_is_restored_at_each_paths_OWN_exercise_time():
    """**The hole the mutation sweep found: every other fixture in this file has `r = q`.**

    Oracle 1 requires `r = q = 0` and oracle 2 requires `r = q` (so that eq (6.50)'s `G` term
    vanishes). Both anchors therefore pin the price with the carry **identically zero**, and three
    mutations survived on that: dropping the `(r−q)` restoration entirely, and applying it at the
    *mean* exercise time instead of each path's own. §4 names this exactly — *an oracle made sharp
    by a quantity being exactly zero is, for that very reason, blind to that quantity*.

    So this fixture has `r ≠ q`, and the reference is transcribed independently of the engine: the
    test derives the carry from `dividend_forward` (C17's definition) and composes the payoff
    itself. Measured, the dropped-carry variant lands **68 SE** from the correct value.
    """
    rate, dividend = 0.05, 0.01
    model = _model(PARAMS, rate=rate, dividend=dividend)
    sampling = (MonteCarloConfig(paths=20000, seed=13), 300)

    times, martingale = simulate_timer_first_passage(
        PARAMS, model.market.equities[ACME], BUDGET, sampling
    )
    underlying = model.market.equities[ACME]
    discount = model.market.curves.discount(USD)
    reference_date = model.market.valuation_date + timedelta(days=365)
    years = discount.time_measure.year_fraction(reference_date)
    carry = (
        math.log(dividend_forward(underlying, reference_date, discount) / underlying.spot) / years
    )
    # the premise: this fixture has a MATERIAL carry, unlike every other one in this file
    assert abs(carry) > 0.03, carry

    spots = martingale * np.exp(carry * times)
    factors = np.array([discount.df_at(float(t)) for t in times])
    samples = factors * np.maximum(spots - STRIKE, 0.0)
    expected = float(samples.mean())

    through_engine = timer_option_estimate(_option(), model, sampling)
    assert through_engine.mean == pytest.approx(expected, rel=1e-12, abs=0.0)

    # and the carry must ride each path's OWN time: applying it at the mean instead changes the
    # answer, because `exp` is convex and the exercise times genuinely spread
    at_mean = factors * np.maximum(martingale * math.exp(carry * times.mean()) - STRIKE, 0.0)
    standard_error = float(samples.std(ddof=1) / math.sqrt(len(samples)))
    assert abs(float(at_mean.mean()) - expected) > 3.0 * standard_error


def test_the_EXPONENTIAL_MARTINGALE_identity_pins_the_carry_without_a_transcription():
    """`E[e^{−carry·τ}·S_τ] = S₀` — exact, and carry-sensitive at 68 SE.

    `S_t·e^{−(r−q)t}` is an exponential martingale, so optional stopping at `τ_B` gives `S₀` on the
    nose. This is the requirement-oracle companion to the transcription above: it survives a
    transcription error in the payoff composition, because it never forms a payoff at all.
    """
    rate, dividend = 0.05, 0.01
    model = _model(PARAMS, rate=rate, dividend=dividend)
    times, martingale = simulate_timer_first_passage(
        PARAMS, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=40000, seed=13), 300)
    )
    standard_error = float(martingale.std(ddof=1) / math.sqrt(len(martingale)))
    assert float(martingale.mean()) == pytest.approx(SPOT, abs=4.0 * standard_error)
    # the premise: WITHOUT the discounting at each path's own time the identity fails badly, so it
    # is the pairing of `exp(-carry*tau)` with `exp(+carry*tau)` that is being asserted
    underlying = model.market.equities[ACME]
    discount = model.market.curves.discount(USD)
    reference_date = model.market.valuation_date + timedelta(days=365)
    years = discount.time_measure.year_fraction(reference_date)
    carry = (
        math.log(dividend_forward(underlying, reference_date, discount) / underlying.spot) / years
    )
    undiscounted = float((martingale * np.exp(carry * times)).mean())
    assert abs(undiscounted - SPOT) > 20.0 * standard_error


def test_a_DISCRETE_dividend_schedule_is_REFUSED_not_silently_averaged():
    """Eq (6.13) assumes a constant continuous carry, so a discrete schedule is out of its domain.

    The guard existed from the start and **survived the sweep** — because every fixture in the file
    supplies a repo/borrow yield and none supplies a discrete dividend, so the rejecting branch was
    never in the test population (T33b's input-population degeneracy). Supplying it is the fix.
    """
    from pricebook.market.equity import DividendSchedule

    lumpy = EquityUnderlying(
        spot=SPOT,
        repo_rate=0.0,
        dividends=DividendSchedule(cash=((VAL + timedelta(days=200), 2.5),)),
    )
    market = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(VAL, 0.03, DC)}),
        equities={ACME: lumpy},
    )
    model = HestonModel(market=market, params=PARAMS)
    with pytest.raises(ValueError, match="dividend"):
        timer_option_estimate(_option(), model, (MonteCarloConfig(paths=64, seed=1), 100))
    # and through the registry it becomes a FAILURE value, never an escaping exception (§3.4)
    assert isinstance(price(_option(), model), PricingFailure)


# ============================================================================================
# 2. ORACLE 1 — the exact reduction: rho = 0 AND r = q = 0 ⇒ Black–Scholes at variance B.
# ============================================================================================

def _black_call(spot: float, strike: float, total_variance: float) -> float:
    """`S·N(d₊) − K·N(d₋)` at total (un-annualised) variance — eq (6.10), the test's reference."""
    vol = math.sqrt(total_variance)
    d_plus = math.log(spot / strike) / vol + vol / 2.0
    d_minus = d_plus - vol
    normal = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return spot * normal(d_plus) - strike * normal(d_minus)


def test_at_rho_ZERO_and_zero_carry_the_timer_is_BLACK_SCHOLES_at_variance_B():
    """**Oracle 1, and `r=q=0` is required, not cosmetic.**

    `µ_τB = (r−q)τ_B − B/2` (eq 6.14 at `ρ=0`). The `(r−q)τ_B` term is **random**, because `τ_B`
    is, so only at `r=q=0` does the drift become the deterministic `−B/2` — whereupon `ln(S_τB/S₀)
    ~ N(−B/2, B)` with no model dependence left at all, and eq (6.10) is exact. Confirmed at step 1
    (thread 0422): the spec flagged this as conditional and the condition **binds**.

    Three strikes share **one** simulation, which is lossless rather than a saving: the
    first-passage law does not depend on the strike, so simulating per strike would be computing
    the same paths three times. One strike also goes through `timer_option_estimate` itself, so the
    production path is pinned and not merely the payoff algebra.
    """
    model = _model(FLAT, rate=0.0, dividend=0.0)
    sampling = (MonteCarloConfig(paths=40000, seed=7), 300)
    times, spots = simulate_timer_first_passage(
        FLAT, model.market.equities[ACME], BUDGET, sampling
    )
    discount = model.market.curves.discount(USD)
    factors = np.array([discount.df_at(float(t)) for t in times])

    for strike in (80.0, 100.0, 130.0):
        # the payoff definition, transcribed independently of the engine
        samples = factors * np.maximum(spots - strike, 0.0)
        mean = float(samples.mean())
        standard_error = float(samples.std(ddof=1) / math.sqrt(len(samples)))
        expected = _black_call(SPOT, strike, BUDGET)
        assert abs(mean - expected) < 4.0 * standard_error, (strike, mean, expected)

    # and the production path itself, on the same fixture, so this is not only a payoff check
    through_engine = timer_option_estimate(_option(), model, sampling)
    assert abs(through_engine.mean - _black_call(SPOT, STRIKE, BUDGET)) < (
        4.0 * through_engine.standard_error
    )


#: `ρ` sweep simulated ONCE and read by both tests below — the price question and the joint-law
#: question are different questions about the **same** paths, so simulating twice would compute the
#: same thing twice (OI-8's first lossless move). `σ_v = 0.4` rather than 0.2 because a larger
#: vol-of-vol *amplifies* `ρ`'s effect on the price, which makes the insensitivity claim harder,
#: not easier.
_RHO_SWEEP = (-0.9, -0.5, 0.0, 0.5, 0.9)


@pytest.fixture(scope="module")
def rho_sweep():
    """`{ρ: (times, martingale_spots)}` at `σ_v = 0.4`, zero carry, one simulation per `ρ`."""
    underlying = _market(0.0, 0.0).equities[ACME]
    sampling = (MonteCarloConfig(paths=12000, seed=5), 300)
    paths = {}
    for rho in _RHO_SWEEP:
        params = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=0.4, rho=rho)
        paths[rho] = simulate_timer_first_passage(params, underlying, BUDGET, sampling)
    return paths


def test_the_PRICE_is_almost_BLIND_to_rho_so_no_price_oracle_can_pin_it(rho_sweep):
    """**A finding, not a premise — and it changes what the oracle set must do (T34's shape).**

    The spec assumed oracle 2 would cover what oracle 1 is blind to, because oracle 1 fixes `ρ=0`.
    Measured, that assumption is wrong at the *price* level: at zero carry, turning `ρ` from 0 to
    **−0.9** moves the price by **0.23 SE** (at `σ_v = 0.2`; 1.04 SE even at `σ_v = 0.9`). No path
    count rescues that — it is T34 exactly: a parameter whose first-order effect on the chosen
    observable is structurally negligible cannot be pinned by that observable at any precision.

    And the reason is the timer's defining feature: `ln S` has instantaneous variance `V`, so its
    quadratic variation at `τ_B` is **exactly the budget `B`, by construction, whatever `ρ` is**.
    Fixing the variance is the whole point of the contract, so `ρ` can only reach the price through
    the *decomposition* of that fixed variance into drift and diffusion — a second-order effect.

    This test records the insensitivity so the suite **states** it, rather than letting a
    `ρ`-mutation survive silently while a band appears to cover it. `ρ` is pinned below instead.
    """
    discount = _market(0.0, 0.0).curves.discount(USD)
    prices = {}
    for rho, (times, spots) in rho_sweep.items():
        factors = np.array([discount.df_at(float(t)) for t in times])
        samples = factors * np.maximum(spots - STRIKE, 0.0)
        prices[rho] = (
            float(samples.mean()),
            float(samples.std(ddof=1) / math.sqrt(len(samples))),
        )
    base, base_error = prices[0.0]
    for rho in (-0.9, -0.5, 0.5, 0.9):
        value, error = prices[rho]
        assert abs(value - base) < 3.0 * (base_error + error), (rho, value, base)


def test_rho_IS_pinned_by_the_JOINT_LAW_of_the_exercise_time_and_the_spot(rho_sweep):
    """The pin the price cannot provide — `corr(ln S_τB, τ_B)`, at 100–500 SE (the T17 move).

    `ρ` enters the *joint* law of `(S_τB, τ_B)` at first order even though it barely enters the
    price: with `ρ < 0` a high variance both pushes the spot **down** and makes the clock run
    **fast** (small `τ`), so a falling spot coincides with early exercise and the correlation is
    **positive**. Measured `+0.824` at `ρ=−0.9`, `+0.457` at `−0.5`, `+0.004` at `0`, and
    sign-flipped and symmetric for positive `ρ` — 513 SE from zero at 40k paths, against a price
    effect of under one SE. Run at **12k** here, where the margin is still ~280 SE and these claims
    need about ten: a reduction in an unused margin, not in the oracle's domain, and stated so the
    choice is measured rather than default.

    The **sign relation and the antisymmetry** are asserted, not only the magnitude: a
    magnitude-only check would pass for `ρ → −ρ`, which is T43's presence-and-sign-versus-magnitude
    trap with the roles reversed.
    """
    measured = {
        rho: float(np.corrcoef(np.log(spots), times)[0, 1])
        for rho, (times, spots) in rho_sweep.items()
    }
    assert measured[0.0] == pytest.approx(0.0, abs=0.02), measured
    for rho in (-0.9, -0.5):
        assert measured[rho] > 0.3, measured
        assert measured[-rho] < -0.3, measured
    ordered = [measured[rho] for rho in _RHO_SWEEP]
    assert ordered == sorted(ordered, reverse=True), measured
    for rho in (0.5, 0.9):
        assert measured[rho] == pytest.approx(-measured[-rho], rel=0.05, abs=0.0), measured


# ============================================================================================
# 3. ORACLE 2 — the §6.2.3 perturbation, non-degenerate in rho, its own error MEASURED.
# ============================================================================================

def _maturity_function(v0: float, kappa: float, theta: float, budget: float) -> float:
    """`T₀(V,0)` — eq (6.56) `(1/κ)ln(z₀/z)` via Lambert W, with `z` from eq (6.54).

    Test-only, and scipy's `lambertw` is why (the T20a arrangement: the oracle may reach for
    scipy, the library may not). Eq (6.57) is the source's own second representation of the same
    quantity and is asserted against this one below — the §4 derivation-step cross-check.
    """
    z_naught = (v0 - theta) / theta
    if z_naught == 0.0:
        return budget / theta
    z = scipy_special.lambertw(z_naught * math.exp(z_naught - kappa * budget / theta)).real
    return math.log(z_naught / z) / kappa


def _maturity_function_alternative(
    v0: float, kappa: float, theta: float, budget: float
) -> float:
    """Eq (6.57) `(z − z₀)/κ + (B−I)/θ` — the source's *other* printed form of `T₀`."""
    z_naught = (v0 - theta) / theta
    if z_naught == 0.0:
        return budget / theta
    z = scipy_special.lambertw(z_naught * math.exp(z_naught - kappa * budget / theta)).real
    return (z - z_naught) / kappa + budget / theta


def test_the_sources_TWO_printed_forms_of_T0_agree_which_is_the_derivation_step_check():
    """Eqs (6.56) and (6.57) are two representations of one quantity — so they must agree.

    §4's derivation-step rule: where a source prints a formula twice, the second is an independent
    check on the first, and stronger than two renderings of one line. They agree **exactly** here
    (measured 0.0 relative at three parameter sets), which follows from `z·e^z = z₀·e^{z₀−κB/θ}`
    ⇒ `ln(z₀/z) = z − z₀ + κB/θ`.
    """
    for v0, kappa, theta, budget in (
        (0.06, 2.0, 0.04, 0.08),
        (0.02, 1.5, 0.04, 0.05),
        (0.09, 3.0, 0.05, 0.20),
    ):
        first = _maturity_function(v0, kappa, theta, budget)
        second = _maturity_function_alternative(v0, kappa, theta, budget)
        assert first == pytest.approx(second, rel=1e-13, abs=0.0)
    # the premise: these are NOT trivially equal -- T0 differs from the naive B/theta guess
    assert abs(_maturity_function(0.06, 2.0, 0.04, 0.08) / (0.08 / 0.04) - 1.0) > 0.1


def test_T0_satisfies_its_own_BOUNDARY_condition_and_its_frozen_variance_value():
    # T0(V,B) = 0 at I = B, and the APPROACH is `B/v0` -- the clock starts at `v0`, not at
    # `theta`, so the reference is derived rather than bounded loosely. A first version asserted
    # `T0 < 10*B`, which is false and was false for a reason: at `v0 = 0.06` the ratio is `1/0.06`.
    for budget in (1e-6, 1e-4):
        assert _maturity_function(0.06, 2.0, 0.04, budget) == pytest.approx(
            budget / 0.06, rel=1e-3, abs=0.0
        )
    # and at v0 = theta the variance sits at its long-run level, so the clock runs at exactly theta
    assert _maturity_function(0.04, 2.0, 0.04, 0.08) == pytest.approx(2.0, rel=1e-14, abs=0.0)


def _perturbation_call(params: HestonParams, budget: float, rate: float) -> float:
    """§6.2.3's leading-order price at `r = q`, where `Σ² = B − I` is EXACT.

    Eq (6.27) is `S·e^{−qT̃}N(d₊) − K·e^{−rT}N(d₋)` with `Σ² = B − I + 2ερ(r−q)G(V,I)` (eq 6.50).
    **At `r = q` the `G` term vanishes identically**, so `Σ² = B` at inception with no second PDE
    to solve — which is why this oracle is run at `r = q` rather than at a general carry. `T̃₀` is
    `T₀` under the modified drift `α̃(V) = α(V) + ερ√V·β(V) = κθ − (κ−ερ)V`, i.e. the same Lambert-W
    formula with `κ → κ−ερ` and `θ → κθ/(κ−ερ)` (the source's `κ − ρε > 0` assumption). `T̃₁ = T₁ =
    0` (eq 6.41), so this is accurate to `O(ε²)` — the error this oracle's band must measure.
    """
    modified_kappa = params.kappa - params.sigma_v * params.rho
    modified_theta = params.kappa * params.theta / modified_kappa
    plain = _maturity_function(params.v0, params.kappa, params.theta, budget)
    tilde = _maturity_function(params.v0, modified_kappa, modified_theta, budget)
    vol = math.sqrt(budget)
    numerator = math.log(SPOT / STRIKE) + rate * plain - rate * tilde
    d_plus = numerator / vol + vol / 2.0
    d_minus = numerator / vol - vol / 2.0
    normal = lambda x: 0.5 * math.erfc(-x / math.sqrt(2.0))
    return SPOT * math.exp(-rate * tilde) * normal(d_plus) - STRIKE * math.exp(
        -rate * plain
    ) * normal(d_minus)


def test_the_MC_matches_the_PERTURBATION_in_its_small_vol_of_vol_regime():
    """**Oracle 2** — the second non-MC pin, and the one that is non-degenerate in `ρ`.

    Banded `4·SE + the perturbation's MEASURED error`, the two kept apart rather than lumped into
    one percentage (§4). The perturbation is `O(ε²)`, so the band's approximation term is measured
    from the `ε`-sweep below rather than guessed: at `ε = 0.1` the gap is ~0.3 SE and at `ε = 0.2`
    ~1.9 SE, so 1% of the price covers it with room at `ε ≤ 0.2`.
    """
    rate = 0.03
    for sigma_v in (0.05, 0.1, 0.2):
        params = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=sigma_v, rho=-0.5)
        model = _model(params, rate=rate, dividend=rate)
        estimate = timer_option_estimate(
            _option(), model, (MonteCarloConfig(paths=40000, seed=3), 300)
        )
        reference = _perturbation_call(params, BUDGET, rate)
        assert abs(estimate.mean - reference) < 4.0 * estimate.standard_error + 0.01 * reference, (
            sigma_v, estimate.mean, reference, estimate.standard_error
        )


def test_the_perturbations_error_GROWS_with_the_vol_of_vol_which_states_its_regime():
    """A validity regime is only stated if it is measured (T25: "a reference is code too").

    The perturbation is an `O(ε²)` approximation, so its disagreement with the MC must *grow* with
    `ε` — and does: it is noise-level at `ε = 0.05` and unmistakable by `ε = 0.6`. Without this the
    band above would be asserting the perturbation is right, rather than right *where it is used*.
    """
    rate = 0.03
    gaps = []
    for sigma_v in (0.05, 0.3, 0.6):
        params = HestonParams(v0=0.06, kappa=2.0, theta=0.04, sigma_v=sigma_v, rho=-0.5)
        model = _model(params, rate=rate, dividend=rate)
        estimate = timer_option_estimate(
            _option(), model, (MonteCarloConfig(paths=20000, seed=3), 300)
        )
        gaps.append(abs(estimate.mean - _perturbation_call(params, BUDGET, rate)))
    assert gaps == sorted(gaps), gaps
    assert gaps[-1] > 10.0 * gaps[0], gaps


# ============================================================================================
# 4. Self-consistency — parity and budget monotonicity.
# ============================================================================================

def test_PUT_CALL_PARITY_holds_PATHWISE_so_it_is_a_guard_not_an_anchor():
    """`(S−K)⁺ − (K−S)⁺ = S − K` **pathwise**, so parity is exact on shared paths.

    Recorded as what it is: a **telescoping** self-consistency check (§4). Both sides read the same
    simulated `(τ_B, S_τB)`, so it is satisfied by construction and would survive a wrong
    first-passage law entirely. It guards the payoff algebra and the discounting, nothing more —
    the anchors are oracles 1 and 2.
    """
    model = _model(PARAMS)
    sampling = (MonteCarloConfig(paths=20000, seed=9), 300)
    call = timer_option_estimate(_option(kind=EquityOptionKind.CALL), model, sampling)
    put = timer_option_estimate(_option(kind=EquityOptionKind.PUT), model, sampling)
    # The forward side is read off the SAME simulation rather than a third run -- lossless, since
    # the paths are a deterministic function of the sampling and would be identical anyway.
    times, martingale = simulate_timer_first_passage(
        PARAMS, model.market.equities[ACME], BUDGET, sampling
    )
    underlying = model.market.equities[ACME]
    discount = model.market.curves.discount(USD)
    reference = model.market.valuation_date + timedelta(days=365)
    years = discount.time_measure.year_fraction(reference)
    carry = math.log(dividend_forward(underlying, reference, discount) / underlying.spot) / years
    spots = martingale * np.exp(carry * times)
    factors = np.array([discount.df_at(float(t)) for t in times])
    expected = float(np.mean(factors * (spots - STRIKE)))
    assert call.mean - put.mean == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_the_price_is_MONOTONE_in_the_variance_budget():
    # More variance budget means more time and more diffusion before exercise, so more optional
    # value. The finance forces it, so it survives a transcription error in the payoff (§4).
    model = _model(PARAMS)
    sampling = (MonteCarloConfig(paths=20000, seed=4), 300)
    prices = [
        timer_option_estimate(_option(budget=budget), model, sampling).mean
        for budget in (0.02, 0.08, 0.20, 0.50)
    ]
    assert prices == sorted(prices), prices
    assert prices[-1] > 2.0 * prices[0], prices


def test_the_standard_error_falls_as_ONE_OVER_ROOT_N():
    model = _model(PARAMS)
    errors = [
        timer_option_estimate(_option(), model, (MonteCarloConfig(paths=paths, seed=2), 200)).standard_error
        for paths in (5000, 20000, 80000)
    ]
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(2.0, rel=0.25), errors


# ============================================================================================
# 5. The engine.
# ============================================================================================

def test_the_timer_option_prices_through_the_REGISTRY():
    result = price(_option(), _model(PARAMS))
    assert isinstance(result, PricingResult)
    assert result.pv.amount > 0.0
    # invariant 7: an MC result carries its standard error, because for an estimate the SE IS part
    # of the result (§3.7). A perpetual option has no expiry, so invariant 6 has no flow to filter.
    assert result.standard_error is not None and result.standard_error > 0.0


def test_the_notional_SCALES_the_price_and_the_error_together():
    model = _model(PARAMS)
    one = price(_option(), model)
    assert isinstance(one, PricingResult)
    many = price(
        TimerOption(
            asset=ACME,
            strike=STRIKE,
            variance_budget=BUDGET,
            notional=Money(1000.0, USD),
            kind=EquityOptionKind.CALL,
        ),
        model,
    )
    assert isinstance(many, PricingResult)
    assert many.pv.amount == pytest.approx(1000.0 * one.pv.amount, rel=1e-12, abs=0.0)
    assert many.standard_error == pytest.approx(1000.0 * one.standard_error, rel=1e-12, abs=0.0)


def test_a_missing_curve_or_missing_equity_is_a_PRICING_FAILURE():
    bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
    assert isinstance(price(_option(), HestonModel(market=bare, params=PARAMS)), PricingFailure)
    no_equity = MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): DiscountCurve.flat(VAL, 0.03, DC)}),
    )
    assert isinstance(
        price(_option(), HestonModel(market=no_equity, params=PARAMS)), PricingFailure
    )


# ============================================================================================
# 6. Refusals — supplying the input each guard exists to reject (T33b).
# ============================================================================================

@pytest.mark.parametrize("strike, budget", [(-1.0, 0.08), (0.0, 0.08), (100.0, 0.0), (100.0, -0.1)])
def test_the_product_REFUSES_values_outside_its_domain(strike, budget):
    with pytest.raises(ValueError):
        TimerOption(
            asset=ACME,
            strike=strike,
            variance_budget=budget,
            notional=Money(1.0, USD),
            kind=EquityOptionKind.CALL,
        )


def test_the_simulation_REFUSES_a_non_positive_step_count():
    model = _model(PARAMS)
    with pytest.raises(ValueError):
        simulate_timer_first_passage(
            PARAMS, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=16, seed=1), 0)
        )


def test_the_simulation_REFUSES_rather_than_TRUNCATING_when_a_path_cannot_reach_the_budget():
    """The guard that exists because the prototype's silent zero was a real measured bias.

    A variance pinned at zero never accumulates, so the budget is unreachable and no horizon
    suffices. The engine must say so rather than return paths scored at zero — which is the input
    this guard exists to reject, supplied here rather than assumed unreachable (T33b).
    """
    # The step size scales as `B/max(theta, v0)`, so a uniformly tiny variance is NOT unreachable
    # -- the clock just takes tiny steps and still arrives. What is genuinely unreachable is a
    # variance that starts far BELOW a large `theta` with almost no mean reversion to lift it: the
    # step is sized by `theta` while the clock advances at `v0`. Supplying that input is what makes
    # the guard reachable at all (T33b), and a first fixture that merely set everything to 1e-300
    # did not fire it.
    starved = HestonParams(v0=1e-8, kappa=1e-6, theta=100.0, sigma_v=0.0, rho=0.0)
    model = _model(starved)
    with pytest.raises(ValueError, match="budget"):
        simulate_timer_first_passage(
            starved, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=16, seed=1), 50)
        )
