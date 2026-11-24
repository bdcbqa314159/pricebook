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
from pricebook.market.equity import EquityAsset, EquityUnderlying
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


def test_the_MEAN_hitting_time_matches_the_PERTURBATION_maturity_function():
    """`E[τ_B]` against `T₀(V₀,0)` — eq (6.56), an independent non-MC reference for the clock.

    This pins the *first-passage law* on its own, before any payoff: a hitting-time bias would
    otherwise hide inside the price, where the discount factor and the payoff could offset it.
    """
    model = _model(PARAMS)
    tau, _ = simulate_timer_first_passage(
        PARAMS, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=20000, seed=5), 400)
    )
    reference = _maturity_function(PARAMS.v0, PARAMS.kappa, PARAMS.theta, BUDGET)
    standard_error = tau.std(ddof=1) / math.sqrt(len(tau))
    # T0 is the ZEROTH-order function, so it carries its own O(sigma_v^2) error -- banded by the
    # SE plus that measured error, never by one flat percentage (§4).
    assert abs(tau.mean() - reference) < 4.0 * standard_error + 0.05 * reference, (
        tau.mean(), reference, standard_error
    )


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
    assert tau.std(ddof=1) == 0.0


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


@pytest.mark.parametrize("strike", [80.0, 100.0, 130.0])
def test_at_rho_ZERO_and_zero_carry_the_timer_is_BLACK_SCHOLES_at_variance_B(strike):
    """**Oracle 1, and `r=q=0` is required, not cosmetic.**

    `µ_τB = (r−q)τ_B − B/2` (eq 6.14 at `ρ=0`). The `(r−q)τ_B` term is **random**, because `τ_B`
    is, so only at `r=q=0` does the drift become the deterministic `−B/2` — whereupon `ln(S_τB/S₀)
    ~ N(−B/2, B)` with no model dependence left at all, and eq (6.10) is exact. Confirmed at step 1
    (thread 0422): the spec flagged this as conditional and the condition **binds**.
    """
    model = _model(FLAT, rate=0.0, dividend=0.0)
    estimate = timer_option_estimate(
        _option(strike=strike), model, (MonteCarloConfig(paths=40000, seed=7), 300)
    )
    expected = _black_call(SPOT, strike, BUDGET)
    assert abs(estimate.mean - expected) < 4.0 * estimate.standard_error, (
        estimate.mean, expected, estimate.standard_error
    )


def test_the_reduction_is_BLIND_to_rho_which_is_why_it_needs_the_perturbation_companion():
    """The T39 premise, stated as a measurement rather than a remark.

    Oracle 1 holds `ρ=0`; the whole first-passage drift `(ρ/ε)(V_τB − V₀ − κθτ_B + κB)` lives in
    `ρ`. So the reduction cannot see it, and a mutation to the `ρ` term would survive oracle 1
    alone. Measured here: turning `ρ` on moves the price materially at the same budget.
    """
    sampling = (MonteCarloConfig(paths=40000, seed=7), 300)
    flat = timer_option_estimate(_option(), _model(FLAT, 0.0, 0.0), sampling)
    skewed = timer_option_estimate(_option(), _model(PARAMS, 0.0, 0.0), sampling)
    gap = abs(skewed.mean - flat.mean)
    assert gap > 4.0 * (flat.standard_error + skewed.standard_error), (gap, flat, skewed)


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
    # T0(V,B) = 0 at I = B, i.e. a vanishing budget is hit immediately.
    for budget in (1e-6, 1e-4):
        assert _maturity_function(0.06, 2.0, 0.04, budget) < 10.0 * budget
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
    tau, spots = simulate_timer_first_passage(
        PARAMS, model.market.equities[ACME], BUDGET, sampling
    )
    discount = model.market.curves.discount(USD)
    factors = np.array([discount.df_at(float(t)) for t in tau])
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
    dead = HestonParams(v0=1e-300, kappa=1e-300, theta=1e-300, sigma_v=0.0, rho=0.0)
    model = _model(dead)
    with pytest.raises(ValueError, match="budget"):
        simulate_timer_first_passage(
            dead, model.market.equities[ACME], BUDGET, (MonteCarloConfig(paths=16, seed=1), 50)
        )
