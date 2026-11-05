"""The European swaption under the LMM by Monte Carlo (L4). MCINERNEY-ZASTAWNIAK §5.9-5.12.

The payoff at expiry `T_first` is `N · A(T_first) · [ω(S(T_first) − K)]⁺` with `ω = +1` for a payer
and `−1` for a receiver. Under the `P_N`-forward measure with `N` the span's end bond, the price is

    V(0) = B(0,T_N) · E[ (A/B_N)(T_first) · (ω(S − K))⁺ ] · notional

and `(A/B_N)(T_first) = Σ_m τ_{first+m} · Π_{k=first+m+1}^{N−1} (1 + f_k τ_k)` is a function of the
terminal forwards alone — which is why T16's `simulate_terminal_forwards` serves this product
unchanged, exactly as the T17 spec ruled (the path-return widening is T18's, pulled by the
Bermudan's intermediate states).

**The anchor is the exact reduction, and it is sharper than a tolerance.** A one-period span's payoff
is `τ_i(f_i − K)⁺` — a caplet — because `A/B_N = τ_i` and `S = f_i` hold IDENTICALLY, per path, not
approximately. So this engine and T16's caplet engine, on the same seed, must agree to machine
precision after the deterministic `τ · B(0,T_{i+1}) · notional` scaling. That is a cross-product
oracle with different code on each side (§4), and it is noise-free.
"""

import math

import pytest

from pricebook.engine.mc_path_engine import PathGrid, mc_caplet_price
from pricebook.engine.swaption_mc import price_european_swaption
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation.money import Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.models.black import black_price
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.libor_market_model import LiborMarketModel, LmmCorrelation, LmmTenor
from pricebook.models.swap_rate_vol import (
    forward_swap_rate,
    rebonato_swaption_vol,
    tenor_discount_factors,
)
from pricebook.products.swaption import EuropeanSwaption, SwapSpan, SwaptionKind

TENOR = LmmTenor(
    accruals=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
    forwards=(0.030, 0.034, 0.038, 0.042, 0.045, 0.047),
    volatilities=(0.22, 0.20, 0.18, 0.16, 0.15, 0.14),
)
RHO = LmmCorrelation(
    tuple(tuple(math.exp(-0.35 * abs(i - j)) for j in range(6)) for i in range(6))
)
NOTIONAL = Money(1_000_000.0)
CONFIG = MonteCarloConfig(paths=20_000, seed=20261106)


def _swaption(first: int, count: int, strike: float, kind=SwaptionKind.PAYER) -> EuropeanSwaption:
    return EuropeanSwaption(SwapSpan(first, count), strike, NOTIONAL, kind)


def _model(first: int, count: int, rho: LmmCorrelation = RHO) -> LiborMarketModel:
    return LiborMarketModel(TENOR, rho, numeraire_index=first + count)


def _expiry(first: int) -> float:
    return math.fsum(TENOR.accruals[:first])


def _grid(first: int, steps: int = 8) -> PathGrid:
    return PathGrid(horizon=_expiry(first), steps=steps)


# ============================================================================================
# ORACLE 1 — THE EXACT REDUCTION: a one-period swaption IS the T16 caplet. Machine precision.
# ============================================================================================


@pytest.mark.parametrize("first", [1, 2, 3, 4])
def test_O1_a_ONE_period_swaption_equals_T16s_CAPLET_engine_to_machine_precision(first):
    """Different code each side, same seed, no Monte Carlo error in the comparison.

    `A/B_N = τ_i` and `S = f_i` are identities for a one-period span, so the swaption payoff is the
    caplet payoff times `τ_i`, path by path. Both engines consume the same stream from the same
    model, so the only difference is the deterministic scaling — a cross-product oracle, which §4
    singles out as the kind that catches a shared source-reading error no within-product check can."""
    model = _model(first, 1)
    strike = TENOR.forwards[first]
    grid = _grid(first)

    priced = price_european_swaption(_swaption(first, 1, strike), model, grid, CONFIG)
    assert isinstance(priced, PricingResult)

    caplet = mc_caplet_price(model, first, strike, grid, CONFIG)
    bonds = tenor_discount_factors(TENOR)
    expected = caplet.mean * TENOR.accruals[first] * bonds[first + 1] * NOTIONAL.amount
    assert priced.pv.amount == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("first", [2, 3])
def test_O1_the_one_period_swaption_reprices_to_our_BLACK_within_four_standard_errors(first):
    """And the reduction lands on a closed form: `v = σ_i` exactly (Rebonato's one-period case), so
    the Black swaption price with the Rebonato vol IS the Black caplet. No freezing slack at all in
    this case — the only gap is Monte Carlo error."""
    model = _model(first, 1)
    strike = TENOR.forwards[first]
    grid = _grid(first)
    priced = price_european_swaption(_swaption(first, 1, strike), model, grid, CONFIG)
    assert isinstance(priced, PricingResult)

    caplet = mc_caplet_price(model, first, strike, grid, CONFIG)
    scale = TENOR.accruals[first] * tenor_discount_factors(TENOR)[first + 1] * NOTIONAL.amount
    vol = rebonato_swaption_vol(model, first, 1)
    assert vol == pytest.approx(TENOR.volatilities[first], rel=1e-15)
    closed = (
        black_price(TENOR.forwards[first], strike, vol**2 * _expiry(first), CapFloorKind.CAP)
        * scale
    )
    assert abs(priced.pv.amount - closed) < 4.0 * caplet.standard_error * scale


# ============================================================================================
# ORACLE 2 — NON-VACUITY: the multi-period price MOVES with the correlation.
# ============================================================================================


@pytest.mark.parametrize("first", [2])
def test_O2_the_price_RISES_with_correlation_by_more_than_the_monte_carlo_noise(first):
    """`first >= 1` throughout this file: a span starting at index 0 expires at `t = 0`, where the
    total variance is zero and every correlation gives the same intrinsic value — the degenerate
    fixture that would make this oracle vacuous."""
    count = 6 - first
    strike = forward_swap_rate(TENOR, first, count)
    grid = _grid(first)
    prices = {}
    for label, rho in (("decaying", RHO), ("flat80", LmmCorrelation.flat(6, 0.80)),
                       ("perfect", LmmCorrelation.flat(6, 1.0))):
        out = price_european_swaption(
            _swaption(first, count, strike), _model(first, count, rho), grid, CONFIG
        )
        assert isinstance(out, PricingResult)
        prices[label] = out.pv.amount
    assert prices["decaying"] < prices["flat80"] < prices["perfect"]
    # Non-vacuity with teeth: the spread must be a real fraction of the price, not a rounding wobble.
    assert (prices["perfect"] - prices["decaying"]) / prices["perfect"] > 0.05


# ============================================================================================
# ORACLE 3 — THE BOUNDED CROSS-CHECK against Black(Rebonato), with the band PROVEN to have power.
# ============================================================================================


@pytest.mark.parametrize("first", [1, 2, 3])
def test_O3_the_MC_swaption_agrees_with_BLACK_REBONATO_within_the_freezing_slack(first):
    """A **bounded** check, never dressed as exact — McInerney §5.11 proves the LMM swap rate is not
    lognormal, so Black(Rebonato) is an approximation and the gap is the freezing error plus MC
    error. The band is stated in advance (3% of the Black price plus 4 SE) and the measured gap is
    reported in the tranche doc rather than tuned to.

    **The band is proven to have power in the same test**, which is the point §4 insists on: a vol
    wrong by 10% must fall OUTSIDE it. Without that, a wide band is a check that cannot fail."""
    count = 6 - first
    strike = forward_swap_rate(TENOR, first, count)
    model = _model(first, count)
    grid = _grid(first)
    out = price_european_swaption(_swaption(first, count, strike), model, grid, CONFIG)
    assert isinstance(out, PricingResult)

    vol = rebonato_swaption_vol(model, first, count)
    annuity_scale = _black_swaption_scale(first, count)
    closed = black_price(strike, strike, vol**2 * _expiry(first), CapFloorKind.CAP) * annuity_scale
    gap = abs(out.pv.amount - closed) / closed
    assert gap < 0.03, f"MC vs Black(Rebonato) differs by {gap:.3%}, beyond the freezing slack"

    wrong = black_price(strike, strike, (vol * 1.10) ** 2 * _expiry(first), CapFloorKind.CAP) * annuity_scale
    assert abs(out.pv.amount - wrong) / wrong > 0.03, "the band is too wide to fail — no power"


def _black_swaption_scale(first: int, count: int) -> float:
    """`A(0) · notional` — the annuity today, which multiplies the Black swaption formula (§5.10)."""
    bonds = tenor_discount_factors(TENOR)
    annuity = math.fsum(TENOR.accruals[first + m] * bonds[first + m + 1] for m in range(count))
    return annuity * NOTIONAL.amount


# ============================================================================================
# ORACLE 5 — the CORRELATION's magnitude, under COMMON RANDOM NUMBERS.
# ============================================================================================


def test_O5_the_correlation_sensitivity_is_pinned_under_COMMON_RANDOM_NUMBERS():
    """T16's two-numeraire trick does NOT carry over unchanged, and this test says so.

    There, only the drift differed between the two runs, so the diffusion cancelled to the last bit
    and the difference was an exact identity. Here `ρ` enters the **factorisation**, so two models
    differing in `ρ` produce genuinely different paths and nothing cancels exactly. What survives is
    common random numbers: the paired difference is far less noisy than the difference of two
    independent estimates, which is what makes a sensitivity measurable at this path count.

    So the claim is bounded, not exact: the MC price difference under a `ρ` change must agree in
    sign and order of magnitude with Black(Rebonato)'s own `ρ` dependence. That still pins the
    correlation's **magnitude** — the thing presence-and-sign oracles miss — which is what the spec
    asked oracle 5 for."""
    first, count = 2, 4
    strike = forward_swap_rate(TENOR, first, count)
    grid = _grid(first)
    low, high = LmmCorrelation.flat(6, 0.60), LmmCorrelation.flat(6, 0.95)

    def mc(rho) -> float:
        out = price_european_swaption(
            _swaption(first, count, strike), _model(first, count, rho), grid, CONFIG
        )
        assert isinstance(out, PricingResult)
        return out.pv.amount

    def closed(rho) -> float:
        vol = rebonato_swaption_vol(_model(first, count, rho), first, count)
        return (
            black_price(strike, strike, vol**2 * _expiry(first), CapFloorKind.CAP)
            * _black_swaption_scale(first, count)
        )

    mc_move = mc(high) - mc(low)
    closed_move = closed(high) - closed(low)
    assert mc_move > 0.0 and closed_move > 0.0
    assert mc_move == pytest.approx(closed_move, rel=0.10), (
        "the MC correlation sensitivity disagrees with Rebonato's beyond the freezing slack"
    )
    # Premise: the sensitivity must be a real signal, or the agreement above is vacuous.
    assert closed_move / closed(low) > 0.05


# ============================================================================================
# ORACLE 6 — determinism, the product round-trip, and failure-is-a-value.
# ============================================================================================


def test_O6_the_same_seed_gives_the_SAME_price_and_a_different_seed_does_not():
    first, count = 2, 4
    strike = forward_swap_rate(TENOR, first, count)
    args = (_swaption(first, count, strike), _model(first, count), _grid(first))
    a = price_european_swaption(*args, CONFIG)
    b = price_european_swaption(*args, MonteCarloConfig(CONFIG.paths, CONFIG.seed))
    c = price_european_swaption(*args, MonteCarloConfig(CONFIG.paths, CONFIG.seed + 1))
    assert isinstance(a, PricingResult) and isinstance(b, PricingResult)
    assert isinstance(c, PricingResult)
    assert a.pv.amount == b.pv.amount
    assert a.pv.amount != c.pv.amount


def test_O6_a_PAYER_and_a_RECEIVER_satisfy_SWAP_PARITY():
    """`payer − receiver = the forward swap value`, which is zero struck at the forward swap rate.

    A requirement oracle from the finance — it survives a transcription error in the payoff that a
    price comparison would share, and it is the swaption analogue of the put-call parity that caught
    v0.4.0's mis-nested convexity after it had landed."""
    first, count = 2, 4
    strike = forward_swap_rate(TENOR, first, count)
    model, grid = _model(first, count), _grid(first)
    payer = price_european_swaption(_swaption(first, count, strike, SwaptionKind.PAYER), model, grid, CONFIG)
    receiver = price_european_swaption(
        _swaption(first, count, strike, SwaptionKind.RECEIVER), model, grid, CONFIG
    )
    assert isinstance(payer, PricingResult) and isinstance(receiver, PricingResult)
    # Struck AT the forward swap rate the parity difference is zero up to Monte Carlo error, which
    # is a far tighter statement than either price being right.
    assert abs(payer.pv.amount - receiver.pv.amount) / payer.pv.amount < 0.05


def test_O6_the_product_ROUND_TRIPS_through_its_dict():
    swaption = _swaption(2, 4, 0.04, SwaptionKind.RECEIVER)
    assert EuropeanSwaption.from_dict(swaption.to_dict()) == swaption
    with pytest.raises(ValueError, match="not a european_swaption"):
        EuropeanSwaption.from_dict({"kind_": "nominal_cap", "schema_version": 1})


def test_O6_a_BAD_span_or_a_MISMATCHED_numeraire_is_a_PricingFailure_not_a_wrong_number():
    """§3.4 — failure is a value. And the numeraire check is the one that matters: the engine's
    `A/B_N` ratio is only the annuity when `N` is the span's end, so a model under another measure
    would silently return a plausible wrong number. That is refused, not accommodated."""
    strike = 0.04
    good = _model(2, 4)
    assert isinstance(
        price_european_swaption(_swaption(2, 4, strike), good, _grid(2), CONFIG), PricingResult
    )
    wrong_measure = LiborMarketModel(TENOR, RHO, numeraire_index=5)
    out = price_european_swaption(_swaption(2, 4, strike), wrong_measure, _grid(2), CONFIG)
    assert isinstance(out, PricingFailure) and "numeraire" in out.reason
    out = price_european_swaption(_swaption(0, 7, strike), good, _grid(2), CONFIG)
    assert isinstance(out, PricingFailure) and "span" in out.reason
