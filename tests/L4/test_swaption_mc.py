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

from pricebook.engine.mc_path_engine import PathGrid, mc_caplet_price, simulate_terminal_forwards
from pricebook.engine.swaption_mc import price_european_swaption
from pricebook.foundation.inflation import CapFloorKind
from pricebook.foundation import Currency
from pricebook.foundation.mc_estimator import estimate
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

# **NON-UNIFORM accruals and a correlation built on real TIMES, and both are deliberate.** Three
# degeneracies have to be excluded here, not two:
#   * `τ = 1` makes the accrual invisible (five prior defects in this project), so no `τ` is 1;
#   * a FLAT `τ` makes the span OFFSET invisible — `accruals[:count]` and `accruals[first:first+count]`
#     are then the same slice, and a mutation dropping the offset survives;
#   * a correlation `exp(−β|i−j|)` is non-uniform yet TRANSLATION-INVARIANT, so `ρ[first+m][first+l]`
#     equals `ρ[m][l]` and the span offset is invisible there too. Jäckel eq 12.12 writes the decay in
#     the actual times `t_i`, not in index distance; with non-uniform accruals that is no longer
#     translation-invariant, which is both the faithful reading and the non-degenerate one.
# All three were found by mutation, not by inspection — two mutants survived the whole suite first.
TENOR = LmmTenor(
    accruals=(0.50, 0.25, 0.50, 0.75, 0.50, 0.25),
    forwards=(0.030, 0.034, 0.038, 0.042, 0.045, 0.047),
    volatilities=(0.22, 0.20, 0.18, 0.16, 0.15, 0.14),
)
#: `T_i`, the cumulative accruals — the times Jäckel's eq 12.12 decays in.
TIMES = tuple(math.fsum(TENOR.accruals[:i]) for i in range(len(TENOR)))
RHO = LmmCorrelation(
    tuple(tuple(math.exp(-0.35 * abs(TIMES[i] - TIMES[j])) for j in range(6)) for i in range(6))
)
EUR = Currency("EUR")
NOTIONAL = Money(1_000_000.0, EUR)
# 6,000 paths and 8 steps, and the split matters. The `slow` gate's ceiling is 10s and CI measured
# **4.3x** this machine in tranche 16 (2.16s -> 9.28s on Windows), so this file needs real margin.
# The reduction is taken entirely in PATHS, never in steps: the path count buys only precision, and
# every oracle below states its margin at this count, whereas the step count sets the model's
# discretisation fidelity — trading that for speed would be changing the answer to save time.
CONFIG = MonteCarloConfig(paths=6_000, seed=20261106)


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


def _estimate_from_paths(first: int, count: int, strike: float, kind=SwaptionKind.PAYER,
                         rho: LmmCorrelation = RHO, config: MonteCarloConfig = CONFIG):
    """The mean AND standard error of the swaption payoff, from the simulation the engine consumes.

    The engine returns a `PricingResult`, which carries a PV and no standard error — correct for the
    charter's engine contract, and insufficient for banding an oracle. So the payoff is recomputed
    here from the same paths, which costs one extra simulation and buys two things: a measured SE,
    and a check that the engine's own composition (`mean x discount x notional`) is what it claims."""
    paths = simulate_terminal_forwards(_model(first, count, rho), _grid(first), config)
    tau = TENOR.accruals
    payoffs = []
    for forwards in paths:
        bonds, running = [], 1.0
        for m in range(count):
            running /= 1.0 + forwards[first + m] * tau[first + m]
            bonds.append(running)
        annuity = math.fsum(tau[first + m] * bonds[m] for m in range(count))
        swap_rate = (1.0 - bonds[-1]) / annuity
        payoffs.append(
            annuity / bonds[-1] * max(kind.omega * (swap_rate - strike), 0.0)
        )
    return estimate(payoffs)


#: The freezing error of Rebonato's approximation, MEASURED at 200,000 paths where sampling error is
#: negligible: the MC price sits 0.15% to 0.96% from Black(Rebonato) depending on the step count,
#: settling near 0.5% as the discretisation refines. Named as a constant so the oracle's band is
#: `4 SE + this`, rather than one flat percentage standing in for two different errors.
_FREEZING_SLACK = 0.015


@pytest.mark.parametrize("first", [1, 2, 3])
def test_O3_the_MC_swaption_agrees_with_BLACK_REBONATO_within_SE_plus_the_freezing_slack(first):
    """A **bounded** check, with the two error sources kept apart rather than lumped together.

    §5.11 proves the LMM swap rate is not lognormal, so Black(Rebonato) is an approximation and the
    gap has two parts: the **freezing error**, measured at 0.5% where sampling error is negligible,
    and **Monte Carlo error** at the committed path count, which at 6,000 paths is the larger of the
    two. An earlier version of this oracle asserted one flat 3% and called it "the freezing slack" —
    which was wrong about which error it was accommodating, and would have let a real freezing
    defect of 2% pass as sampling noise.

    **This case's RESOLUTION is stated rather than implied, and it is coarse.** At 6,000 paths the
    band is dominated by `4 SE`, and it resolves only a vol error of ~20% or more — so these three
    spans are a *breadth* check against a gross structural error, which is what they caught in
    mutation (the numeraire bond, the discount index, the annuity ratio, `ω`). The oracle's real
    power is established by the test below, at a path count where the band is narrow enough to have
    some. Saying so is the point: a green here is evidence about gross errors and nothing finer."""
    count = 6 - first
    strike = forward_swap_rate(TENOR, first, count)
    model = _model(first, count)
    out = price_european_swaption(_swaption(first, count, strike), model, _grid(first), CONFIG)
    assert isinstance(out, PricingResult)

    scale = tenor_discount_factors(TENOR)[first + count] * NOTIONAL.amount
    sampled = _estimate_from_paths(first, count, strike)
    # The engine's composition, pinned: PV is the payoff mean times the numeraire bond and notional.
    assert out.pv.amount == pytest.approx(sampled.mean * scale, rel=1e-12)

    vol = rebonato_swaption_vol(model, first, count)
    closed = (
        black_price(strike, strike, vol**2 * _expiry(first), CapFloorKind.CAP)
        * _black_swaption_scale(first, count)
    )
    band = 4.0 * sampled.standard_error * scale + _FREEZING_SLACK * closed
    assert abs(out.pv.amount - closed) < band


def test_O3_the_band_HAS_POWER_at_the_path_count_where_it_is_narrow_enough():
    """The companion the breadth check needs, and the reason it is a separate test.

    A band is only evidence if it can fail, and that is a function of the path count — which the
    `slow` gate caps. Measured: the band resolves a vol error of ~20% at 6,000 paths, **10% at
    12,000**, and 5% at 25,000. So the power demonstration runs once, on one span, at 12,000 paths,
    where a 10% vol error provably falls outside; paying 25,000 everywhere would buy a finer
    resolution than the gate's time ceiling allows.

    This test exists because the breadth check's first version asserted power at 6,000 paths and
    **failed** — the band was wider than the 10% error it claimed to exclude. That failure is the
    finding: without an explicit power assertion the three green breadth cases would have looked
    like a 10%-accurate cross-check, and they are not."""
    first, count = 2, 4
    powerful = MonteCarloConfig(paths=12_000, seed=CONFIG.seed)
    strike = forward_swap_rate(TENOR, first, count)
    model = _model(first, count)
    out = price_european_swaption(
        _swaption(first, count, strike), model, _grid(first), powerful
    )
    assert isinstance(out, PricingResult)

    scale = tenor_discount_factors(TENOR)[first + count] * NOTIONAL.amount
    sampled = _estimate_from_paths(first, count, strike, config=powerful)
    vol = rebonato_swaption_vol(model, first, count)
    closed = (
        black_price(strike, strike, vol**2 * _expiry(first), CapFloorKind.CAP)
        * _black_swaption_scale(first, count)
    )
    band = 4.0 * sampled.standard_error * scale + _FREEZING_SLACK * closed
    assert abs(out.pv.amount - closed) < band

    wrong = black_price(
        strike, strike, (vol * 1.10) ** 2 * _expiry(first), CapFloorKind.CAP
    ) * _black_swaption_scale(first, count)
    assert abs(out.pv.amount - wrong) > band, "a 10% vol error sits INSIDE the band — no power"


def _black_swaption_scale(first: int, count: int) -> float:
    """`A(0) · notional` — the annuity today, which multiplies the Black swaption formula (§5.10)."""
    bonds = tenor_discount_factors(TENOR)
    annuity = math.fsum(TENOR.accruals[first + m] * bonds[first + m + 1] for m in range(count))
    return annuity * NOTIONAL.amount


# ============================================================================================
# ORACLE 5 — the CORRELATION's magnitude, under COMMON RANDOM NUMBERS.
# ============================================================================================


def test_O5_every_correlation_PAIR_is_pinned_by_the_REALISED_covariance():
    """The correlation's MAGNITUDE, pair by pair — and this replaces what the spec asked for, because
    what it asked for is not available.

    **Why the specced instrument does not carry over.** T16 pinned the drift noise-free by running
    two numeraires on one seed: only the drift differed, so the diffusion cancelled to the last bit.
    Oracle 5 was specced as the same trick on `ρ`. But `ρ` enters the **factorisation** `A` of
    `C = σσρ·dt`, so two models differing in `ρ` produce genuinely different paths and nothing
    cancels exactly. The identity is simply not there to exploit.

    **What is available is stronger than the price-level claim anyway.** `ρ` enters the engine in
    exactly one place, and that place is directly observable: the realised covariance of the
    simulated log-increments must reproduce `σ_i σ_j ρ_ij dt` for **every pair**. That pins all 36
    entries individually, where a swaption price pins only one correlation-weighted scalar. A `ρ`
    transposed, offset, or scaled would show here and could hide inside a single price.

    The error is Monte Carlo and is bounded by the estimator's own rate, not by a tuned number: the
    relative SE of a covariance entry is `≈ √((1+ρ²)/N)`, about 0.8% at 20,000 paths, and the worst
    of 36 correlated pairs runs a small multiple of that. Asserted at 5%, measured at 3.2%, and the
    `−½` rate is asserted by quadrupling the paths rather than assumed."""
    model = _model(2, 4)
    grid = PathGrid(horizon=1.0, steps=1)

    def worst_pairwise_error(paths: int) -> float:
        drawn = simulate_terminal_forwards(model, grid, MonteCarloConfig(paths, 4242))
        increments = [
            [math.log(path[i] / TENOR.forwards[i]) for i in range(len(TENOR))] for path in drawn
        ]
        means = [
            math.fsum(row[i] for row in increments) / paths for i in range(len(TENOR))
        ]
        worst = 0.0
        for i in range(len(TENOR)):
            for j in range(len(TENOR)):
                realised = math.fsum(
                    (row[i] - means[i]) * (row[j] - means[j]) for row in increments
                ) / (paths - 1)
                # The target is transcribed here from eq 12.3, not read from the engine.
                target = TENOR.volatilities[i] * TENOR.volatilities[j] * RHO.matrix[i][j] * grid.dt
                worst = max(worst, abs(realised - target) / abs(target))
        return worst

    coarse = worst_pairwise_error(20_000)
    fine = worst_pairwise_error(80_000)
    assert coarse < 0.05, f"the realised covariance misses its target by {coarse:.3%}"
    assert fine < coarse / 1.5, (
        f"quadrupling the paths took the worst pair from {coarse:.3%} to {fine:.3%}, which is not "
        f"the -1/2 rate — a BIAS in the factorisation, not sampling error"
    )


def test_O5_the_price_level_correlation_sensitivity_agrees_within_the_DERIVED_slack():
    """The price-level companion, with its band derived rather than guessed — and it is loose.

    A difference of two prices amplifies relative error: each price carries up to the 3% freezing
    slack oracle 3 measures, the two need not err in the same direction, and the move itself is only
    ~5% of the price. So the finance permits the move to be wrong by up to `(3% + 3%)/5% ≈ 120%`,
    and any tighter assertion here would be a number tuned to pass rather than a bound. Measured at
    13.9%, asserted at 50% — which still rules out a sign error or a factor-of-two misweighting,
    and nothing finer. The magnitude is pinned by the pairwise oracle above, not by this."""
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
    assert mc_move == pytest.approx(closed_move, rel=0.50)
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
