"""The gamma swap and the corridor variance swap by static replication (L4). KWOK-ZHENG §1.4.

**The source section is §1.4, not ch 4.3, and that is finding one.** T74's spec cites "ch 4.3 (the
generalized variance swaps)" for the static replication; ch 4.3 is *Moment generating function
methods* — a model-based pricing route (the Heston MGF), not a replication. The replication of both
products is **Proposition 1.2 and eq (1.26) in §1.4**, derived there from the `h(F_t)` choices; the
product *payoffs* are §1.3. Transcribed from §1.4 in `-raw` and `-layout`, reconciled, per §5.

**Finding two: the spec understated the gamma strip.** It says "a `1/K`-weighted strip of OTM
vanillas"; eq (1.24) is

    K_g = (2 e^{rT} / (T · F_0)) · [ ∫_0^{F_0} p(K)/K dK + ∫_{F_0}^∞ c(K)/K dK ]

— a `1/K` weight **and** a `1/F_0` normalisation. Building from the spec's words would have been
wrong by a factor of the forward, which is exactly why §5 sends the build to the source. (The
`e^{rT}` undiscounts the source's discounted `p`/`c`; our strip integrates *undiscounted* Black
prices, so it is already absorbed — the same convention T31 uses.)

Eq (1.26), the corridor, is the vanilla `1/K²` strip with the range restricted to the corridor, as
the spec said:

    K_v = (2 e^{rT} / T) · [ ∫_L^{F_0} p(K)/K² dK + ∫_{F_0}^U c(K)/K² dK ]

**Why the `1/F_0` folds into the weight rather than the prefactor.** Written as `p/(K·F_0)` the three
products share one prefactor `2/T`, so the strip walk has exactly one varying part — the weight —
and the vanilla path stays **bit-identical** to T31's `price/(K·K)` (one division either way, no
reordering). Putting `1/F_0` in the prefactor instead would have given the walk two varying parts
and bought nothing.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.variance_replication import fair_variance, generalized_fair_variance
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.equity import (
    DividendSchedule,
    EquityAsset,
    EquityUnderlying,
    dividend_forward,
)
from pricebook.market.equity_skew import ImpliedVolSkew
from pricebook.models.black import black_price
from pricebook.models.variance_replication_model import (
    StrikeGrid,
    VarianceReplicationModel,
    VarianceStrip,
)
from pricebook.products.equity_option import EquityOptionKind
from pricebook.products.generalized_variance_swap import CorridorVarianceSwap, GammaSwap

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
EUR = Currency("EUR")
ACME = EquityAsset("ACME")
DISCOUNT_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
SPOT = 100.0
RATE = 0.03
ATM_VOL = 0.28
#: A real equity skew — vol FALLS with strike. **This is the load-bearing premise of the whole
#: file:** under a FLAT skew the gamma swap equals the vanilla variance swap exactly, so a mutation
#: swapping `1/K` for `1/K²` survives. The skew is what makes them differ (measured: 9.66%), which
#: is §4's `τ=1` shape living in the *skew* rather than an accrual.
SKEW_SLOPE = -0.15
#: A NON-UNIT horizon, so the `2/T` factor is not arithmetically invisible (T31's own recorded trap).
EXPIRY = VAL + timedelta(days=913)
YEARS = 913.0 / 365.0
WIDE = StrikeGrid(lower=0.05, upper=5.0, count=1601)
#: **The gamma strip needs a WIDER grid than the vanilla one for the same accuracy**, and that is a
#: measured property rather than caution. Its `1/(K·F₀)` weight decays one power of `K` more slowly
#: than `1/K²`, so truncating the upper tail at `5·F` costs it far more: on `WIDE` the flat-skew
#: identity (below) holds only to **2.98e-05**, and that gap is **invariant in the node count** —
#: 2.98e-05 at 1601 nodes and at 6401 — so it is truncation, not discretisation. Widening fixes it:
#: 8.8e-09 on `[0.02,10]` and 1.0e-10 on `[0.01,20]`. §4's "a reference is code too, and it needs
#: its own accuracy argument in the regime where it is used" — the regime differs per weight, so the
#: grid T31 measured as adequate for the vanilla strip is **not** adequate for the gamma strip.
UNTRUNCATED = StrikeGrid(lower=0.02, upper=10.0, count=1601)
CURVE = DiscountCurve.flat(VAL, RATE, DC)
UNDERLYING = EquityUnderlying(SPOT, 0.0, DividendSchedule())
FORWARD = dividend_forward(UNDERLYING, EXPIRY, CURVE)
#: A corridor that EXCLUDES REAL MASS — asserted below. A corridor wide enough to capture the whole
#: distribution is `[0,∞)` again, i.e. the reduction wearing the companion's clothes (§4/T39).
CORRIDOR = (0.80 * FORWARD, 1.30 * FORWARD)
N = Money(1_000_000.0, EUR)


def _market(skew: ImpliedVolSkew) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({DISCOUNT_KEY: CURVE}),
        equities={ACME: UNDERLYING},
        equity_skews={ACME: skew},
    )


def _skewed(slope: float = SKEW_SLOPE) -> ImpliedVolSkew:
    return ImpliedVolSkew(ATM_VOL, slope)


def _flat(vol: float = ATM_VOL) -> ImpliedVolSkew:
    return ImpliedVolSkew(vol, 0.0)


def _reference_strip(skew: ImpliedVolSkew, weight, lower: float, upper: float) -> float:
    """The strip, formed independently of the engine: `2/T · ∫ weight(K) · OTM(K) dK` over `(L,U]`.

    Trapezoid on the same nodes, because the quadrature *rule* is not what is under test — the
    weight, the range and the `2/T` factor are. `weight` is written as a plain `1/K**n` here rather
    than reusing the engine's weighting object, which is the §4 independence the transcription needs.
    """
    nodes = [k for k in WIDE.strikes(FORWARD) if lower < k <= upper]
    values = []
    for strike in nodes:
        volatility = skew.volatility(strike, FORWARD, YEARS)
        kind = EquityOptionKind.PUT if strike < FORWARD else EquityOptionKind.CALL
        values.append(
            weight(strike) * black_price(FORWARD, strike, volatility**2 * YEARS, kind)
        )
    area = sum(
        0.5 * (values[i] + values[i + 1]) * (nodes[i + 1] - nodes[i])
        for i in range(len(nodes) - 1)
    )
    return 2.0 / YEARS * area


# ============================================================================================
# 0. The premises this file's power rests on, asserted rather than assumed (§4).
# ============================================================================================

def test_the_SKEW_premise_a_flat_surface_makes_the_two_strips_IDENTICAL():
    """Under a flat surface the gamma swap IS the vanilla variance swap, so every gamma oracle below
    must run on a skewed surface or the `1/K` weight is invisible.

    The identity is exact in the continuum — `F_t` is a martingale, so `E[∫(F_t/F_0)σ²dt]/T = σ²`
    just as `E[∫σ²dt]/T` does — and it is checked on `UNTRUNCATED` rather than `WIDE` because the
    two strips have **different truncation errors** at the same bounds (see `UNTRUNCATED`). Running
    it on `WIDE` would band a real identity at 3e-05 for a reason that has nothing to do with the
    weight it is testing."""
    flat_vanilla = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _flat(), VarianceStrip.vanilla(UNTRUNCATED)
    )
    flat_gamma = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _flat(), VarianceStrip.gamma(UNTRUNCATED)
    )
    assert flat_gamma == pytest.approx(flat_vanilla, rel=1e-7, abs=0.0)

    skewed_vanilla = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.vanilla(WIDE)
    )
    skewed_gamma = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.gamma(WIDE)
    )
    assert abs(skewed_gamma / skewed_vanilla - 1.0) > 0.05, (skewed_gamma, skewed_vanilla)


def test_the_flat_skew_gap_on_a_TRUNCATED_grid_is_truncation_and_NOT_discretisation():
    """The diagnosis behind `UNTRUNCATED`, asserted rather than asserted-about.

    A gap that does not move when the node count quadruples is not a quadrature-step error; one that
    falls when the bounds widen is a truncation error. Both halves are measured here, which is what
    turns "use a wider grid" from a fudge into a stated validity regime (§4/T25)."""
    def gap(grid: StrikeGrid) -> float:
        vanilla = generalized_fair_variance(
            UNDERLYING, EXPIRY, CURVE, _flat(), VarianceStrip.vanilla(grid)
        )
        gamma = generalized_fair_variance(
            UNDERLYING, EXPIRY, CURVE, _flat(), VarianceStrip.gamma(grid)
        )
        return abs(gamma / vanilla - 1.0)

    coarse, fine = gap(StrikeGrid(0.05, 5.0, 1601)), gap(StrikeGrid(0.05, 5.0, 6401))
    assert coarse == pytest.approx(fine, rel=1e-3, abs=0.0), (coarse, fine)   # count: no effect
    assert gap(StrikeGrid(0.02, 10.0, 1601)) < 0.01 * coarse                  # width: it collapses


def test_the_CORRIDOR_premise_it_excludes_real_mass():
    # A corridor that captures everything is `[0,inf)` in disguise, and the restriction — the whole
    # product — would be invisible.
    lower, upper = CORRIDOR
    restricted = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.corridor(WIDE, lower, upper)
    )
    whole = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.vanilla(WIDE)
    )
    assert restricted < 0.9 * whole, (restricted, whole)


# ============================================================================================
# 1. The pointwise transcriptions — eq (1.24) and eq (1.26).
# ============================================================================================

def test_the_GAMMA_strip_matches_eq_1_24_pointwise():
    # K_g = 2/(T·F_0) · [∫_0^{F_0} p/K dK + ∫_{F_0}^∞ c/K dK], the `1/F_0` folded into the weight.
    want = _reference_strip(_skewed(), lambda k: 1.0 / (k * FORWARD), 0.0, math.inf)
    got = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.gamma(WIDE)
    )
    assert got == pytest.approx(want, rel=1e-14, abs=0.0)


def test_the_CORRIDOR_strip_matches_eq_1_26_pointwise():
    lower, upper = CORRIDOR
    want = _reference_strip(_skewed(), lambda k: 1.0 / (k * k), lower, upper)
    got = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.corridor(WIDE, lower, upper)
    )
    assert got == pytest.approx(want, rel=1e-14, abs=0.0)


def test_the_two_WEIGHTS_really_differ_so_the_transcriptions_are_not_one_test_twice():
    gamma = _reference_strip(_skewed(), lambda k: 1.0 / (k * FORWARD), 0.0, math.inf)
    vanilla = _reference_strip(_skewed(), lambda k: 1.0 / (k * k), 0.0, math.inf)
    assert abs(gamma / vanilla - 1.0) > 0.05


# ============================================================================================
# 2. The reduction, and the companion §4 requires beside it.
# ============================================================================================

def test_the_corridor_at_the_FULL_RANGE_is_the_vanilla_variance_swap_BIT_IDENTICALLY():
    # The anchor: an unrestricted corridor is the vanilla strip, and `VarianceStrip.vanilla` is the
    # same walk with the same weight — so this is exact, not banded. It is also a TAUTOLOGY on its
    # own (T70), which is why the finite-corridor companion above carries the correctness.
    unrestricted = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.corridor(WIDE, 0.0, math.inf)
    )
    assert unrestricted == fair_variance(UNDERLYING, EXPIRY, CURVE, _skewed(), WIDE)


def test_the_vanilla_path_is_UNCHANGED_by_the_extraction():
    # T31's entry point still produces T31's number, through the generalized walk.
    assert fair_variance(UNDERLYING, EXPIRY, CURVE, _skewed(), WIDE) == (
        generalized_fair_variance(
            UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.vanilla(WIDE)
        )
    )


# ============================================================================================
# 3. Requirement oracles — what the finance forces for ALL parameters.
# ============================================================================================

@pytest.mark.parametrize("width", [0.05, 0.15, 0.30, 0.60])
def test_restricting_the_range_can_only_LOWER_the_accrued_variance(width: float):
    lower, upper = (1.0 - width) * FORWARD, (1.0 + width) * FORWARD
    restricted = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.corridor(WIDE, lower, upper)
    )
    whole = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.vanilla(WIDE)
    )
    assert 0.0 < restricted < whole


def test_a_WIDER_corridor_accrues_MORE_than_a_narrower_one_nested_inside_it():
    values = [
        generalized_fair_variance(
            UNDERLYING, EXPIRY, CURVE, _skewed(),
            VarianceStrip.corridor(WIDE, (1.0 - w) * FORWARD, (1.0 + w) * FORWARD),
        )
        for w in (0.05, 0.15, 0.30, 0.60)
    ]
    assert values == sorted(values), values


def test_all_three_strips_are_POSITIVE_and_the_gamma_is_damped_under_a_falling_skew():
    # The finance: a negative skew puts the extra variance in the LOW strikes, and the gamma weight
    # `1/K` discounts exactly those relative to `1/K²` — so the gamma strike sits BELOW the vanilla.
    # A sign, not a magnitude, which is all the finance forces (§4 obligation-not-form).
    vanilla = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.vanilla(WIDE)
    )
    gamma = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.gamma(WIDE)
    )
    assert 0.0 < gamma < vanilla


# ============================================================================================
# 4. The engines.
# ============================================================================================

def _model(skew: ImpliedVolSkew = None) -> VarianceReplicationModel:
    return VarianceReplicationModel(_market(skew or _skewed()), WIDE)


def _pv(product: object) -> float:
    out = price(product, _model())
    assert isinstance(out, PricingResult), out
    return out.pv.amount


def test_the_GAMMA_SWAP_prices_as_the_discounted_variance_difference():
    strike = 0.06
    swap = GammaSwap(ACME, EXPIRY, strike, N)
    fair = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.gamma(WIDE)
    )
    want = N.amount * CURVE.df(EXPIRY) * (fair - strike)
    assert _pv(swap) == pytest.approx(want, rel=1e-14, abs=0.0)


def test_the_CORRIDOR_SWAP_prices_as_the_discounted_variance_difference():
    strike = 0.04
    lower, upper = CORRIDOR
    swap = CorridorVarianceSwap(ACME, EXPIRY, strike, N, CORRIDOR)
    fair = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.corridor(WIDE, lower, upper)
    )
    want = N.amount * CURVE.df(EXPIRY) * (fair - strike)
    assert _pv(swap) == pytest.approx(want, rel=1e-14, abs=0.0)


def test_a_MATURED_swap_is_PV_ZERO_for_both_invariant_6():
    after = EXPIRY + timedelta(days=1)
    market = MarketSnapshot(
        valuation_date=after,
        curves=CurveSet({DISCOUNT_KEY: DiscountCurve.flat(after, RATE, DC)}),
        equities={ACME: UNDERLYING},
        equity_skews={ACME: _skewed()},
    )
    model = VarianceReplicationModel(market, WIDE)
    for product in (
        GammaSwap(ACME, EXPIRY, 0.06, N),
        CorridorVarianceSwap(ACME, EXPIRY, 0.06, N, CORRIDOR),
    ):
        out = price(product, model)
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == 0.0


def test_a_MISSING_underlying_or_skew_is_a_PricingFailure_for_both():
    bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({DISCOUNT_KEY: CURVE}))
    model = VarianceReplicationModel(bare, WIDE)
    for product in (
        GammaSwap(ACME, EXPIRY, 0.06, N),
        CorridorVarianceSwap(ACME, EXPIRY, 0.06, N, CORRIDOR),
    ):
        assert isinstance(price(product, model), PricingFailure)


def test_an_off_forward_corridor_is_a_PricingFailure_and_NOT_a_raise_through_the_engine():
    """§3.4 — and the first draft of the engine got this wrong.

    `VarianceStrip.restricted` refuses a corridor that does not contain the forward, and
    `_price_generalized` originally caught only `KeyError`, so that refusal propagated as a
    `ValueError` **out of a registered pricer**. Failure is a value. This is also the guard's second
    independent killer: it reaches the refusal through the engine rather than through the direct
    call below, so neither test's deletion reopens the hole."""
    swap = CorridorVarianceSwap(ACME, EXPIRY, 0.05, N, (1.5 * FORWARD, 2.0 * FORWARD))
    out = price(swap, _model())
    assert isinstance(out, PricingFailure), out
    assert "forward" in out.reason


def test_the_gamma_swap_and_the_vanilla_swap_DISAGREE_at_the_same_strike():
    """The gamma pricer's second independent killer: a cross-product check.

    Its sibling asserts the gamma PV equals the gamma strip's discounted difference; this asserts it
    is **not** the vanilla swap's PV at the same strike — so swapping the strip at the pricer is
    caught by a test that never computes the gamma strip itself. Under the skewed fixture the two
    fair variances differ by ~9.7% (the premise above), which is what makes this discriminate."""
    from pricebook.products import VarianceSwap

    strike = 0.06
    gamma = price(GammaSwap(ACME, EXPIRY, strike, N), _model())
    vanilla = price(VarianceSwap(ACME, EXPIRY, strike, N), _model())
    assert isinstance(gamma, PricingResult) and isinstance(vanilla, PricingResult)
    assert abs(gamma.pv.amount / vanilla.pv.amount - 1.0) > 0.1, (gamma.pv, vanilla.pv)


def test_the_corridor_is_HALF_OPEN_and_the_fixture_puts_a_grid_node_on_each_bound():
    """`(L,U]` — the lower bound excluded, the upper included (§1.3).

    **The premise is the whole test:** the half-openness is observable only when a strike lands
    *exactly* on a bound, and no other fixture here does — so a mutation flipping `L < k <= U` to
    `L <= k < U` survived the whole suite. The bounds are therefore taken **from the grid's own
    nodes**, which makes the two conventions select different strike sets and different values."""
    nodes = WIDE.strikes(FORWARD)
    # Chosen RELATIVE to the forward, not by index: the grid spans 0.05x..5x, so the forward sits
    # near node 307 and a fixed index is a fixture that silently moves if the grid's bounds change.
    below = [k for k in nodes if k < FORWARD]
    above = [k for k in nodes if k >= FORWARD]
    lower, upper = below[len(below) // 2], above[len(above) // 2]
    assert lower < FORWARD <= upper, (lower, FORWARD, upper)

    selected = VarianceStrip.corridor(WIDE, lower, upper).restricted(FORWARD)
    assert lower not in selected and upper in selected, (selected[0], selected[-1])
    # and the flipped convention would select a DIFFERENT set, which is why this discriminates
    flipped = tuple(k for k in nodes if lower <= k < upper)
    assert flipped != selected and len(flipped) == len(selected)


def _on_grid_bounds() -> tuple[float, float]:
    """A corridor whose bounds are grid nodes straddling the forward — the only fixture in which
    the half-openness is observable at all."""
    nodes = WIDE.strikes(FORWARD)
    below = [k for k in nodes if k < FORWARD]
    above = [k for k in nodes if k >= FORWARD]
    return below[len(below) // 2], above[len(above) // 2]


def test_the_corridor_strip_matches_eq_1_26_pointwise_ON_GRID_BOUNDS_too():
    """The half-openness convention's **second** killer, and the right shape for it.

    The pointwise transcription above uses an off-grid corridor, where no strike lands on a bound and
    `(L,U]` versus `[L,U)` select the same set — so it cannot see the convention. Run with bounds
    **on grid nodes**, the reference's own `L < k <= U` filter and a flipped implementation select
    different sets and produce different numbers, so this kills the flip independently of the
    node-set test above. The first attempt here asserted that shifting both bounds moves the value,
    which does not discriminate at all: *both* conventions move when the bounds move."""
    lower, upper = _on_grid_bounds()
    want = _reference_strip(_skewed(), lambda k: 1.0 / (k * k), lower, upper)
    got = generalized_fair_variance(
        UNDERLYING, EXPIRY, CURVE, _skewed(), VarianceStrip.corridor(WIDE, lower, upper)
    )
    assert got == pytest.approx(want, rel=1e-14, abs=0.0)


def test_a_corridor_that_EXCLUDES_THE_FORWARD_is_refused_the_source_assumes_it_inside():
    # §1.4: "It is reasonable to choose the corridor so that F0 lies within the corridor" — the
    # replication splits the strip AT the forward, so a corridor entirely above or below it would
    # silently integrate one empty half. Refused rather than returning a plausible number.
    with pytest.raises(ValueError, match="forward"):
        generalized_fair_variance(
            UNDERLYING, EXPIRY, CURVE, _skewed(),
            VarianceStrip.corridor(WIDE, 1.5 * FORWARD, 2.0 * FORWARD),
        )


# ============================================================================================
# 5. The products themselves — pure data (KWOK-ZHENG §1.3).
# ============================================================================================

def _gamma_product() -> GammaSwap:
    return GammaSwap(ACME, EXPIRY, 0.0625, N)


def _corridor_product() -> CorridorVarianceSwap:
    return CorridorVarianceSwap(ACME, EXPIRY, 0.0484, N, CORRIDOR)


def test_both_products_ROUND_TRIP_through_their_payloads():
    assert GammaSwap.from_dict(_gamma_product().to_dict()) == _gamma_product()
    assert CorridorVarianceSwap.from_dict(_corridor_product().to_dict()) == _corridor_product()


def test_the_payload_carries_the_CORRIDOR_which_is_constitutive_of_the_contract():
    # A corridor swap without its corridor is a different trade, so the round trip must carry it
    # (§6: a carried field is allowed when constitutive of identity, and is then exercised).
    payload = _corridor_product().to_dict()
    assert payload["corridor"] == list(CORRIDOR)
    assert CorridorVarianceSwap.from_dict(payload).corridor == CORRIDOR


def test_a_payload_of_the_WRONG_kind_is_refused_by_each():
    with pytest.raises(ValueError, match="not a gamma_swap"):
        GammaSwap.from_dict(_corridor_product().to_dict())
    with pytest.raises(ValueError, match="not a corridor_variance_swap"):
        CorridorVarianceSwap.from_dict(_gamma_product().to_dict())


@pytest.mark.parametrize("strike", [-1e-9, -0.05])
def test_a_NEGATIVE_variance_strike_is_refused(strike: float):
    with pytest.raises(ValueError, match="variance strike"):
        GammaSwap(ACME, EXPIRY, strike, N)
    with pytest.raises(ValueError, match="variance strike"):
        CorridorVarianceSwap(ACME, EXPIRY, strike, N, CORRIDOR)


@pytest.mark.parametrize("corridor", [(-1.0, 130.0), (130.0, 80.0), (100.0, 100.0)])
def test_an_INVALID_corridor_is_refused(corridor: tuple[float, float]):
    with pytest.raises(ValueError, match="corridor"):
        CorridorVarianceSwap(ACME, EXPIRY, 0.05, N, corridor)


def test_a_ZERO_lower_bound_is_ACCEPTED_because_the_source_allows_it():
    # §1.3's corridor is `(L,U]` with `L >= 0`, so a one-sided corridor is a real contract.
    assert CorridorVarianceSwap(ACME, EXPIRY, 0.05, N, (0.0, 130.0)).corridor == (0.0, 130.0)


def test_the_two_products_are_DISTINCT_types_with_no_hierarchy_against_the_vanilla_swap():
    from pricebook.products import VarianceSwap

    assert not issubclass(GammaSwap, (VarianceSwap, CorridorVarianceSwap))
    assert not issubclass(CorridorVarianceSwap, (VarianceSwap, GammaSwap))
    assert not issubclass(VarianceSwap, (GammaSwap, CorridorVarianceSwap))


def test_a_STRIP_with_an_inverted_range_is_refused_at_construction():
    with pytest.raises(ValueError, match="upper bound"):
        VarianceStrip.corridor(WIDE, 130.0, 80.0)
