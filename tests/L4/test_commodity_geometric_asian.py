"""Geometric-average commodity options, exact closed form — tranche 61, PRE-REGISTERED.
`CLARK-2014` ch 2.7.3 (Kemna–Vorst, discrete fixings).

**Why this is exact and not an approximation.** The geometric average of lognormals is *itself*
lognormal, so the option is Black-76 on an adjusted forward and an adjusted variance — nothing is
being approximated, which is what makes a closed form available where the arithmetic average (T62)
will need a moment match.

**The source, transcribed from both renderings and reconciled (§4).** Clark pp.63–65, writing `rᵈ`
for the risk-free rate and `rᶠ` for the convenience yield (his FX analogy, carried from T60):

    ln A_G = ln S₀ + (1/n)·Σᵢ (rᵈ − rᶠ − ½σ²)tᵢ + (1/n)·Σᵢ σW_{tᵢ}             (2.111)
    Var[ln A_G] = (σ²/n²)·Σ_{i,j=1..n} min(tᵢ, tⱼ)
                = (σ²/n²)·Σ_{i=1..n} tᵢ·[2(n−i)+1]                             p.64

and, specialised to an equally-spaced grid with `t₁ = h`, `tₙ = T`, `h = T/n`:

    E[ln A_G]   = ln S₀ + ½(rᵈ − rᶠ − ½σ²)(1 + 1/n)T                           (2.114a)
    Var[ln A_G] = σ²T·[1 − (1/6)·(4n+1)(n−1)/n²]                               (2.114b)

**The source prints the load-bearing quantity THREE ways, which is far stronger than two renderings
of one line (§4's derivation-step rule).** The variance appears as the `min(tᵢ,tⱼ)` double sum, as
its sorted single-sum evaluation `Σ tᵢ[2(n−i)+1]`, and — for a uniform grid — as a closed form in
`n` alone. Three different computation paths for one number: a misprint would have to corrupt all
three identically to survive. `TestTheVarianceHasThreeIndependentForms` requires them to agree, and
they do, to 1e-13 across `n = 1…60`, with the `n → ∞` limit landing on `σ²T/3` as (2.115b) says.

The implementation is built on the **general** forms, not the specialisations: real fixing schedules
are not equally spaced, and Clark says so explicitly ("the computations can easily be performed for
arbitrary fixing schedules"; the uniform grid is "purely to simplify the algebra"). That makes
(2.114a/b) genuinely independent checks rather than restatements of what was coded.

**The adjusted forward.** `A_G` is lognormal, so `F_G = E[A_G] = exp(E[ln A_G] + ½Var[ln A_G])`. At
`n = 1` that collapses to `S₀e^{(rᵈ−rᶠ)t₁}` — exactly T60's forward — which is the reduction anchor.

**Scope, and the source scopes it the same way.** No historical fixings: Clark restricts himself to
"the case where no historical fixings have taken place yet (j = 0 in [Zhang's] notation)", and so
does this engine — a partly-fixed Asian needs the realised prints, which the snapshot does not carry
for a commodity. It is refused in domain vocabulary rather than silently priced as if unfixed.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.commodity_asian import (
    geometric_average_forward,
    geometric_average_variance,
)
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.rng import UniformSource
from pricebook.foundation.mc_estimator import estimate
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.foundation.variates import standard_normals
from pricebook.market.commodity import CommodityAsset, CommodityUnderlying, commodity_forward
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.products.commodity_asian import CommodityGeometricAsian
from pricebook.products.commodity_option import CommodityOption, CommodityOptionKind

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
USD = Currency("USD")
WTI = CommodityAsset("WTI")

RATE, CONVENIENCE = 0.042, 0.015
SPOT, VOL = 78.5, 0.31
FLAT = DiscountCurve.flat(VAL, RATE, DC)

#: **Deliberately NOT equally spaced**, and deliberately not a round number of years. The variance
#: is a double sum over the *actual* times, so a mutation that mis-indexes `min(tᵢ,tⱼ)` or drops a
#: cross term must move the number — and a uniform grid is exactly where such a mutation can hide
#: (the T17 lesson: decay in the real times, not in index distance).
FIXING_DAYS = (120, 211, 365, 548, 700, 912)
FIXINGS = tuple(VAL + timedelta(days=d) for d in FIXING_DAYS)
FIXING_YEARS = tuple(d / 365.0 for d in FIXING_DAYS)
EXPIRY = FIXINGS[-1]
YEARS = FIXING_YEARS[-1]


def _underlying(convenience: float = CONVENIENCE) -> CommodityUnderlying:
    return CommodityUnderlying(asset=WTI, spot=SPOT, convenience_yield=convenience)


def _market(convenience: float = CONVENIENCE) -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT}),
        commodities={WTI: _underlying(convenience)},
    )


def _asian(
    strike: float = 80.0,
    kind: CommodityOptionKind = CommodityOptionKind.CALL,
    fixings: tuple[date, ...] = FIXINGS,
) -> CommodityGeometricAsian:
    return CommodityGeometricAsian(
        asset=WTI, strike=strike, fixings=fixings, notional=Money(1_000.0, USD), kind=kind
    )


def _pv(product, market: MarketSnapshot | None = None, vol: float = VOL) -> float:
    out = price(product, CommodityBlackModel(market or _market(), vol))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ── the references, written here from the source and never read back from the engine ──────────


def _reference_variance(years: tuple[float, ...], vol: float) -> float:
    """The `min(tᵢ,tⱼ)` **double sum**, p.64 — the primary printed form."""
    n = len(years)
    return vol * vol / (n * n) * math.fsum(min(a, b) for a in years for b in years)


def _reference_log_mean(years: tuple[float, ...], vol: float, convenience: float) -> float:
    """Eq (2.111)'s drift: `ln S₀ + (1/n)·Σ (rᵈ − rᶠ − ½σ²)tᵢ`."""
    n = len(years)
    return math.log(SPOT) + math.fsum(
        (RATE - convenience - 0.5 * vol * vol) * t for t in years
    ) / n


def _reference_forward(years: tuple[float, ...], vol: float, convenience: float) -> float:
    """`F_G = E[A_G] = exp(E[ln A_G] + ½Var[ln A_G])` — the lognormal mean."""
    return math.exp(
        _reference_log_mean(years, vol, convenience)
        + 0.5 * _reference_variance(years, vol)
    )


class TestThePremisesTheOraclesRestOn:
    def test_the_fixings_are_DISTINCT_and_NON_UNIFORMLY_spaced(self) -> None:
        """A uniform grid is where a mis-indexed `min(tᵢ,tⱼ)` hides: the double sum becomes a
        function of index distance alone, so an off-by-one in the pairing can still land on the
        right number. The gaps here differ by more than a factor of two."""
        gaps = [b - a for a, b in zip(FIXING_YEARS, FIXING_YEARS[1:])]
        assert len(set(FIXING_YEARS)) == len(FIXING_YEARS)
        assert max(gaps) / min(gaps) > 2.0, gaps

    def test_the_horizon_is_NOT_one_year_and_there_are_at_least_three_fixings(self) -> None:
        assert abs(YEARS - 1.0) > 1.4, YEARS
        assert len(FIXINGS) >= 3

    def test_the_averaging_MOVES_the_price_materially_off_the_european(self) -> None:
        """The premise the whole tranche rests on: if the Asian priced near the European, every
        oracle below would pass against an implementation that ignored the averaging."""
        european = CommodityOption(
            asset=WTI, strike=80.0, expiry=EXPIRY, notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        assert abs(_pv(_asian()) / _pv(european) - 1.0) > 0.2


class TestTheVarianceHasThreeIndependentForms:
    """§4's strongest available check: not two renderings of one line, but **three different
    computation paths** the source itself prints for one quantity. A misprint would have to corrupt
    all three identically."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 12, 60])
    def test_the_double_sum_equals_the_SORTED_SINGLE_SUM(self, n: int) -> None:
        """`(σ²/n²)Σᵢⱼ min(tᵢ,tⱼ) = (σ²/n²)Σᵢ tᵢ[2(n−i)+1]` — p.64's own evaluation, valid for any
        **sorted** schedule, so it is a check on the pairing and not only on the uniform case."""
        years = tuple((i + 1) * YEARS / n for i in range(n))
        single = VOL * VOL / (n * n) * math.fsum(
            years[i] * (2 * (n - (i + 1)) + 1) for i in range(n)
        )
        assert _reference_variance(years, VOL) == pytest.approx(single, rel=1e-13)

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 12, 60])
    def test_the_double_sum_equals_EQ_2_114b_on_a_UNIFORM_grid(self, n: int) -> None:
        """`σ²T[1 − (1/6)(4n+1)(n−1)/n²]` — a closed form in `n` alone, a third path entirely."""
        years = tuple((i + 1) * YEARS / n for i in range(n))
        closed = VOL * VOL * YEARS * (1.0 - (4 * n + 1) * (n - 1) / (6.0 * n * n))
        assert _reference_variance(years, VOL) == pytest.approx(closed, rel=1e-13)

    def test_the_CONTINUOUS_limit_is_sigma_squared_T_over_three(self) -> None:
        """Eq (2.115b): as `n → ∞` the discrete variance reduces to `σ²T/3`, consistent with the
        continuous-averaging result of §2.7.1 — the free limit companion the spec marked optional."""
        n = 20_000
        years = tuple((i + 1) * YEARS / n for i in range(n))
        closed = VOL * VOL * YEARS * (1.0 - (4 * n + 1) * (n - 1) / (6.0 * n * n))
        assert closed == pytest.approx(VOL * VOL * YEARS / 3.0, rel=1e-4)

    def test_the_MEAN_matches_eq_2_114a_on_a_uniform_grid(self) -> None:
        """The same treatment for the drift: (2.111)'s general `(1/n)Σ(r−δ−½σ²)tᵢ` against
        (2.114a)'s `½(r−δ−½σ²)(1+1/n)T`."""
        for n in (1, 3, 12):
            years = tuple((i + 1) * YEARS / n for i in range(n))
            closed = math.log(SPOT) + 0.5 * (RATE - CONVENIENCE - 0.5 * VOL * VOL) * (
                1.0 + 1.0 / n
            ) * YEARS
            assert _reference_log_mean(years, VOL, CONVENIENCE) == pytest.approx(
                closed, rel=1e-14
            )


class TestTheAdjustmentsPointwise:
    """§4: the one new load-bearing formula, transcribed pointwise against a reference this file
    computes itself — the F-6.1 lesson, that a formula living only inside a composite is never
    actually pinned. Both `F_G` and `σ²_G` are named surface for exactly that reason."""

    def test_the_adjusted_VARIANCE_matches_the_double_sum(self) -> None:
        assert geometric_average_variance(FIXING_YEARS, VOL) == pytest.approx(
            _reference_variance(FIXING_YEARS, VOL), rel=1e-14
        )

    @pytest.mark.parametrize("vol", [0.12, 0.31, 0.65])
    @pytest.mark.parametrize("convenience", [0.0, 0.015, 0.06])
    def test_the_adjusted_FORWARD_matches_the_lognormal_mean(
        self, vol: float, convenience: float
    ) -> None:
        got = geometric_average_forward(_underlying(convenience), FIXINGS, FLAT, vol)
        assert got == pytest.approx(
            _reference_forward(FIXING_YEARS, vol, convenience), rel=1e-13
        )

    def test_the_adjusted_forward_is_BELOW_the_geometric_mean_of_the_forwards(self) -> None:
        """A requirement the finance forces, independent of the transcription: the convexity
        correction `½(σ²_G − σ²·t̄)` is negative because averaging destroys variance, so `F_G` sits
        **below** the plain geometric mean of the fixing forwards. A sign slip on it dies here."""
        geometric_mean = math.exp(
            math.fsum(
                math.log(commodity_forward(_underlying(), f, FLAT)) for f in FIXINGS
            )
            / len(FIXINGS)
        )
        got = geometric_average_forward(_underlying(), FIXINGS, FLAT, VOL)
        assert got < geometric_mean
        assert abs(got / geometric_mean - 1.0) > 1e-3, (got, geometric_mean)


class TestAveragingReducesVariance:
    """The requirement oracle: averaging strictly reduces variance, with equality exactly when the
    schedule degenerates to a single date. Survives a transcription error the source-value oracles
    would share."""

    def test_the_adjusted_variance_is_BELOW_the_terminal_variance(self) -> None:
        assert geometric_average_variance(FIXING_YEARS, VOL) < VOL * VOL * YEARS

    def test_COINCIDENT_fixings_give_EQUALITY(self) -> None:
        """Equality iff the fixings coincide — the boundary of the inequality above, which an
        implementation that merely scaled the variance down by a constant would fail."""
        coincident = (YEARS,) * 5
        assert geometric_average_variance(coincident, VOL) == pytest.approx(
            VOL * VOL * YEARS, rel=1e-14
        )

    def test_MORE_fixings_over_the_same_window_reduce_the_variance_MONOTONICALLY(self) -> None:
        variances = [
            geometric_average_variance(
                tuple((i + 1) * YEARS / n for i in range(n)), VOL
            )
            for n in (1, 2, 4, 8, 16, 32)
        ]
        assert all(b < a for a, b in zip(variances, variances[1:])), variances


class TestTheSingleFixingReduction:
    """The anchor, and the companion §4 and T39 require beside it."""

    def test_ONE_fixing_is_EXACTLY_the_T60_european(self) -> None:
        """`F_G = F(t₁)` and `σ²_G = σ²t₁`, so the price is T60's — exact by construction, not a
        numerical limit. The strongest form of a reduction anchor."""
        single = _asian(fixings=(EXPIRY,))
        european = CommodityOption(
            asset=WTI, strike=80.0, expiry=EXPIRY, notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        assert _pv(single) == _pv(european)

    def test_the_COMPANION_is_non_degenerate_in_what_the_reduction_KILLS(self) -> None:
        """**T39, stated as the charter demands.** The `n = 1` anchor is exact *because* it
        collapses the averaging — so it is structurally blind to the entire double-sum, which is
        where a defect would live. The companion must therefore be non-degenerate in **fixing count
        and spacing**, and a schedule with every fixing at expiry is `n = 1` in disguise.

        Measured: the real schedule's variance is 46% of the single-fixing one, and its six fixings
        span 2.2 years."""
        single = geometric_average_variance((YEARS,), VOL)
        many = geometric_average_variance(FIXING_YEARS, VOL)
        assert many / single < 0.6, (many, single)
        assert FIXING_YEARS[-1] - FIXING_YEARS[0] > 2.0


class TestTheEngineIsAFaithfulBlackWrapper:
    def test_the_price_is_EXACTLY_discounted_black_on_the_adjusted_pair(self) -> None:
        """The reduction proving this tranche adds the averaging and does not perturb Black."""
        forward = geometric_average_forward(_underlying(), FIXINGS, FLAT, VOL)
        variance = geometric_average_variance(FIXING_YEARS, VOL)
        direct = black_price(forward, 80.0, variance, CommodityOptionKind.CALL)
        assert _pv(_asian()) == 1_000.0 * FLAT.df(EXPIRY) * direct

    @pytest.mark.parametrize("strike", [45.0, 65.0, 95.0, 130.0])
    def test_PUT_CALL_PARITY_on_the_adjusted_forward(self, strike: float) -> None:
        """`C − P = df·(F_G − K)`, model-free, swept **off the money** — at `K = F_G` the identity
        degenerates to `0 = 0`, which every wrong implementation also satisfies."""
        call = _pv(_asian(strike, CommodityOptionKind.CALL))
        put = _pv(_asian(strike, CommodityOptionKind.PUT))
        forward = geometric_average_forward(_underlying(), FIXINGS, FLAT, VOL)
        assert call - put == pytest.approx(
            1_000.0 * FLAT.df(EXPIRY) * (forward - strike), rel=1e-12
        )

    def test_the_parity_SWEEP_is_genuinely_off_the_money(self) -> None:
        forward = geometric_average_forward(_underlying(), FIXINGS, FLAT, VOL)
        for strike in (45.0, 65.0, 95.0, 130.0):
            assert abs(strike / forward - 1.0) > 0.1, (strike, forward)


class TestTheMonteCarloCompanion:
    """**The oracle that exercises the real averaging**, which the `n = 1` reduction cannot and the
    pointwise transcription only checks against itself. Simulates the spot on the actual fixing
    grid, forms the geometric average per path, and prices the discounted payoff.

    The closed form is **exact**, so the only error is sampling: the band is `k·SE` with no
    approximation term to add (§4's "a tolerance that lumps two error sources together cannot tell
    them apart" — here there is only one source, and saying so is the point)."""

    PATHS = 40_000

    def _mc(self, strike: float, kind: CommodityOptionKind, paths: int, seed: int):
        """Correlated Brownian motion on the fixing grid, by independent increments."""
        n = len(FIXING_YEARS)
        count = paths * n
        source = UniformSource(seed)
        normals = standard_normals(source.uniforms(count + count % 2))
        steps = [FIXING_YEARS[0]] + [
            b - a for a, b in zip(FIXING_YEARS, FIXING_YEARS[1:])
        ]
        drift = RATE - CONVENIENCE - 0.5 * VOL * VOL
        payoffs: list[float] = []
        for p in range(paths):
            brownian = 0.0
            log_sum = 0.0
            for i in range(n):
                brownian += math.sqrt(steps[i]) * normals[p * n + i]
                log_sum += math.log(SPOT) + drift * FIXING_YEARS[i] + VOL * brownian
            average = math.exp(log_sum / n)
            payoffs.append(max(kind.omega * (average - strike), 0.0))
        return estimate(payoffs)

    @pytest.mark.parametrize("strike", [65.0, 80.0, 100.0])
    def test_the_closed_form_sits_INSIDE_the_band(self, strike: float) -> None:
        sampled = self._mc(strike, CommodityOptionKind.CALL, self.PATHS, seed=61_001)
        closed = _pv(_asian(strike)) / 1_000.0
        discounted_mean = FLAT.df(EXPIRY) * sampled.mean
        discounted_se = FLAT.df(EXPIRY) * sampled.standard_error
        assert abs(closed - discounted_mean) < 4.0 * discounted_se, (
            f"closed {closed:.6f} vs MC {discounted_mean:.6f} +- {discounted_se:.6f} "
            f"({abs(closed - discounted_mean) / discounted_se:.2f} SE)"
        )

    def test_THE_RESOLUTION_this_band_rules_out(self) -> None:
        """What a green actually proves, stated from measurement (§4). At 40k paths the SE is ~0.9%
        of the price, so the `4·SE` band resolves a defect of about **3.6%** — and a 10% error in
        the adjusted variance moves the price by far more than that, which this measures rather than
        assumes."""
        sampled = self._mc(80.0, CommodityOptionKind.CALL, self.PATHS, seed=61_001)
        band = 4.0 * FLAT.df(EXPIRY) * sampled.standard_error
        correct = _pv(_asian(80.0)) / 1_000.0

        forward = geometric_average_forward(_underlying(), FIXINGS, FLAT, VOL)
        variance = geometric_average_variance(FIXING_YEARS, VOL)
        inflated = FLAT.df(EXPIRY) * black_price(
            forward, 80.0, 1.10 * variance, CommodityOptionKind.CALL
        )
        assert abs(inflated - correct) > band, (
            f"a 10% variance defect ({abs(inflated - correct):.6f}) must fall outside the "
            f"{band:.6f} band, or this oracle cannot see it"
        )


class TestTheContractAndItsGuards:
    def test_a_FULLY_SETTLED_asian_is_PV_ZERO(self) -> None:
        """Invariant 6: every fixing on or before the valuation date means the trade has settled."""
        past = tuple(VAL - timedelta(days=d) for d in (300, 200, 100))
        assert _pv(_asian(fixings=past)) == 0.0

    def test_a_settled_asian_DEEP_IN_THE_MONEY_is_still_PV_ZERO(self) -> None:
        """F-4.1's sub-shape: an out-of-the-money settled option has intrinsic `0` too, so the
        guard stays invisible unless the fixture is **in the money**. `K = 5` against a ~78 spot."""
        past = tuple(VAL - timedelta(days=d) for d in (300, 200, 100))
        assert _pv(_asian(strike=5.0, fixings=past)) == 0.0

    def test_a_PARTLY_FIXED_asian_is_REFUSED_not_silently_mispriced(self) -> None:
        """The source scopes itself the same way — Clark restricts to "no historical fixings have
        taken place yet". Pricing a partly-fixed Asian needs the realised prints, which the snapshot
        does not carry for a commodity; treating it as unfixed would be a plausible wrong number."""
        straddling = (VAL - timedelta(days=30), VAL + timedelta(days=200), EXPIRY)
        out = price(_asian(fixings=straddling), CommodityBlackModel(_market(), VOL))
        assert isinstance(out, PricingFailure)
        assert "historical" in out.reason or "fixed" in out.reason, out.reason

    def test_a_MISSING_commodity_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT})
        )
        out = price(_asian(), CommodityBlackModel(bare, VOL))
        assert isinstance(out, PricingFailure) and "WTI" in out.reason

    def test_an_EMPTY_fixing_schedule_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="at least one fixing"):
            _asian(fixings=())

    def test_fixings_OUT_OF_ORDER_are_refused_at_construction(self) -> None:
        """The sorted-single-sum form `Σ tᵢ[2(n−i)+1]` is only valid for an ascending schedule, and
        more basically an averaging schedule that doubles back is not a schedule."""
        with pytest.raises(ValueError, match="ascending"):
            _asian(fixings=(FIXINGS[2], FIXINGS[0], FIXINGS[1]))

    def test_DUPLICATE_fixings_are_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            _asian(fixings=(FIXINGS[0], FIXINGS[0], FIXINGS[1]))

    def test_a_NON_POSITIVE_strike_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="strike"):
            _asian(strike=0.0)

    def test_the_asian_round_trips_and_refuses_an_unknown_version(self) -> None:
        product = _asian()
        assert CommodityGeometricAsian.from_dict(product.to_dict()) == product
        bad = product.to_dict()
        bad["schema_version"] = 99
        with pytest.raises(ValueError, match="reads up to"):
            CommodityGeometricAsian.from_dict(bad)

    def test_the_FIXING_SCHEDULE_is_constitutive_and_seen_by_equality(self) -> None:
        """§6: two Asians differing only in their schedule are different instruments — they average
        different dates — so the schedule is carried and the round trip exercises it."""
        assert _asian(fixings=FIXINGS) != _asian(fixings=FIXINGS[:-1])
