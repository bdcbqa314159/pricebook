"""Arithmetic-average commodity options — tranche 62, PRE-REGISTERED. `CLARK-2014` ch 2.7.4
(Turnbull–Wakeman, 1991).

**This one is an approximation, and the tranche's job is to prove the method knows that.** A sum of
lognormals is not lognormal, so unlike T61's geometric average there is no exact closed form.
Turnbull–Wakeman matches the arithmetic average's first two moments to a lognormal and prices with
Black-76. The moments are **exact**; the *only* approximation is the lognormal assumption — so the
moments get a tight pointwise oracle and the price gets a band whose approximation term is
**measured**, never a flat percentage standing in for both error sources (§4, T17).

Clark is explicit that this is the one that matters: *"The overwhelming majority of Asian options in
actual traded markets are with arithmetic averaging not geometric averaging."*

**The source, transcribed from both renderings and reconciled (§4).** pp.66–68:

    M₁ = E[A]  = (1/n)·Σᵢ f(0,tᵢ)                                              (2.118)
    E[S_{tᵢ}S_{tⱼ}] = f(0,tᵢ)·f(0,tⱼ)·exp(σᵢ²tᵢ)      for i < j                (2.121)
    M₂ = E[A²] = (1/n²)·Σ_{i,j} E[S_{tᵢ}S_{tⱼ}]                                (2.120)
               = (1/n²)·Σᵢ f(0,tᵢ)²e^{σᵢ²tᵢ} + (2/n²)·Σ_{j≥2} f(0,tⱼ)·Σ_{i<j} f(0,tᵢ)e^{σᵢ²tᵢ}
                                                                               (2.122)
    σ_A = √( (1/tₙ)·ln( M₂ / M₁² ) )                                           (2.125)

so the Black-76 **total variance** is `ln(M₂/M₁²)` and the forward is `M₁` — priced "on a synthetic
asset with a volatility of σ_A … an initial futures price of E[A], a strike of K".

**Two independent printed forms again, and this time the structure is load-bearing twice over.**
(2.120) gives `M₂` as a symmetric double sum and as diagonal-plus-twice-upper-triangle; with a flat
`σ` the kernel is `exp(σ²·min(tᵢ,tⱼ))`, the same structure T61 pinned. `TestM2HasTwoIndependentForms`
requires them to agree.

**A correction to what this file's author believed before reading the source.** At T61's handoff I
reported that the ascending-schedule guard was justified only by the *geometric* engine's sorted
single-sum form, and that a shared schedule object would therefore carry a constraint one consumer
merely tolerated. That is **wrong**: (2.122)'s evaluation uses `exp(σᵢ²tᵢ)` for `i < j`, which is
only `exp(σ²·min)` when the schedule ascends — so the arithmetic engine needs the identical
constraint for the identical reason. Re-derived from the print rather than carried forward from my
own report, which is the rule §9 states about exactly this.
"""

import math
from datetime import date, timedelta

import numpy as np
import pytest

from pricebook.engine import price
from pricebook.engine.commodity_asian import (
    arithmetic_average_moments,
    geometric_average_forward,
)
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.results import PricingFailure, PricingResult
from pricebook.market.commodity import CommodityAsset, CommodityUnderlying, commodity_forward
from pricebook.market.curve import DiscountCurve
from pricebook.market.curve_set import CurveKey, CurveRole, CurveSet
from pricebook.market.snapshot import MarketSnapshot
from pricebook.models.black import black_price
from pricebook.models.commodity_black import CommodityBlackModel
from pricebook.products.commodity_asian import (
    CommodityArithmeticAsian,
    CommodityGeometricAsian,
)
from pricebook.products.commodity_option import CommodityOption, CommodityOptionKind

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
USD = Currency("USD")
WTI = CommodityAsset("WTI")

RATE, CONVENIENCE = 0.042, 0.015
SPOT, VOL = 78.5, 0.31
FLAT = DiscountCurve.flat(VAL, RATE, DC)

#: Non-uniformly spaced, as at T61 — a uniform grid is where a mis-indexed `min(tᵢ,tⱼ)` hides.
FIXING_DAYS = (120, 211, 365, 548, 700, 912)
FIXINGS = tuple(VAL + timedelta(days=d) for d in FIXING_DAYS)
FIXING_YEARS = tuple(d / 365.0 for d in FIXING_DAYS)
EXPIRY = FIXINGS[-1]
YEARS = FIXING_YEARS[-1]


def _underlying(convenience: float = CONVENIENCE) -> CommodityUnderlying:
    return CommodityUnderlying(asset=WTI, spot=SPOT, convenience_yield=convenience)


def _market() -> MarketSnapshot:
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT}),
        commodities={WTI: _underlying()},
    )


def _arithmetic(
    strike: float = 80.0,
    kind: CommodityOptionKind = CommodityOptionKind.CALL,
    fixings: tuple[date, ...] = FIXINGS,
) -> CommodityArithmeticAsian:
    return CommodityArithmeticAsian(
        asset=WTI, strike=strike, fixings=fixings, notional=Money(1_000.0, USD), kind=kind
    )


def _geometric(
    strike: float = 80.0,
    kind: CommodityOptionKind = CommodityOptionKind.CALL,
    fixings: tuple[date, ...] = FIXINGS,
) -> CommodityGeometricAsian:
    return CommodityGeometricAsian(
        asset=WTI, strike=strike, fixings=fixings, notional=Money(1_000.0, USD), kind=kind
    )


def _pv(product, vol: float = VOL) -> float:
    out = price(product, CommodityBlackModel(_market(), vol))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


# ── references written here from the source, never read back from the engine ───────────────────


def _forwards(fixings: tuple[date, ...] = FIXINGS) -> list[float]:
    return [commodity_forward(_underlying(), f, FLAT) for f in fixings]


def _reference_m1(forwards: list[float]) -> float:
    """Eq (2.118): `M₁ = (1/n)·Σ f(0,tᵢ)`."""
    return math.fsum(forwards) / len(forwards)


def _reference_m2_double(forwards: list[float], years: tuple[float, ...], vol: float) -> float:
    """Eq (2.120)'s symmetric **double sum**, with (2.121)'s kernel at flat `σ`."""
    n = len(forwards)
    return (
        math.fsum(
            forwards[i] * forwards[j] * math.exp(vol * vol * min(years[i], years[j]))
            for i in range(n)
            for j in range(n)
        )
        / (n * n)
    )


def _reference_m2_sorted(forwards: list[float], years: tuple[float, ...], vol: float) -> float:
    """Eq (2.122): diagonal plus **twice the upper triangle** — the source's own evaluation, valid
    for an ascending schedule."""
    n = len(forwards)
    diagonal = math.fsum(
        forwards[i] * forwards[i] * math.exp(vol * vol * years[i]) for i in range(n)
    )
    upper = math.fsum(
        forwards[j]
        * math.fsum(forwards[i] * math.exp(vol * vol * years[i]) for i in range(j))
        for j in range(1, n)
    )
    return (diagonal + 2.0 * upper) / (n * n)


class TestThePremisesTheOraclesRestOn:
    def test_the_fixings_are_DISTINCT_and_NON_UNIFORMLY_spaced(self) -> None:
        gaps = [b - a for a, b in zip(FIXING_YEARS, FIXING_YEARS[1:])]
        assert len(set(FIXING_YEARS)) == len(FIXING_YEARS)
        assert max(gaps) / min(gaps) > 2.0, gaps

    def test_the_horizon_is_NOT_one_year_and_there_are_at_least_three_fixings(self) -> None:
        assert abs(YEARS - 1.0) > 1.4, YEARS
        assert len(FIXINGS) >= 3

    def test_the_AM_GM_gap_is_MATERIALLY_non_zero_on_this_fixture(self) -> None:
        """The premise the ordering oracle rests on: a near-coincident schedule makes AM ≈ GM and
        the geometric-≤-arithmetic check toothless. Measured at 2.09% on the forwards."""
        arithmetic = _reference_m1(_forwards())
        geometric = geometric_average_forward(_underlying(), FIXINGS, FLAT, VOL)
        assert arithmetic / geometric - 1.0 > 0.01, (arithmetic, geometric)


class TestM2HasTwoIndependentForms:
    """§4's derivation-step rule: the source prints `M₂` as a symmetric double sum *and* as
    diagonal-plus-twice-upper-triangle. Two different computation paths for one number, so a
    misprint would have to corrupt both identically."""

    @pytest.mark.parametrize("vol", [0.0, 0.12, 0.31, 0.65, 1.0])
    def test_the_double_sum_equals_the_SORTED_EVALUATION(self, vol: float) -> None:
        forwards = _forwards()
        assert _reference_m2_double(forwards, FIXING_YEARS, vol) == pytest.approx(
            _reference_m2_sorted(forwards, FIXING_YEARS, vol), rel=1e-14
        )

    def test_the_two_forms_would_DISAGREE_on_a_non_ascending_schedule(self) -> None:
        """**Why the ascending guard is load-bearing for the ARITHMETIC engine too**, which this
        file's header records as a correction to an earlier claim of mine. (2.122) uses
        `exp(σᵢ²tᵢ)` for `i < j`, which is `exp(σ²·min(tᵢ,tⱼ))` only when the schedule ascends. Fed
        a descending schedule the two forms part company — so the guard is not a geometric-only
        constraint, and both consumers need it for the identical reason."""
        descending = tuple(reversed(FIXING_YEARS))
        forwards = list(reversed(_forwards()))
        double = _reference_m2_double(forwards, descending, VOL)
        sorted_form = _reference_m2_sorted(forwards, descending, VOL)
        assert double != pytest.approx(sorted_form, rel=1e-6), (double, sorted_form)


class TestTheMomentsPointwise:
    """The moments are **exact** — the lognormal match is the only approximation — so they get a
    tight oracle, not a band. Named surface for the F-6.1 reason: a formula living only inside the
    final price is never actually pinned."""

    @pytest.mark.parametrize("vol", [0.12, 0.31, 0.65])
    def test_M1_matches_eq_2_118(self, vol: float) -> None:
        first, _ = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, vol)
        assert first == pytest.approx(_reference_m1(_forwards()), rel=1e-14)

    def test_M1_does_NOT_depend_on_the_volatility(self) -> None:
        """Eq (2.118) is the average of the forwards — a martingale statement with no vol in it.
        A mutation leaking `σ` into the first moment dies here, and it is the kind of leak the
        price-level oracles would absorb."""
        values = [
            arithmetic_average_moments(_underlying(), FIXINGS, FLAT, v)[0]
            for v in (0.0, 0.31, 1.2)
        ]
        assert max(values) == pytest.approx(min(values), rel=1e-15), values

    @pytest.mark.parametrize("vol", [0.0, 0.12, 0.31, 0.65, 1.0])
    def test_M2_matches_the_double_sum(self, vol: float) -> None:
        _, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, vol)
        assert second == pytest.approx(
            _reference_m2_double(_forwards(), FIXING_YEARS, vol), rel=1e-14
        )

    def test_the_TOTAL_VARIANCE_is_the_log_moment_ratio(self) -> None:
        """Eq (2.125) rearranged: Black-76 wants `σ_A²·tₙ = ln(M₂/M₁²)`, so the engine must feed
        exactly that. A mutation dividing by `tₙ` and forgetting to multiply back dies here."""
        first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        expected = math.log(second / (first * first))
        direct = black_price(first, 80.0, expected, CommodityOptionKind.CALL)
        assert _pv(_arithmetic()) == 1_000.0 * FLAT.df(EXPIRY) * direct

    def test_M2_EXCEEDS_M1_squared_so_the_log_is_defined(self) -> None:
        """A requirement the finance forces: `Var[A] = M₂ − M₁² ≥ 0` by Jensen, with equality only
        at zero vol. Without it (2.125) takes the log of something ≤ 1 and the method is undefined
        rather than merely inaccurate."""
        for vol in (0.0, 0.12, 0.31, 0.65, 1.0, 2.0):
            first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, vol)
            assert second >= first * first, (vol, first, second)
        first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, 0.0)
        assert second == pytest.approx(first * first, rel=1e-13)


class TestTheSingleFixingReduction:
    def test_ONE_fixing_REDUCES_to_the_T60_european(self) -> None:
        """`M₁ = F(t₁)` and `M₂ = F(t₁)²e^{σ²t₁}`, so `ln(M₂/M₁²) = σ²t₁` and the price is T60's.

        **Banded, not `==` — the T61 lesson, which the spec pre-paid rather than let me
        re-discover.** Exact in ℝ; in floating point a `log`/`exp` round trip stands between the two
        paths, so bit-identity must not be asserted (§4, T12a)."""
        single = _arithmetic(fixings=(EXPIRY,))
        european = CommodityOption(
            asset=WTI, strike=80.0, expiry=EXPIRY, notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        assert _pv(single) == pytest.approx(_pv(european), rel=1e-13)

    def test_the_reduction_gap_is_INVARIANT_IN_VOL(self) -> None:
        """The T61 proof, reused because it is the right one: a residual that does not scale with
        `σ²` is a round-trip artefact, where a real defect in the `M₂ = M₁²e^{σ²t}` relation would
        grow across this sweep."""
        european = CommodityOption(
            asset=WTI, strike=80.0, expiry=EXPIRY, notional=Money(1_000.0, USD),
            kind=CommodityOptionKind.CALL,
        )
        gaps = [
            abs(_pv(_arithmetic(fixings=(EXPIRY,)), vol) / _pv(european, vol) - 1.0)
            for vol in (0.12, 0.31, 0.65)
        ]
        assert max(gaps) < 1e-13, gaps

    def test_the_COMPANION_is_non_degenerate_in_what_the_reduction_KILLS(self) -> None:
        """T39: `n = 1` collapses the averaging, so the anchor is blind to the whole `M₂` double
        sum. The companion spans six fixings over 2.2 years, and its variance is measurably below
        the single-fixing one."""
        _, single = arithmetic_average_moments(_underlying(), (EXPIRY,), FLAT, VOL)
        m1_single = _reference_m1(_forwards((EXPIRY,)))
        single_var = math.log(single / (m1_single * m1_single))
        first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        many_var = math.log(second / (first * first))
        assert many_var / single_var < 0.6, (many_var, single_var)
        assert FIXING_YEARS[-1] - FIXING_YEARS[0] > 2.0


class TestGeometricIsBelowArithmetic:
    """**The free cross-product oracle (§4).** AM ≥ GM pathwise, so for a *call* the arithmetic
    price dominates the geometric one — with T61's exact closed form on one side and T62's moment
    match on the other, i.e. **different code each side**. That is what lets it catch a shared
    source-reading error neither within-product check can; it has done so twice before in this
    project (T3's strip reconciliation, T5's cap−floor)."""

    @pytest.mark.parametrize("strike", [55.0, 70.0, 80.0, 95.0, 120.0])
    def test_the_arithmetic_CALL_dominates_the_geometric_one(self, strike: float) -> None:
        assert _pv(_arithmetic(strike)) > _pv(_geometric(strike))

    @pytest.mark.parametrize("strike", [55.0, 70.0, 95.0, 120.0])
    def test_the_arithmetic_PUT_is_DOMINATED_in_turn(self, strike: float) -> None:
        """The ordering reverses for a put, which is the companion that stops the call check from
        passing against an implementation that simply returned a larger number."""
        assert _pv(_arithmetic(strike, CommodityOptionKind.PUT)) < _pv(
            _geometric(strike, CommodityOptionKind.PUT)
        )

    def test_the_two_AGREE_at_a_single_fixing(self) -> None:
        """The boundary of the inequality: with one fixing there is no averaging, both reduce to
        the T60 European, and a strict-inequality implementation would fail here."""
        assert _pv(_arithmetic(fixings=(EXPIRY,))) == pytest.approx(
            _pv(_geometric(fixings=(EXPIRY,))), rel=1e-12
        )


class TestTheApproximationErrorIsMEASURED:
    """**The heart of the tranche.** Turnbull–Wakeman is an approximation, so the honest deliverable
    is its measured error — not a band loose enough to hide it (§4: a tolerance that lumps two
    error sources together cannot tell them apart).

    The MC prices the *real* arithmetic average. The band is `4·SE + the measured TW error`, where
    the TW error is measured at a path count where sampling noise is negligible — never one flat
    percentage standing in for both.
    """

    PATHS = 60_000
    BIG_PATHS = 400_000

    @staticmethod
    def _mc_mean_se(strike: float, vol: float, paths: int, seed: int) -> tuple[float, float]:
        """Vectorised GBM on the fixing grid; returns the discounted mean and its standard error."""
        rng = np.random.default_rng(seed)
        years = np.array(FIXING_YEARS)
        steps = np.diff(np.concatenate(([0.0], years)))
        increments = rng.standard_normal((paths, len(years))) * np.sqrt(steps)
        brownian = np.cumsum(increments, axis=1)
        drift = (RATE - CONVENIENCE - 0.5 * vol * vol) * years
        paths_spot = SPOT * np.exp(drift + vol * brownian)
        average = paths_spot.mean(axis=1)
        payoff = np.maximum(average - strike, 0.0)
        df = FLAT.df(EXPIRY)
        return float(df * payoff.mean()), float(df * payoff.std(ddof=1) / math.sqrt(paths))

    def _tw(self, strike: float, vol: float) -> float:
        return _pv(_arithmetic(strike), vol) / 1_000.0

    def test_the_TW_error_is_measured_across_a_SIGMA_ROOT_T_sweep(self) -> None:
        """The honest negative, measured rather than claimed. At a high path count the MC SE is
        small enough that the gap *is* the approximation error, and it grows with `σ√T` exactly as
        Turnbull–Wakeman's derivation predicts."""
        errors: list[tuple[float, float, float]] = []
        for vol in (0.10, 0.31, 0.60):
            mean, se = self._mc_mean_se(80.0, vol, self.BIG_PATHS, seed=62_000 + int(vol * 100))
            tw = self._tw(80.0, vol)
            relative = abs(tw - mean) / mean
            errors.append((vol * math.sqrt(YEARS), relative, se / mean))
        # The error must be dominated by approximation, not by sampling, or this measures noise.
        for root_t_vol, relative, relative_se in errors:
            assert relative_se < 0.01, (root_t_vol, relative_se)
        # And it must GROW with sigma*sqrt(T) — the shape of the approximation, not a constant.
        assert errors[-1][1] > errors[0][1], errors

    @pytest.mark.parametrize("strike", [70.0, 80.0, 95.0])
    def test_the_TW_price_sits_inside_FOUR_SE_PLUS_the_measured_error(self, strike: float) -> None:
        """The band §4 demands: `k·SE + the measured approximation error`, with the two terms kept
        apart. A single flat percentage here would be exactly the T17 defect."""
        mean, se = self._mc_mean_se(strike, VOL, self.PATHS, seed=62_101)
        big_mean, big_se = self._mc_mean_se(strike, VOL, self.BIG_PATHS, seed=62_202)
        tw = self._tw(strike, VOL)
        measured_tw_error = abs(tw - big_mean)
        assert big_se < 0.25 * measured_tw_error, (
            f"the TW error must be resolved above the reference run's noise, or it is not a "
            f"measurement: error {measured_tw_error:.6f} vs SE {big_se:.6f}"
        )
        assert abs(tw - mean) < 4.0 * se + measured_tw_error, (
            f"TW {tw:.6f} vs MC {mean:.6f} ± {se:.6f}, measured TW error "
            f"{measured_tw_error:.6f}"
        )

    def test_THE_RESOLUTION_this_band_rules_out(self) -> None:
        """What a green proves, named per defect (§4, F-5.3). The band resolves a defect in the
        **second moment**: a 10% error in `M₂ − M₁²` moves the price far outside `4·SE` plus the
        measured TW error. It is silent about anything smaller, and saying so is the point."""
        mean, se = self._mc_mean_se(80.0, VOL, self.PATHS, seed=62_101)
        big_mean, big_se = self._mc_mean_se(80.0, VOL, self.BIG_PATHS, seed=62_202)
        tw = self._tw(80.0, VOL)
        band = 4.0 * se + abs(tw - big_mean)

        first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        inflated_variance = math.log(
            (first * first + 1.10 * (second - first * first)) / (first * first)
        )
        inflated = FLAT.df(EXPIRY) * black_price(
            first, 80.0, inflated_variance, CommodityOptionKind.CALL
        )
        assert abs(inflated - tw) > band, (
            f"a 10% second-moment defect ({abs(inflated - tw):.6f}) must fall outside the "
            f"{band:.6f} band, or this oracle cannot see it"
        )


class TestTheContractAndItsGuards:
    @pytest.mark.parametrize("strike", [45.0, 65.0, 95.0, 130.0])
    def test_PUT_CALL_PARITY_on_the_TW_forward(self, strike: float) -> None:
        """`C − P = df·(M₁ − K)`, model-free in the moment-matched measure, swept off the money."""
        first, _ = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        call = _pv(_arithmetic(strike, CommodityOptionKind.CALL))
        put = _pv(_arithmetic(strike, CommodityOptionKind.PUT))
        assert call - put == pytest.approx(
            1_000.0 * FLAT.df(EXPIRY) * (first - strike), rel=1e-12
        )

    def test_the_parity_SWEEP_is_genuinely_off_the_money(self) -> None:
        first, _ = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        for strike in (45.0, 65.0, 95.0, 130.0):
            assert abs(strike / first - 1.0) > 0.1, (strike, first)

    def test_a_FULLY_SETTLED_asian_is_PV_ZERO_even_DEEP_IN_THE_MONEY(self) -> None:
        past = tuple(VAL - timedelta(days=d) for d in (300, 200, 100))
        assert _pv(_arithmetic(fixings=past)) == 0.0
        assert _pv(_arithmetic(strike=5.0, fixings=past)) == 0.0

    def test_a_PARTLY_FIXED_asian_is_REFUSED(self) -> None:
        straddling = (VAL - timedelta(days=30), VAL + timedelta(days=200), EXPIRY)
        out = price(_arithmetic(fixings=straddling), CommodityBlackModel(_market(), VOL))
        assert isinstance(out, PricingFailure)
        assert "historical" in out.reason, out.reason

    def test_a_MISSING_commodity_is_a_PricingFailure(self) -> None:
        bare = MarketSnapshot(
            valuation_date=VAL, curves=CurveSet({CurveKey(CurveRole.DISCOUNT, USD): FLAT})
        )
        out = price(_arithmetic(), CommodityBlackModel(bare, VOL))
        assert isinstance(out, PricingFailure) and "WTI" in out.reason

    def test_the_schedule_guards_match_the_GEOMETRIC_siblings_exactly(self) -> None:
        """One guard, two products — the duplication removed at this tranche rather than written
        twice. These assertions are what make the sharing checkable rather than asserted."""
        for bad, match in (
            ((), "at least one fixing"),
            ((FIXINGS[2], FIXINGS[0], FIXINGS[1]), "ascending"),
            ((FIXINGS[0], FIXINGS[0]), "ascending"),
        ):
            with pytest.raises(ValueError, match=match):
                _arithmetic(fixings=bad)
            with pytest.raises(ValueError, match=match):
                _geometric(fixings=bad)

    def test_a_NON_POSITIVE_strike_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="strike"):
            _arithmetic(strike=0.0)

    def test_the_asian_round_trips_and_refuses_an_unknown_version(self) -> None:
        product = _arithmetic()
        assert CommodityArithmeticAsian.from_dict(product.to_dict()) == product
        bad = product.to_dict()
        bad["schema_version"] = 99
        with pytest.raises(ValueError, match="reads up to"):
            CommodityArithmeticAsian.from_dict(bad)

    def test_the_two_ASIAN_PRODUCTS_are_DISTINCT_TYPES_not_interchangeable(self) -> None:
        """§6: siblings, not one type with a flag. They are field-identical and priced by
        **different methods** — exact versus moment-matched — so the dispatch must tell them apart,
        and an equality that collapsed them would price an arithmetic Asian with the exact
        geometric formula."""
        assert _arithmetic() != _geometric()
        assert type(_arithmetic()) is not type(_geometric())
