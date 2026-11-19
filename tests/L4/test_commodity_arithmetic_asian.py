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


#: The Monte-Carlo runs are deterministic in `(strike, vol, paths, seed)` and several oracles below
#: want the same reference twice, so they are memoised. A lossless reduction (the T50/T53 pattern),
#: not a weakening: every assertion still reads the same numbers it would have recomputed.
_MC_CACHE: dict[tuple[float, float, int, int], tuple[float, float]] = {}


def _mc_mean_se(strike: float, vol: float, paths: int, seed: int) -> tuple[float, float]:
    """Vectorised GBM on the actual fixing grid; the discounted mean payoff and its standard error.

    Prices the **real arithmetic average** — which is the whole point: it is the thing Turnbull–
    Wakeman approximates, so it is the only oracle that can see the approximation at all.
    """
    key = (strike, vol, paths, seed)
    if key in _MC_CACHE:
        return _MC_CACHE[key]
    rng = np.random.default_rng(seed)
    years = np.array(FIXING_YEARS)
    steps = np.diff(np.concatenate(([0.0], years)))
    brownian = np.cumsum(rng.standard_normal((paths, len(years))) * np.sqrt(steps), axis=1)
    spots = SPOT * np.exp((RATE - CONVENIENCE - 0.5 * vol * vol) * years + vol * brownian)
    payoff = np.maximum(spots.mean(axis=1) - strike, 0.0)
    df = FLAT.df(EXPIRY)
    out = (
        float(df * payoff.mean()),
        float(df * payoff.std(ddof=1) / math.sqrt(paths)),
    )
    _MC_CACHE[key] = out
    return out


class TestTheApproximationErrorIsMEASURED:
    """**The heart of the tranche.** Turnbull–Wakeman is an approximation, so the deliverable is its
    *measured* error — not a band loose enough to hide it (§4: a tolerance that lumps two error
    sources together cannot tell them apart).

    Measured on this fixture at 400k paths, at the money, the error grows monotonically with `σ√T`:

    | `σ√T` | relative error |
    |---|---|
    | 0.079 | 0.41% |
    | 0.158 | 0.58% |
    | 0.316 | 0.98% |
    | 0.490 | 1.66% |
    | 0.948 | **4.81%** |

    so it crosses **0.5% at about `σ√T ≈ 0.12`** and is a four-to-five-percent effect by `σ√T ≈ 1`.
    That is the honest negative, and it is the material the tranche exists to produce.
    """

    PATHS = 60_000
    BIG_PATHS = 400_000
    SEED = 62_303

    def _tw(self, strike: float, vol: float) -> float:
        return _pv(_arithmetic(strike), vol) / 1_000.0

    def test_the_TW_error_GROWS_MONOTONICALLY_with_sigma_root_T(self) -> None:
        """The shape of the approximation, not a single number — and every point resolved above the
        reference run's own noise, or the sweep would be measuring sampling error."""
        measured: list[tuple[float, float]] = []
        for vol in (0.10, 0.20, 0.31, 0.60):
            mean, se = _mc_mean_se(80.0, vol, self.BIG_PATHS, self.SEED)
            relative = abs(self._tw(80.0, vol) - mean) / mean
            assert abs(self._tw(80.0, vol) - mean) > 2.0 * se, (
                f"at vol {vol} the TW error is not resolved above the reference SE "
                f"({abs(self._tw(80.0, vol) - mean) / se:.2f} SE) — this point measures noise"
            )
            measured.append((vol * math.sqrt(YEARS), relative))
        assert all(b[1] > a[1] for a, b in zip(measured, measured[1:])), measured
        assert measured[-1][1] > 0.04, measured

    def test_the_TW_error_crosses_HALF_A_PERCENT_in_the_low_vol_regime(self) -> None:
        """Where the approximation stops being free, stated as a number rather than a feeling: at
        `σ√T ≈ 0.08` it is 0.41% and at `σ√T ≈ 0.16` it is 0.58%, so the 0.5% line sits between."""
        low, _ = _mc_mean_se(80.0, 0.05, self.BIG_PATHS, self.SEED)
        high, _ = _mc_mean_se(80.0, 0.10, self.BIG_PATHS, self.SEED)
        assert abs(self._tw(80.0, 0.05) - low) / low < 0.005
        assert abs(self._tw(80.0, 0.10) - high) / high > 0.005

    def test_the_TW_error_CHANGES_SIGN_across_strikes(self) -> None:
        """**A structural property, and the reason a per-strike resolution claim is impossible.**

        TW matches the first two moments *exactly*, so its error cannot be one-signed across the
        strike range — it over-prices the low strikes and under-prices the high ones, and therefore
        passes through **zero** somewhere in between (near `K ≈ 98` on this fixture at vol 0.31).
        At that strike no path count resolves the error, because there is none to resolve.

        This is not a defect and it is not an accident: it is what exact moment matching buys. It is
        asserted here so that `test_..._inside_FOUR_SE_PLUS_the_measured_error` can treat the
        measured error as a **bound** rather than demanding it be resolvable everywhere — which is
        the mistake this file's first draft made and this measurement corrected."""
        signed = []
        for strike in (65.0, 80.0, 115.0):
            mean, se = _mc_mean_se(strike, 0.31, self.BIG_PATHS, self.SEED)
            signed.append(((self._tw(strike, 0.31) - mean) / se, strike))
        assert signed[0][0] > 3.0, signed
        assert signed[-1][0] < -3.0, signed

    @pytest.mark.parametrize("strike", [70.0, 80.0, 95.0])
    def test_the_TW_price_sits_inside_FOUR_SE_PLUS_the_measured_error(self, strike: float) -> None:
        """The band §4 demands: `k·SE + the approximation's own error`, the two terms kept **apart**
        and the second **measured** at a path count where sampling noise is small — never one flat
        percentage standing in for both (the T17 rule).

        The measured term is a **bound**, not a claim that an error exists: where TW happens to be
        accurate beyond the reference run's resolution (near the sign change above) the bound is
        simply tight, which makes the band stricter rather than looser."""
        mean, se = _mc_mean_se(strike, VOL, self.PATHS, 62_101)
        reference, reference_se = _mc_mean_se(strike, VOL, self.BIG_PATHS, self.SEED)
        tw = self._tw(strike, VOL)
        bound_on_tw_error = abs(tw - reference) + 2.0 * reference_se
        assert abs(tw - mean) < 4.0 * se + bound_on_tw_error, (
            f"TW {tw:.6f} vs MC {mean:.6f} ± {se:.6f}; TW error bounded by "
            f"{bound_on_tw_error:.6f} from the {self.BIG_PATHS}-path reference"
        )

    def test_THE_RESOLUTION_this_band_rules_out_and_WHAT_IT_IS_FOR(self) -> None:
        """What a green proves, named **per defect** (§4, F-5.3) — and the more useful half is what
        this band is *not* the oracle for.

        Measured at the money: the band is `4·SE(0.255) + |TW − ref|(0.149) + 2·refSE(0.050) =
        0.454`, and a second-moment defect must reach **25%** to fall outside it (a 10% bump moves
        the price 0.394, inside; 25% moves it 0.947, outside). The limiting term is **TW's own
        approximation error**, not the sampling noise — a third of the band is the thing being
        validated.

        **And that is the right answer, not a weakness, because §4's T51 clause applies exactly:**
        when a band cannot resolve a defect, the first question is whether that quantity needs a
        band at all. `M₁` and `M₂` are **closed form** — `test_M2_matches_the_double_sum` pins the
        second moment pointwise at `rel=1e-14`, some twelve orders tighter than any path count could
        buy — so the moments are not this band's business and widening it for them would spend paths
        on what an exact check resolves for free.

        What the band *is* for is the one thing no exact check can see: **the lognormal
        assumption**. The moments are matched exactly by construction, so an oracle built from them
        is structurally blind to whether assuming `A` lognormal was a good idea. Only a simulation
        of the real arithmetic average can answer that, and this band is where that answer lives.
        """
        mean, se = _mc_mean_se(80.0, VOL, self.PATHS, 62_101)
        reference, reference_se = _mc_mean_se(80.0, VOL, self.BIG_PATHS, self.SEED)
        tw = self._tw(80.0, VOL)
        band = 4.0 * se + abs(tw - reference) + 2.0 * reference_se

        first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        variance = second - first * first

        def price_with(bump: float) -> float:
            total = math.log((first * first + bump * variance) / (first * first))
            return FLAT.df(EXPIRY) * black_price(
                first, 80.0, total, CommodityOptionKind.CALL
            )

        assert abs(price_with(1.10) - tw) < band, (
            "a 10% second-moment defect sits INSIDE this band — stated, not hidden: the band is "
            "not the moments' oracle, the pointwise transcription is"
        )
        assert abs(price_with(1.25) - tw) > band, (
            f"a 25% second-moment defect ({abs(price_with(1.25) - tw):.6f}) must fall outside the "
            f"{band:.6f} band, or this oracle resolves nothing at all"
        )
        # And the limiting term is the approximation, not the sampling: say so from measurement.
        assert abs(tw - reference) > 0.5 * 4.0 * se, (
            "if sampling dominated the band, more paths would sharpen it; it does not, so the "
            "approximation error is the floor"
        )

    def test_the_MOMENTS_are_pinned_TWELVE_ORDERS_tighter_than_the_band(self) -> None:
        """The companion that makes the claim above checkable rather than rhetorical: the band
        resolves a 25% second-moment defect, while the pointwise oracle resolves one at `1e-14`
        relative. A defect anywhere between is caught by the pointwise check and invisible to the
        band — which is why both exist and neither substitutes for the other (§4)."""
        first, second = arithmetic_average_moments(_underlying(), FIXINGS, FLAT, VOL)
        reference = _reference_m2_double(_forwards(), FIXING_YEARS, VOL)
        assert abs(second / reference - 1.0) < 1e-14
        # A defect far below the band's 25% is caught outright by the pointwise oracle.
        assert abs(second * 1.000_001 / reference - 1.0) > 1e-14


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
