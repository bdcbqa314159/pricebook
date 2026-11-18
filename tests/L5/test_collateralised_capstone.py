"""The arc capstone: collateralised CVA of a collateralised swap (L5) — tranche 58, PRE-REGISTERED.

The whole collateralised-discounting arc, wired end to end: the **LGM (T57)** simulates rate paths →
the **collateralised IRS (T55)** marks `V(t)` on each → **`collateral.py` (T29)** takes the
margin-period gap → **`credit_adjustment`/`mc_exposure_cva` (T25/T26)** prices the CVA. Computed in
**both discounting eras**, which is this arc's design invariant having its last and most exposed
test.

**Two findings from the spec, both named here because both change what is built.**

**1. The spec's threshold formula is ONE-WAY and cannot satisfy the spec's own reduction.** §1 gives
posted collateral as `max(V(t−δ) − H, 0)`, and asks that `H = 0` reproduce T29's gap to the ULP. It
cannot: T29 posts `V(t−δ)` **unfloored** (`max(V(t) − V(t−δ), 0)`), so the one-way form differs on
**every path whose previous mark is negative** — measured, 3 of 5 sample cases, and for a swap that
is about half the paths. The form that does reduce is the **two-way** CSA threshold, which is also
what a two-way CSA actually is:

    C(t−δ) = max(V(t−δ) − H, 0) − max(−V(t−δ) − H, 0)

— we hold collateral above `+H`, post below `−H`, nothing in the band. At `H = 0` that is
`V⁺ − V⁻ = V` **exactly**, so T29 falls out; as `H → ∞` it is `0`, so the uncollateralised case
falls out. Both of the spec's reductions hold for the two-way form and only one holds for the
one-way. The same shape as T57's eq (11.4): a formula correct in one degenerate corner, and the
spec's own named reduction is what catches it.

**2. The LGM→IRS wiring needs a surface none of the four modules exposes**, which the spec flagged
as a possible Ask. `swap_values_on_paths` (T26) is **LMM-specific** — it takes forward-curve paths,
an `LmmTenor` and a `SwapSpan` — so it cannot value a T55 `InterestRateSwap` on LGM states. Hence
`swap_exposure_on_lgm`.

**And that surface needs a basis assumption, which is a modelling choice rather than a
transcription**, so it is stated rather than buried: the **tenor basis is frozen** at today's
forward basis, so a projection bond on a path is the LGM-evolved discount bond times today's
deterministic basis factor. It is the standard assumption, and it has a property worth having —
in single-curve mode the factor is identically `1`, so **one code path serves both eras** and the
invariant holds by construction rather than by branch, exactly as it does in T55's pricer.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.money import Currency, Money
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.tenor import Tenor
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.market.curve_set import ProjectionKey
from pricebook.market.multi_curve import projection_curve_from_par_swaps
from pricebook.market.nominal_bootstrap import ParSwapQuote, nominal_curve_from_par_swaps
from pricebook.market.survival_curve import SurvivalCurve
from pricebook.models.black_mc import MonteCarloConfig
from pricebook.models.hull_white import HullWhiteParams
from pricebook.products.interest_rate_swap import InterestRateSwap, SwapDirection
from pricebook.risk.collateral import CollateralAgreement, collateralized_exposure
from pricebook.risk.collateralised_capstone import swap_exposure_on_lgm
from pricebook.risk.mc_exposure import mc_exposure_cva

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
SIX_MONTH = Tenor(months=6)
PARAMS = HullWhiteParams(kappa=0.08, sigma=0.012)
PATHS = MonteCarloConfig(paths=4_000, seed=5_801)

#: **182-day periods, not calendar semi-annual** — so the year fractions are *exactly* uniform.
#: Calendar half-years give 0.49863 and 0.50137 alternately on ACT/365F, and
#: `collateralized_exposure` refuses a non-uniform margin-call grid (rightly: `V(t−δ)` has to be a
#: point the simulation visited). The schedule's own dates are therefore the margin-call grid, and
#: no period straddles an exposure date.
_PERIOD_DAYS = 182
MARGIN_PERIOD = _PERIOD_DAYS / 365.0
HORIZON = 3
OIS_RATES = {1: 0.0200, 2: 0.0215, 3: 0.0228}
TENOR_RATES = {1: 0.0260, 2: 0.0278, 3: 0.0295}
SURVIVAL = SurvivalCurve.flat(VAL, 0.02, DC)
LGD = 0.6


def _schedule(years: int = HORIZON) -> CouponSchedule:
    """Semi-annual dates, which is also the exposure grid."""
    return CouponSchedule(
        tuple(VAL + timedelta(days=_PERIOD_DAYS * i) for i in range(years * 2 + 1)), DC
    )


def _quotes(rates: dict[int, float]) -> tuple[ParSwapQuote, ...]:
    return tuple(
        ParSwapQuote(
            CouponSchedule(tuple(date(VAL.year + i, 1, 1) for i in range(n + 1)), DC), rates[n]
        )
        for n in sorted(rates)
    )


def _ois() -> DiscountCurve:
    return nominal_curve_from_par_swaps(MEASURE, _quotes(OIS_RATES))


def _multi_curve() -> MarketSnapshot:
    ois = _ois()
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {CurveKey(CurveRole.DISCOUNT, EUR): ois},
            projections={
                ProjectionKey(EUR, SIX_MONTH): projection_curve_from_par_swaps(
                    MEASURE, _quotes(TENOR_RATES), ois
                )
            },
        ),
    )


def _single_curve(rates: dict[int, float] = TENOR_RATES) -> MarketSnapshot:
    """The pre-crisis snapshot: one curve, no projections store at all."""
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {CurveKey(CurveRole.DISCOUNT, EUR): nominal_curve_from_par_swaps(MEASURE, _quotes(rates))}
        ),
    )


def _swap(fixed_rate: float = 0.025) -> InterestRateSwap:
    return InterestRateSwap(
        schedule=_schedule(),
        fixed_rate=fixed_rate,
        notional=Money(10_000_000.0, EUR),
        tenor=SIX_MONTH,
        direction=SwapDirection.PAYER,
    )


def _exposure(market: MarketSnapshot, swap: InterestRateSwap | None = None):
    return swap_exposure_on_lgm(swap or _swap(), market, PARAMS, PATHS)


def _cva(exposure, market: MarketSnapshot) -> float:
    return mc_exposure_cva(
        exposure, SURVIVAL, market.curves.discount(EUR), LGD, EUR
    ).pv.amount


class TestThePremises:
    """Asserted, because every reduction here collapses something that must be alive elsewhere."""

    def test_there_is_a_REAL_basis(self) -> None:
        for n in OIS_RATES:
            assert TENOR_RATES[n] > OIS_RATES[n] + 0.002, f"basis too small at {n}y"

    def test_the_mean_reversion_is_structural(self) -> None:
        assert PARAMS.kappa > 0.0
        with pytest.raises(ValueError):
            HullWhiteParams(kappa=0.0, sigma=0.012)

    def test_the_exposure_grid_IS_the_margin_call_grid(self) -> None:
        """`collateralized_exposure` refuses any other grid, and a period straddling an exposure
        date would need a fixing the state does not carry."""
        times = _exposure(_multi_curve()).times
        spacings = [b - a for a, b in zip((0.0,) + times, times)]
        assert all(abs(s - MARGIN_PERIOD) < 1e-9 for s in spacings), spacings

    def test_the_marks_take_BOTH_signs(self) -> None:
        """The premise the threshold findings turn on: if every previous mark were positive, the
        one-way and two-way forms would agree and the whole distinction would be invisible."""
        values = _exposure(_multi_curve()).values
        flat = [v for row in values for v in row]
        assert min(flat) < 0.0 < max(flat), "the fixture must produce marks of both signs"


class TestTheThresholdReducesBOTHWays:
    """The two reductions the spec named — and the one-way form fails the first of them."""

    def _agreement(self, threshold: float) -> CollateralAgreement:
        return CollateralAgreement(
            margin_period=MARGIN_PERIOD, initial_mark=0.0, threshold=threshold
        )

    def test_a_ZERO_threshold_reproduces_T29s_gap_exactly(self) -> None:
        exposure = _exposure(_multi_curve())
        with_threshold = collateralized_exposure(exposure, self._agreement(0.0))
        legacy = collateralized_exposure(
            exposure, CollateralAgreement(margin_period=MARGIN_PERIOD, initial_mark=0.0)
        )
        assert with_threshold.values == legacy.values, "H=0 must BE the built gap, not approximate it"

    def test_THE_ONE_WAY_FORM_would_fail_that_reduction(self) -> None:
        """Why the two-way form is built: with `posted = max(V(t−δ) − H, 0)`, a negative previous
        mark posts zero instead of its (negative) value, so the gap is understated. Measured here
        rather than argued, on the paths this fixture actually produces."""
        exposure = _exposure(_multi_curve())
        legacy = collateralized_exposure(
            exposure, CollateralAgreement(margin_period=MARGIN_PERIOD, initial_mark=0.0)
        )
        differing = 0
        for row, gap_row in zip(exposure.values, legacy.values):
            for value, previous, gap in zip(row, (0.0,) + row[:-1], gap_row):
                one_way = max(value - max(previous - 0.0, 0.0), 0.0)
                if abs(one_way - gap) > 1e-12:
                    differing += 1
        assert differing > 0, (
            "the one-way form must differ from T29 somewhere, or this finding is not real"
        )

    def test_an_ENORMOUS_threshold_reproduces_the_uncollateralised_exposure(self) -> None:
        exposure = _exposure(_multi_curve())
        huge = collateralized_exposure(exposure, self._agreement(1e12))
        for row, raw in zip(huge.values, exposure.values):
            for collateralised, uncollateralised in zip(row, raw):
                assert collateralised == pytest.approx(max(uncollateralised, 0.0), abs=1e-9)

    def test_THE_COMPANION_a_live_threshold_is_between_the_two(self) -> None:
        """T39: both reductions kill the threshold (one sets it to zero, the other to infinity), so
        a mechanic that did nothing would pass both. The exposure must be **monotone in `H`** and
        strictly between the endpoints for a real threshold."""
        exposure = _exposure(_multi_curve())
        cvas = [
            _cva(collateralized_exposure(exposure, self._agreement(h)), _multi_curve())
            for h in (0.0, 25_000.0, 100_000.0, 400_000.0, 1e12)
        ]
        for lower, higher in zip(cvas, cvas[1:]):
            assert higher >= lower - 1e-9, f"CVA must not fall as the threshold rises: {cvas}"
        assert cvas[-1] > cvas[0] * 1.05, (
            f"the threshold must move the economics materially: {cvas}"
        )
        assert cvas[0] < cvas[2] < cvas[-1], f"a live threshold sits strictly between: {cvas}"


class TestTheEraInvariant:
    """**The arc invariant's last and most exposed test.** Here the failure mode is not a dropped
    branch but a *reasonable-looking requirement* — an exposure engine mandating OIS because
    "collateralised exposure means OIS". Such an engine passes *agree* (trivially, one mode) and
    fails *differ*."""

    def test_the_PRE_CRISIS_mode_computes_a_collateralised_CVA_at_all(self) -> None:
        market = _single_curve()
        assert market.curves.projections == {}
        exposure = collateralized_exposure(
            _exposure(market), CollateralAgreement(MARGIN_PERIOD, 0.0, 50_000.0)
        )
        assert math.isfinite(_cva(exposure, market))

    def test_AGREE_where_the_eras_coincide(self) -> None:
        collapsed = _multi_curve()
        # OIS = projection: build the "multi-curve" snapshot from the OIS quotes on both legs
        coincident = MarketSnapshot(
            valuation_date=VAL,
            curves=CurveSet(
                {CurveKey(CurveRole.DISCOUNT, EUR): _ois()},
                projections={
                    ProjectionKey(EUR, SIX_MONTH): projection_curve_from_par_swaps(
                        MEASURE, _quotes(OIS_RATES), _ois()
                    )
                },
            ),
        )
        single = _single_curve(OIS_RATES)
        assert collapsed is not coincident
        agreement = CollateralAgreement(MARGIN_PERIOD, 0.0, 50_000.0)
        multi_cva = _cva(collateralized_exposure(_exposure(coincident), agreement), coincident)
        single_cva = _cva(collateralized_exposure(_exposure(single), agreement), single)
        assert multi_cva == pytest.approx(single_cva, rel=1e-9)

    def test_DIFFER_by_the_basis_where_a_real_basis_exists(self) -> None:
        """The load-bearing half — an engine that quietly required OIS cannot produce two numbers
        here at all, and one that ignored the projection curve would produce the same one twice."""
        agreement = CollateralAgreement(MARGIN_PERIOD, 0.0, 50_000.0)
        multi = _multi_curve()
        single = _single_curve()
        multi_cva = _cva(collateralized_exposure(_exposure(multi), agreement), multi)
        single_cva = _cva(collateralized_exposure(_exposure(single), agreement), single)
        assert abs(multi_cva - single_cva) > 1.0, (
            f"the eras must differ with a real basis: {multi_cva} vs {single_cva}"
        )


class TestCollateralNeverIncreasesTheCharge:
    """A requirement oracle reused from T29 — and **it is what caught this tranche's one real bug**,
    which is worth recording because of *why* it could.

    T29 files this under "the requirement oracles the finance forces". Pathwise it is not forced:
    with `V(t−δ) = −100` and `V(t) = +10` the margin-period gap is `110`, well above the
    uncollateralised `10` — we posted 100 and may not get it back. What makes the aggregate hold is
    that a *well-behaved* mark moves little over one margin period, so the gap stays small.

    Which is exactly why a violation is diagnostic. Before the basis factor was fixed, this read
    **982.63 against an uncollateralised 650.41** — collateral apparently increasing the charge by
    half — because the reciprocal basis made the simulated marks swing wildly between dates. The
    oracle was not detecting a collateral error; it was detecting that the *mark* was wrong, and it
    did so because it is a property of plausible paths rather than a law. Whether T29's universal
    phrasing should be narrowed is Cowork's call; it is raised rather than restated here."""

    @pytest.mark.parametrize("threshold", [0.0, 50_000.0, 250_000.0])
    def test_the_collateralised_cva_is_at_most_the_uncollateralised_one(
        self, threshold: float
    ) -> None:
        market = _multi_curve()
        exposure = _exposure(market)
        bare = _cva(exposure, market)
        agreed = _cva(
            collateralized_exposure(
                exposure, CollateralAgreement(MARGIN_PERIOD, 0.0, threshold)
            ),
            market,
        )
        assert agreed <= bare + 1e-9, f"collateral increased the CVA: {agreed} > {bare}"
        assert bare > 0.0, "the premise: a live uncollateralised charge"


class TestTheSimulatedMarkIsTheSwapsOwnValue:
    """The wiring's correctness anchor: at `t → 0` the simulated mark must be the swap's own price
    off the same snapshot, which ties the new surface to T55's pricer (different code each side)."""

    def test_the_first_exposure_date_marks_close_to_todays_price(self) -> None:
        from pricebook.engine import price
        from pricebook.models.discounting_model import DiscountingModel

        market = _multi_curve()
        swap = _swap()
        today = price(swap, DiscountingModel(market)).pv.amount
        exposure = _exposure(market, swap)
        first = [row[0] for row in exposure.values]
        mean = sum(first) / len(first)
        # half a year of drift on a 3y swap: the mean mark must be in the same region as today's
        assert abs(mean - today) < 0.25 * abs(today) + 1_000.0, (
            f"the simulated mark at {exposure.times[0]}y is {mean}, today's price is {today}"
        )

    def test_the_mark_falls_as_the_fixed_rate_rises(self) -> None:
        """Direction pinned by the finance: a payer paying more fixed is worth less."""
        market = _multi_curve()
        cheap = _exposure(market, _swap(0.01))
        dear = _exposure(market, _swap(0.05))
        for cheap_row, dear_row in zip(cheap.values, dear.values):
            assert cheap_row[0] > dear_row[0]


class TestTheGuards:
    def test_a_negative_threshold_is_refused(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            CollateralAgreement(margin_period=MARGIN_PERIOD, initial_mark=0.0, threshold=-1.0)

    def test_an_exposure_grid_that_straddles_a_period_is_refused(self) -> None:
        """A period straddling an exposure date has already fixed, and the LGM state does not carry
        the fixing — so it is refused rather than approximated (the `collateralized_exposure` house
        style)."""
        annual = InterestRateSwap(
            schedule=CouponSchedule(
                tuple(date(VAL.year + i, 1, 1) for i in range(HORIZON + 1)), DC
            ),
            fixed_rate=0.025,
            notional=Money(10_000_000.0, EUR),
            tenor=SIX_MONTH,
            direction=SwapDirection.PAYER,
        )
        with pytest.raises(ValueError, match="straddle"):
            swap_exposure_on_lgm(annual, _multi_curve(), PARAMS, PATHS, margin_period=0.5)

    def test_a_missing_discount_curve_is_refused(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
        with pytest.raises(KeyError):
            swap_exposure_on_lgm(_swap(), bare, PARAMS, PATHS)
