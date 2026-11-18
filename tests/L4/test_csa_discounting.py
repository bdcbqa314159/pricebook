"""CSA discounting — the collateralised swap (L4), tranche 55, PRE-REGISTERED. `LICHTERS-2015` ch 5.

A perfectly-collateralised trade (cash CSA in the trade currency) earns the collateral rate on
posted collateral, so its cashflows **discount at OIS** while each floating coupon **projects off
its tenor curve**. This is the first pricer that reads both of tranche 54's curves, and the first
that runs in **either era** — chosen by what the snapshot holds, not by a flag.

**The scope finding, stated here because it changes what this tranche builds.** The spec names "an
existing swap/leg product and the single-curve swap pricer" as reuse anchors. **Neither exists.**
The tree prices inflation swaps, caps, swaptions, equity options and credit, and has never had a
vanilla fixed-vs-float interest-rate swap: no `InterestRateSwap` product, no registered pricer for
one. So three sentences of the spec are about something that is not there — "the existing
single-curve swap pricer is untouched", the reduction's "must equal the existing single-curve swap
price", and the invariant's "the existing path".

The spec's **design** survives that intact, which is why this tranche proceeds rather than stalling:
§1 describes *one* pricer discounting off "a source that is either a single discount curve or the
`CurveSet`'s OIS curve, chosen by what the snapshot holds". That needs no second pricer — genuine
single-curve mode is a snapshot with no projections, where the pricer projects off the discount curve
itself. So the two eras are two **snapshot shapes** through one pricer, which is a stronger form of
the arc invariant than two code paths would be: there is no second path to atrophy.

**What that costs is the product**, which this tranche therefore builds (L2), with `Tenor` (L0) and
the `projections` store (L1) as ruled at T54.

**The era test is the pair, and the second half is the load-bearing one.** Both modes price the same
swap; they **agree** where the eras coincide (OIS = projection) and **differ by the basis** where a
real basis exists. A pricer that quietly fell back to one curve passes "agree" and fails "differ" —
which is the T33b guard, since the way a mode rots is by leaving the test population.
"""

import math
from datetime import date

import pytest

from pricebook.engine import price
from pricebook.engine.interest_rate_swap import (
    fixed_leg_value,
    floating_leg_value,
    par_swap_rate,
)
from pricebook.foundation import Currency, Money, PricingFailure, PricingResult
from pricebook.foundation.day_count import DayCountConvention
from pricebook.foundation.schedule import CouponSchedule
from pricebook.foundation.tenor import Tenor
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, MarketSnapshot
from pricebook.market.curve_set import ProjectionKey
from pricebook.market.multi_curve import projection_curve_from_par_swaps
from pricebook.market.nominal_bootstrap import ParSwapQuote, nominal_curve_from_par_swaps
from pricebook.models.discounting_model import DiscountingModel
from pricebook.products.interest_rate_swap import InterestRateSwap, SwapDirection

VAL = date(2026, 1, 1)
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
EUR = Currency("EUR")
SIX_MONTH = Tenor(months=6)
YEARS = (1, 2, 3, 4, 5)

OIS_RATES = {1: 0.0200, 2: 0.0215, 3: 0.0228, 4: 0.0237, 5: 0.0245}
#: Quoted above OIS — a **real basis**, so multi-curve-ness is visible at all (§4: the single-curve
#: case is this tranche's degenerate fixture, not its general one).
TENOR_RATES = {1: 0.0225, 2: 0.0243, 3: 0.0258, 4: 0.0269, 5: 0.0278}

NOTIONAL = Money(10_000_000.0, EUR)


def _schedule(years: int) -> CouponSchedule:
    return CouponSchedule(tuple(date(VAL.year + i, 1, 1) for i in range(years + 1)), DC)


def _quotes(rates: dict[int, float]) -> tuple[ParSwapQuote, ...]:
    return tuple(ParSwapQuote(_schedule(n), rates[n]) for n in YEARS)


def _ois():
    return nominal_curve_from_par_swaps(MEASURE, _quotes(OIS_RATES))


def _projection(discount, rates: dict[int, float] = TENOR_RATES):
    return projection_curve_from_par_swaps(MEASURE, _quotes(rates), discount)


def _multi_curve_market(rates: dict[int, float] = TENOR_RATES) -> MarketSnapshot:
    """The post-crisis snapshot: an OIS discount curve **and** a tenor projection curve."""
    ois = _ois()
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {CurveKey(CurveRole.DISCOUNT, EUR): ois},
            projections={ProjectionKey(EUR, SIX_MONTH): _projection(ois, rates)},
        ),
    )


def _single_curve_market(rates: dict[int, float] = TENOR_RATES) -> MarketSnapshot:
    """The pre-crisis snapshot: **one** curve, which both discounts and projects. No projections
    store at all — the era is what the snapshot holds, not a flag the pricer is passed."""
    return MarketSnapshot(
        valuation_date=VAL,
        curves=CurveSet(
            {CurveKey(CurveRole.DISCOUNT, EUR): nominal_curve_from_par_swaps(MEASURE, _quotes(rates))}
        ),
    )


def _swap(years: int, fixed_rate: float, direction: SwapDirection = SwapDirection.PAYER):
    return InterestRateSwap(
        schedule=_schedule(years),
        fixed_rate=fixed_rate,
        notional=NOTIONAL,
        tenor=SIX_MONTH,
        direction=direction,
    )


def _pv(swap: InterestRateSwap, market: MarketSnapshot) -> float:
    out = price(swap, DiscountingModel(market))
    assert isinstance(out, PricingResult), out
    return out.pv.amount


class TestThePremises:
    """Asserted, because this tranche's degenerate fixture is the era the tree already held."""

    def test_there_is_a_REAL_basis(self) -> None:
        for n in YEARS:
            assert TENOR_RATES[n] > OIS_RATES[n], f"no basis at {n}y"

    def test_the_fixture_has_more_than_ONE_period(self) -> None:
        """One period is the single-curve degeneracy — the discount factor cancels, so the eras
        coincide whatever curve carries it (measured at T54: identical at 1y)."""
        assert len(_schedule(5).periods()) >= 2

    def test_the_two_snapshots_are_genuinely_different_SHAPES(self) -> None:
        assert _multi_curve_market().curves.projections != {}
        assert _single_curve_market().curves.projections == {}


class TestTheCSADiscountingTranscription:
    """Each cashflow discounted at OIS, each floating coupon projected off its tenor curve — the
    legs composed independently here, so the pricer is checked against the test's own arithmetic."""

    @pytest.mark.parametrize("years", YEARS)
    def test_the_floating_leg_projects_at_TENOR_and_discounts_at_OIS(self, years: int) -> None:
        ois = _ois()
        projection = _projection(ois)
        schedule = _schedule(years)
        expected = sum(
            (projection.df(start) / projection.df(pay) - 1.0) / psi * psi * ois.df(pay)
            for start, pay, psi in schedule.periods()
        )
        assert floating_leg_value(schedule, projection, ois, VAL) == pytest.approx(
            expected, rel=1e-14
        )

    @pytest.mark.parametrize("years", YEARS)
    def test_the_fixed_leg_discounts_at_OIS(self, years: int) -> None:
        ois = _ois()
        schedule = _schedule(years)
        expected = 0.03 * sum(psi * ois.df(pay) for _, pay, psi in schedule.periods())
        assert fixed_leg_value(schedule, 0.03, ois, VAL) == pytest.approx(expected, rel=1e-14)

    def test_the_two_legs_are_discounted_on_the_SAME_curve(self) -> None:
        """The CSA statement, as a structural check: both legs discount at OIS, and only the
        projection differs. A pricer discounting the floating leg on the projection curve is the
        single-curve error, and it is the first mutation in the table."""
        ois = _ois()
        projection = _projection(ois)
        schedule = _schedule(5)
        wrong = sum(
            (projection.df(start) / projection.df(pay) - 1.0) * psi * projection.df(pay) / psi * psi
            for start, pay, psi in schedule.periods()
        )
        assert floating_leg_value(schedule, projection, ois, VAL) != pytest.approx(wrong, rel=1e-6)


class TestThePARRequirement:
    """A par collateralised swap is worth zero — derived from the finance, so it survives a
    transcription error the source and the code might share."""

    @pytest.mark.parametrize("years", YEARS)
    def test_a_swap_struck_at_the_MULTI_CURVE_par_rate_is_worth_zero(self, years: int) -> None:
        market = _multi_curve_market()
        ois = _ois()
        schedule = _schedule(years)
        # the par rate computed HERE, from the curves — not read back from the pricer
        projection = _projection(ois)
        floating = sum(
            (projection.df(start) / projection.df(pay) - 1.0) * ois.df(pay)
            for start, pay, _ in schedule.periods()
        )
        annuity = sum(psi * ois.df(pay) for _, pay, psi in schedule.periods())
        independent_par = floating / annuity
        assert _pv(_swap(years, independent_par), market) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("years", YEARS)
    def test_THE_CONSISTENCY_GUARD_the_pricers_own_par_rate_also_prices_to_zero(
        self, years: int
    ) -> None:
        """Kept as a consistency guard and labelled as one: it telescopes, because the par rate and
        the price compose the same legs. The correctness anchor is the independent rate above and
        the reduction below."""
        market = _multi_curve_market()
        rate = par_swap_rate(_schedule(year_count := years), market, EUR, SIX_MONTH)
        assert year_count == years
        assert _pv(_swap(years, rate), market) == pytest.approx(0.0, abs=1e-6)

    def test_the_par_rate_sits_between_the_ois_and_tenor_quotes(self) -> None:
        """A sanity bound with real content: the multi-curve par rate of the 5y swap must be the
        tenor quote it was calibrated from, since that is what the bootstrap inverted."""
        market = _multi_curve_market()
        assert par_swap_rate(_schedule(5), market, EUR, SIX_MONTH) == pytest.approx(
            TENOR_RATES[5], rel=1e-9
        )


class TestThePreCrisisReduction:
    """The exact anchor: with OIS = projection the CSA price is the single-curve price."""

    @pytest.mark.parametrize("years", YEARS)
    @pytest.mark.parametrize("fixed", [0.02, 0.0245, 0.03])
    def test_ois_equal_projection_reproduces_the_single_curve_price(
        self, years: int, fixed: float
    ) -> None:
        """Asserted as a **relative tolerance**: the two routes are one quantity in ℝ reached by
        different operation orders (the projection recursion against a single curve's own forwards),
        so §4's op-order condition fails and bit-identity is unavailable — the T54/T49 discipline."""
        collapsed = _multi_curve_market(OIS_RATES)
        single = _single_curve_market(OIS_RATES)
        assert _pv(_swap(years, fixed), collapsed) == pytest.approx(
            _pv(_swap(years, fixed), single), rel=1e-11
        )


class TestTheSingleCurveInvariant:
    """**The arc invariant's first real test** — and the pair matters, not either half.

    `agree` proves the eras *coincide* at the boundary. `differ` proves both are *reachable and
    exercised*: a pricer that quietly fell back to one curve would pass `agree` and fail `differ`.
    That is the T33b guard — a mode rots by leaving the test population, silently."""

    @pytest.mark.parametrize("years", YEARS)
    def test_the_PRE_CRISIS_mode_prices_with_no_projection_curve_at_all(self, years: int) -> None:
        """Genuine single-curve mode: the snapshot holds one curve and no projections store, and
        the pricer projects off the discount curve itself."""
        market = _single_curve_market()
        assert market.curves.projections == {}
        pv = _pv(_swap(years, 0.025), market)
        assert math.isfinite(pv)

    @pytest.mark.parametrize("years", YEARS)
    def test_AGREE_where_the_eras_coincide(self, years: int) -> None:
        assert _pv(_swap(years, 0.025), _multi_curve_market(OIS_RATES)) == pytest.approx(
            _pv(_swap(years, 0.025), _single_curve_market(OIS_RATES)), rel=1e-11
        )

    def test_DIFFER_by_the_basis_where_a_real_basis_exists(self) -> None:
        """The load-bearing half. Measured at T54: the inter-era gap is **0 at one period** (the df
        cancels) and grows with maturity, so this asserts the shape rather than a single number —
        and asserting it at the short end would have been vacuous."""
        near = abs(
            _pv(_swap(1, 0.025), _multi_curve_market()) - _pv(_swap(1, 0.025), _single_curve_market())
        )
        far = abs(
            _pv(_swap(5, 0.025), _multi_curve_market()) - _pv(_swap(5, 0.025), _single_curve_market())
        )
        assert near < 1.0, f"the one-period swap must coincide across eras, differs by {near}"
        assert far > 100.0, f"the 5y swap must differ across eras, differs by only {far}"
        assert far > 100.0 * max(near, 1e-9)


class TestTheRegisteredPricer:
    def test_a_payer_and_a_receiver_are_exact_negatives(self) -> None:
        market = _multi_curve_market()
        payer = _pv(_swap(5, 0.025, SwapDirection.PAYER), market)
        receiver = _pv(_swap(5, 0.025, SwapDirection.RECEIVER), market)
        assert payer == pytest.approx(-receiver, rel=1e-14)
        assert abs(payer) > 1.0, "the premise: a swap away from par, or both sides are zero"

    def test_the_payer_GAINS_when_rates_are_above_its_fixed_rate(self) -> None:
        """Direction pinned by the finance, not by convention: a payer of a low fixed rate against
        a higher floating leg is in the money."""
        market = _multi_curve_market()
        assert _pv(_swap(5, 0.01, SwapDirection.PAYER), market) > 0.0
        assert _pv(_swap(5, 0.10, SwapDirection.PAYER), market) < 0.0

    def test_an_EXPIRED_swap_is_PV_zero_invariant_6(self) -> None:
        """Every period settled, so there is nothing left to value. The fixture is struck far from
        par so the fall-through would be a large number, not a coincidental zero (the tenth-shape
        premise, with F-4.1's sharpening)."""
        market = _multi_curve_market()
        matured = InterestRateSwap(
            schedule=CouponSchedule((date(2024, 1, 1), date(2025, 1, 1)), DC),
            fixed_rate=0.0,
            notional=NOTIONAL,
            tenor=SIX_MONTH,
            direction=SwapDirection.PAYER,
        )
        out = price(matured, DiscountingModel(market))
        assert isinstance(out, PricingResult), out
        assert out.pv.amount == 0.0

    def test_a_STRADDLING_swap_excludes_only_its_settled_periods(self) -> None:
        """The tenth-shape premise in its sharpened form: the schedule **straddles** the valuation
        date, so the invariant-6 filter has something to filter and a dropped filter would discount
        a settled flow at negative `t`."""
        market = _multi_curve_market()
        straddling = InterestRateSwap(
            schedule=CouponSchedule(
                tuple(date(VAL.year + i, 1, 1) for i in range(-2, 4)), DC
            ),
            fixed_rate=0.025,
            notional=NOTIONAL,
            tenor=SIX_MONTH,
            direction=SwapDirection.PAYER,
        )
        live_only = _swap(3, 0.025)
        assert _pv(straddling, market) == pytest.approx(_pv(live_only, market), rel=1e-12)

    def test_a_missing_discount_curve_is_a_FAILURE_not_an_exception(self) -> None:
        bare = MarketSnapshot(valuation_date=VAL, curves=CurveSet({}))
        assert isinstance(price(_swap(5, 0.025), DiscountingModel(bare)), PricingFailure)


class TestTheTenorAndTheProjectionsStore:
    """The new vocabulary, introduced where consumed (§6) — and keyed on its own dimension rather
    than by widening `CurveKey`, which would hang a `None` tenor on every DISCOUNT/REAL key."""

    def test_a_tenor_is_hashable_and_compares_by_value(self) -> None:
        assert Tenor(months=6) == Tenor(months=6)
        assert Tenor(months=3) != Tenor(months=6)
        assert len({Tenor(months=3), Tenor(months=6), Tenor(months=6)}) == 2

    @pytest.mark.parametrize("months", [0, -3])
    def test_a_non_positive_tenor_is_refused(self, months: int) -> None:
        with pytest.raises(ValueError, match="positive"):
            Tenor(months=months)

    def test_a_missing_projection_curve_RAISES_rather_than_defaulting(self) -> None:
        """The `CurveSet` house style: a silent fallback to the discount curve would price every
        collateralised swap as if there were no basis — a plausible wrong number."""
        curves = _multi_curve_market().curves
        with pytest.raises(KeyError, match="projection"):
            curves.projection(EUR, Tenor(months=3))

    def test_the_projections_store_does_not_disturb_the_existing_keys(self) -> None:
        """The reason it is a separate store: every existing `CurveSet` construction still works
        and no existing key grew a field."""
        curves = _multi_curve_market().curves
        assert curves.discount(EUR) is not None
        assert CurveKey(CurveRole.DISCOUNT, EUR) in curves.curves
