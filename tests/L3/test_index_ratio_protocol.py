"""The model capability seam: `IndexRatioModel`. Tranche 12(a).

The year-on-year and cap/floor pricers ask a model exactly two questions about a period — `m`, the
`T_i`-forward expectation of `I(T_i)/I(T_{i-1})` (BERHAN eq 2.6.2), and `V^2`, the variance of its
log (eq 2.6.4). This protocol is that pair, so a second model is served by implementing it rather
than by a branch inside a pricer.

**What these oracles are for, and what they are NOT.** They do not re-check eq 2.5.11 or 2.6.4 --
tranches 4 and 5 oracled those against the PDF independently and still do. They check the three
things the seam itself can get wrong:

  1. the date-based capability agrees with the year-fraction form it delegates to (no clock drift);
  2. `m` is the same quantity the cap/floor's `d1/d2` were built on, so the refactor did not
     silently change what the pricers read;
  3. the pricers really depend on the PROTOCOL and not on `JarrowYildirim` -- proved by passing a
     stub model that implements only the two methods, which is a test a type annotation cannot be.

The `isinstance` check is included but is deliberately NOT the evidence: a `runtime_checkable`
protocol checks that the NAMES exist, never the signatures, and tranche 5 learned the hard way that
a runtime probe can report a protocol satisfied while pyright -- correctly -- disagrees.
"""

import math
from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.capfloor import caplet_value
from pricebook.engine.yoy import yoy_inflation_leg_value
from pricebook.foundation import (
    CapFloorKind,
    CouponSchedule,
    Currency,
    DayCountConvention,
    IndexInterpolation,
    InflationIndex,
    Money,
    PricingResult,
    YoYLeg,
)
from pricebook.foundation.time_measure import TimeMeasure
from pricebook.market import CurveKey, CurveRole, CurveSet, DiscountCurve, MarketSnapshot
from pricebook.models import HullWhiteParams, JarrowYildirim, JYCorrelations, JYParams
from pricebook.models.index_ratio import IndexRatioModel
from pricebook.products import InflationCapFloor, YearOnYearInflationSwap

VAL = date(2026, 1, 15)
EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
MEASURE = TimeMeasure(VAL, DC)
HICP = InflationIndex("HICPxt", publication_lag=3, interpolation=IndexInterpolation.LINEAR)
N = 1_000_000.0
NOMINAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.026, 0.030, 0.034, 0.038))
REAL = DiscountCurve(MEASURE, (0.0, 1.0, 2.0, 5.0), (0.008, 0.011, 0.013, 0.016))
MARKET = MarketSnapshot(
    valuation_date=VAL,
    curves=CurveSet({CurveKey(CurveRole.DISCOUNT, EUR): NOMINAL,
                     CurveKey(CurveRole.REAL, EUR): REAL}),
)
PARAMS = JYParams(HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.008), 0.012,
                  JYCorrelations(0.30, -0.20, 0.15))
JY = JarrowYildirim(MARKET, PARAMS)
DATES = tuple(VAL + timedelta(days=365 * k) for k in range(5))


def test_jarrow_yildirim_satisfies_the_protocol_at_runtime():
    """A cheap extra, not the evidence — see the module docstring."""
    assert isinstance(JY, IndexRatioModel)


def test_the_date_capability_agrees_with_the_year_fraction_form():
    """No clock drift across the seam: the date-based methods must return exactly what the
    year-fraction methods they delegate to return."""
    for k in range(1, 4):
        start, pay = DATES[k], DATES[k + 1]
        t0, t1 = MEASURE.year_fraction(start), MEASURE.year_fraction(pay)
        expected_ratio = (
            (NOMINAL.df(start) / NOMINAL.df(pay))
            * (REAL.df(pay) / REAL.df(start))
            * math.exp(JY.convexity(0.0, t0, t1))
        )
        assert JY.expected_index_ratio(start, pay) == expected_ratio
        assert JY.index_ratio_variance(start, pay) == JY.log_ratio_variance(0.0, t0, t1)


def test_the_forward_ratio_IS_the_quantity_the_capfloor_prices_on():
    """`m` is eq 2.6.2 — transcribed here from the formula, so the seam exposes the quantity the
    source names rather than whatever the old inline expression happened to compute."""
    start, pay = DATES[2], DATES[3]
    pn0, pn1 = NOMINAL.df(start), NOMINAL.df(pay)
    pr0, pr1 = REAL.df(start), REAL.df(pay)
    c = JY.convexity(0.0, MEASURE.year_fraction(start), MEASURE.year_fraction(pay))
    assert JY.expected_index_ratio(start, pay) == pytest.approx(
        (pn0 / pn1) * (pr1 / pr0) * math.exp(c), rel=1e-15
    )


def test_m_exceeds_one_on_an_inflationary_pair_ONLY_once_the_convexity_is_removed():
    """**My first version of this asserted `m > 1` outright, and it was wrong.** With the nominal
    curve 2.1% above the real one the carry does push `m` up — but `C` can be negative enough to
    overcome it: at 3y->4y, `m = 0.9914`. The convexity is the whole point of the model, and a
    requirement that ignores it is not a requirement.

    The honest statement separates the two drivers: at `sigma_r = 0` the convexity vanishes exactly
    (it is `C`'s overall factor, eq 2.5.11) and then `m > 1` IS forced by the curve pair. So this
    asserts the carry alone, and that the convexity is what moves it -- which is a stronger oracle
    than the one it replaces, because it pins the sign of `C`'s contribution too."""
    no_real_vol = JarrowYildirim(
        MARKET,
        JYParams(PARAMS.nominal, HullWhiteParams(PARAMS.real.kappa, 0.0), PARAMS.index_vol,
                 PARAMS.correlations),
    )
    for k in range(1, 4):
        start, pay = DATES[k], DATES[k + 1]
        carry_only = no_real_vol.expected_index_ratio(start, pay)
        assert carry_only > 1.0, f"the carry alone should grow the ratio at {k}y->{k + 1}y"
        assert no_real_vol.convexity(
            0.0, MEASURE.year_fraction(start), MEASURE.year_fraction(pay)
        ) == 0.0
        # And the convexity moves it -- downward on this parameter set, enough to cross 1.0 at the
        # far end, which is why the naive `m > 1` failed there.
        assert JY.expected_index_ratio(start, pay) < carry_only
    assert JY.expected_index_ratio(DATES[3], DATES[4]) < 1.0 < JY.expected_index_ratio(
        DATES[1], DATES[2]
    )


# ---- the seam itself: a stub model the pricers must accept ----

@dataclass(frozen=True)
class _StubRatioModel:
    """A model that is NOT a `JarrowYildirim` and implements only the two protocol methods.

    This is the oracle a type annotation cannot be: if either pricer still reached for
    `model.convexity`, `model.params` or anything else JY-specific, pricing with this would raise."""

    market: MarketSnapshot
    ratio: float
    variance: float

    def expected_index_ratio(self, period_start: date, period_pay: date) -> float:
        return self.ratio

    def index_ratio_variance(self, period_start: date, period_pay: date) -> float:
        return self.variance


STUB = _StubRatioModel(MARKET, ratio=1.025, variance=0.0004)


def test_the_stub_satisfies_the_protocol():
    assert isinstance(STUB, IndexRatioModel)


def test_the_CAPFLOOR_pricer_depends_on_the_protocol_not_on_JY():
    product = InflationCapFloor(HICP, CouponSchedule(DATES, DC), 0.02, Money(N, EUR),
                                CapFloorKind.FLOOR)
    out = price(product, STUB)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount > 0.0


def test_the_YOY_pricer_depends_on_the_protocol_not_on_JY():
    """The year-on-year leg needs the index *fixings* only for its seasoned first period; at
    inception it is curves plus `m`, so a stub suffices."""
    swap = YearOnYearInflationSwap(HICP, YoYLeg(CouponSchedule(DATES, DC), 0.02), Money(N, EUR))
    assert yoy_inflation_leg_value(swap, STUB) != 0.0


def test_the_capflooor_price_is_EXACTLY_black_on_the_stubs_two_numbers():
    """With the model's two answers pinned, the caplet reduces to Black on them — which proves the
    pricer uses *only* `m` and `V^2` from the model, and nothing else."""
    from pricebook.models.black import black_price

    product = InflationCapFloor(HICP, CouponSchedule(DATES, DC), 0.02, Money(N, EUR),
                                CapFloorKind.FLOOR)
    expected = 0.0
    for start, pay, accrual in CouponSchedule(DATES, DC).periods():
        expected += N * accrual * NOMINAL.df(pay) * black_price(
            STUB.ratio, 1.02, STUB.variance, CapFloorKind.FLOOR
        )
    assert caplet_value(product, STUB, next(iter(CouponSchedule(DATES, DC).periods()))) > 0.0
    out = price(product, STUB)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == pytest.approx(expected, rel=1e-15)


def test_a_model_MISSING_a_protocol_method_is_not_accepted():
    """The negative case, so the protocol check is known to discriminate."""

    @dataclass(frozen=True)
    class _Incomplete:
        market: MarketSnapshot

        def expected_index_ratio(self, period_start: date, period_pay: date) -> float:
            return 1.0

    assert not isinstance(_Incomplete(MARKET), IndexRatioModel)
