"""Seasoned valuation -- pricing at `t > 0`. BERHAN-2012 eq 2.5.12.

    YYIIS(t,T,Psi,N) = N*psi_i(t) [ I(t)/I(T_i(t)-1) * Pr(t,T_i(t)) - Pn(t,T_i(t)) ]
                     + N * SUM_{i=i(t)+1} psi_i [ Pn(t,T_i-1)*Pr(t,T_i)/Pr(t,T_i-1)*e^C - Pn(t,T_i) ]

with `iota(t) = min{i : T_i > t}`. Transcribed from the PDF in **both `-raw` and `-layout`**, which
agree (charter section 4).

**The defect this tranche closes, measured before it was fixed.** eq 2.5.13 is 2.5.12 at
`I(0)/I(T_0) = 1`, and the tranche 4 engine implemented 2.5.13 *unconditionally* -- so at any
seasoned valuation date it silently returned the ratio-equals-one answer, never asking for the
published reset `I(T_iota-1)` and never refusing either. On a 3y annual swap valued 18 months in,
with inflation realised at 2%/yr since the reset, that is **98.7 bp of notional, 34.9% of the
inflation leg** (196.5 bp at 4%). A wrong number, not a missing feature -- which is why
`test_the_current_period_is_NOT_priced_as_if_the_index_had_not_moved` is the companion that matters.
"""

import math
from datetime import date, timedelta

import pytest

from pricebook.engine import price
from pricebook.engine.yoy import yoy_inflation_leg_value
from pricebook.foundation import (
    CouponSchedule,
    Currency,
    DayCountConvention,
    IndexInterpolation,
    InflationIndex,
    Money,
    PricingFailure,
    PricingResult,
    YoYLeg,
)
from pricebook.market import (
    CurveKey,
    CurveRole,
    CurveSet,
    DiscountCurve,
    FixingSeries,
    MarketSnapshot,
    reference_level,
)
from pricebook.models import (
    DiscountingModel,
    HullWhiteParams,
    JarrowYildirim,
    JYCorrelations,
    JYParams,
    yoy_convexity,
)
from pricebook.products import YearOnYearInflationSwap, ZeroCouponInflationSwap
from pricebook.foundation.term import Term

EUR = Currency("EUR")
DC = DayCountConvention.ACT_365_FIXED
N = 1_000_000.0
START = date(2026, 1, 15)
DATES = tuple(START + timedelta(days=365 * k) for k in range(4))   # 3 annual periods
NOMINAL_KEY = CurveKey(CurveRole.DISCOUNT, EUR)
REAL_KEY = CurveKey(CurveRole.REAL, EUR)
PARAMS = JYParams(
    HullWhiteParams(0.05, 0.010), HullWhiteParams(0.04, 0.008), 0.012,
    JYCorrelations(nominal_real=0.30, real_index=-0.20, nominal_index=0.15),
)

# Zero lag and FLAT, so the oracles below isolate the SEASONING from the resolution convention --
# resolution has its own oracles in tests/L1/test_fixing_series.py. A sloped monthly series: the
# index compounds at 2%/yr from 100.0 at 2026-01.
HICP = InflationIndex("HICPxt", publication_lag=0, interpolation=IndexInterpolation.FLAT)


def _series(annual_rate: float = 0.02, base: float = 100.0) -> FixingSeries:
    prints = {}
    for k in range(48):
        y, m = divmod(2026 * 12 + k, 12)
        prints[date(y, m + 1, 1)] = base * (1.0 + annual_rate) ** (k / 12.0)
    return FixingSeries(prints)


FLAT_INDEX = _series(0.0)          # no realised inflation: every print is 100.0
RISING = _series(0.02)


def _model(vd: date, series: FixingSeries, nominal=0.03, real=0.01) -> JarrowYildirim:
    return JarrowYildirim(
        MarketSnapshot(
            valuation_date=vd,
            curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(vd, nominal, DC),
                             REAL_KEY: DiscountCurve.flat(vd, real, DC)}),
            fixings={HICP: series},
        ),
        PARAMS,
    )


def _swap(fixed_rate: float = 0.02) -> YearOnYearInflationSwap:
    return YearOnYearInflationSwap(HICP, YoYLeg(CouponSchedule(DATES, DC), fixed_rate),
                                   Money(N, EUR))


# ============================================================================================
# ORACLE 1 -- the anchor. Seasoned reduces to inception EXACTLY, by construction.
# ============================================================================================

def test_at_inception_the_seasoned_form_IS_the_tranche_4_inception_form():
    """`iota(t) = 1` and `I(t) = I(T_0)` resolve to the same print, so the ratio is exactly 1 and
    2.5.12 collapses to 2.5.13. Not a numerical limit -- the same division, giving 1.0.

    The reference value is eq 2.5.13 transcribed here, independent of the engine."""
    m = _model(START, RISING)
    nominal, real = m.market.curves.discount(EUR), m.market.curves.real(EUR)
    measure = nominal.time_measure
    periods = list(CouponSchedule(DATES, DC).periods())

    _, first_pay, first_psi = periods[0]
    expected = first_psi * (real.df(first_pay) - nominal.df(first_pay))
    for start, pay, psi in periods[1:]:
        c = yoy_convexity(0.0, measure.year_fraction(start), measure.year_fraction(pay), PARAMS)
        expected += psi * (nominal.df(start) * real.df(pay) / real.df(start) * math.exp(c)
                           - nominal.df(pay))
    assert yoy_inflation_leg_value(_swap(), m) == pytest.approx(expected * N, rel=1e-14)


def test_a_one_point_series_reproduces_the_hand_fed_scalar_for_the_zciis():
    """The other half of the reduction: a ZCIIS resolving `I(t)` from a one-point series equals
    `N[I(t)/I0*Pr - Pn]` with that same `I(t)`, which is what tranche 2 was given directly."""
    series = FixingSeries({date(2026, 1, 1): 105.0})
    m = _model(START, series)
    swap = ZeroCouponInflationSwap(HICP, 100.0, 0.02, Term(START, DATES[2], DC), Money(N, EUR))
    nominal, real = m.market.curves.discount(EUR), m.market.curves.real(EUR)
    expected_leg = N * (105.0 / 100.0 * real.df(DATES[2]) - nominal.df(DATES[2]))
    fixed = N * (1.02 ** Term(START, DATES[2], DC).years() - 1.0) * nominal.df(DATES[2])
    out = price(swap, DiscountingModel(m.market))
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == pytest.approx(expected_leg - fixed, rel=1e-14)


# ============================================================================================
# ORACLE 2 -- the mandatory companion. The seasoned path is NOT a no-op.
# ============================================================================================

def test_the_current_period_is_NOT_priced_as_if_the_index_had_not_moved():
    """**The defect, as a test.** Value the same swap 18 months in against two series: one where
    the index never moved since the reset, one where it rose 2%/yr. The old engine returned the
    same number for both, because it never read a print. They must now differ -- and by the
    magnitude the ratio implies, not merely 'differ'."""
    vd = START + timedelta(days=547)
    still = yoy_inflation_leg_value(_swap(), _model(vd, FLAT_INDEX))
    risen = yoy_inflation_leg_value(_swap(), _model(vd, RISING))

    m = _model(vd, RISING)
    reset_date = DATES[1]                      # T_iota-1, the start of the in-progress period
    ratio = (reference_level(RISING, vd, HICP)
             / reference_level(RISING, reset_date, HICP))
    real = m.market.curves.real(EUR)
    psi = [p for p in CouponSchedule(DATES, DC).periods() if p[1] > vd][0][2]
    expected_gap = N * psi * real.df(DATES[2]) * (ratio - 1.0)

    assert ratio > 1.0, "the rising fixture did not actually rise -- the test has no power"
    assert risen - still == pytest.approx(expected_gap, rel=1e-12)
    assert abs(risen - still) > 1e-3 * N, "the seasoning moves the leg by less than 10bp"


def test_the_seasoned_ratio_moves_the_PRICE_not_only_the_leg():
    vd = START + timedelta(days=547)
    a = price(_swap(), _model(vd, FLAT_INDEX))
    b = price(_swap(), _model(vd, RISING))
    assert isinstance(a, PricingResult) and isinstance(b, PricingResult)
    assert a.pv.amount != b.pv.amount


# ============================================================================================
# ORACLE 5 -- the current period uses the HISTORICAL reset, not a forward.
# ============================================================================================

def test_the_in_progress_period_reads_the_PUBLISHED_reset_not_a_curve_implied_one():
    """Where 2.5.12 differs from 2.5.13: `I(T_iota-1)` is a print that happened, so perturbing the
    series at the reset month must move the price, while the curves stay fixed."""
    vd = START + timedelta(days=547)
    base = yoy_inflation_leg_value(_swap(), _model(vd, RISING))

    reset_month = date(DATES[1].year, DATES[1].month, 1)
    bumped = FixingSeries({**RISING.prints,
                           reset_month: RISING.prints[reset_month] * 1.01})
    moved = yoy_inflation_leg_value(_swap(), _model(vd, bumped))
    assert moved != base
    # A HIGHER reset means a LOWER year-on-year ratio, so the leg must fall.
    assert moved < base


def test_a_FUTURE_period_does_not_read_the_series_at_all():
    """The converse, and it guards against resolution leaking into the forward periods: bump a
    print in a month that no remaining reset falls in, and nothing moves."""
    vd = START + timedelta(days=547)
    base = yoy_inflation_leg_value(_swap(), _model(vd, RISING))
    far_future = date(2029, 6, 1)
    bumped = FixingSeries({**RISING.prints, far_future: RISING.prints[far_future] * 1.50})
    assert yoy_inflation_leg_value(_swap(), _model(vd, bumped)) == pytest.approx(base, rel=1e-15)


# ============================================================================================
# ORACLE 6 -- invariant 6 across a seasoned schedule: THREE cases, not two.
# ============================================================================================

def test_the_schedule_partitions_into_elapsed_in_progress_and_future():
    """At 18 months: period 1 fully elapsed (excluded), period 2 in progress (included, with its
    historical reset), period 3 future (included, with convexity). Asserted as a count so a
    mis-partition cannot hide behind a plausible total."""
    vd = START + timedelta(days=547)
    remaining = [p for p in CouponSchedule(DATES, DC).periods() if p[1] > vd]
    assert len(remaining) == 2
    assert remaining[0][0] < vd < remaining[0][1], "the first remaining period is in progress"
    assert remaining[1][0] > vd, "the second remaining period has not started"


def test_a_fully_elapsed_swap_is_PV_zero_and_needs_no_fixings():
    """Invariant 6 unchanged by seasoning: past the last payment the trade is settled, and that is
    decided before any market or series lookup."""
    vd = DATES[-1] + timedelta(days=1)
    m = JarrowYildirim(
        MarketSnapshot(valuation_date=vd,
                       curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(vd, 0.03, DC),
                                        REAL_KEY: DiscountCurve.flat(vd, 0.01, DC)})),
        PARAMS,
    )
    out = price(_swap(), m)
    assert isinstance(out, PricingResult), out
    assert out.pv.amount == 0.0


# ============================================================================================
# ORACLE 7 -- failure is a value.
# ============================================================================================

def test_a_missing_series_is_a_FAILURE_not_a_guessed_ratio():
    """The regression that matters most: with no fixings at all the engine must refuse. Before
    this tranche it priced happily, assuming the ratio was 1."""
    vd = START + timedelta(days=547)
    m = JarrowYildirim(
        MarketSnapshot(valuation_date=vd,
                       curves=CurveSet({NOMINAL_KEY: DiscountCurve.flat(vd, 0.03, DC),
                                        REAL_KEY: DiscountCurve.flat(vd, 0.01, DC)})),
        PARAMS,
    )
    out = price(_swap(), m)
    assert isinstance(out, PricingFailure), out
    assert "fixing" in out.reason or "print" in out.reason


def test_a_series_that_starts_after_the_reset_is_a_FAILURE():
    """The reset print is missing, so the ratio is unknowable. Refusing beats extrapolating."""
    vd = START + timedelta(days=547)
    late = FixingSeries({date(2027, 8, 1): 115.0, date(2027, 9, 1): 115.2})
    out = price(_swap(), _model(vd, late))
    assert isinstance(out, PricingFailure), out
