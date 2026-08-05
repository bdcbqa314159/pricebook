"""L4 oracle — HAGAN_WEST as a curve interpolation mode, end to end.

A dual curve built with HAGAN_WEST (via the simultaneous solve — HW is non-local, so the
sequential bootstrap is out of scope this slice) reprices every calibrating instrument to
zero through the L4 engine; and where HW and log-linear agree (constant forwards) the df's
match. The point-based `interpolate()` rejects HAGAN_WEST (it is a curve mode, not a point scheme).
"""

import math
from dataclasses import replace
from datetime import date

import pytest

from pricebook_ng.calibration.calibrate import (
    CalibrationMethod,
    CalibrationSpec,
    CurveBuild,
    ParSwapQuote,
    SolveConfig,
    calibrate,
    product,
)
from pricebook_ng.engine import price
from pricebook_ng.foundation import (
    Currency,
    DayCountConvention,
    Frequency,
    Interpolation,
    PricingResult,
    Tenor,
    TenorUnit,
    TimeMeasure,
    get_rate_index,
    interpolate,
)
from pricebook_ng.market.curve import DiscountCurve

VAL = date(2026, 1, 15)
ESTR = get_rate_index("ESTR")
EURIBOR_3M = get_rate_index("EURIBOR_3M")
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
HW = Interpolation.HAGAN_WEST

_OIS = tuple(ParSwapQuote(Tenor(y, Y), r) for y, r in [(1, 0.030), (2, 0.032), (3, 0.034), (5, 0.036)])
_IBOR = tuple(ParSwapQuote(Tenor(y, Y), r) for y, r in [(1, 0.0312), (2, 0.0332), (3, 0.0352), (5, 0.0372)])
DISCOUNT = CurveBuild(ESTR, Frequency.ANNUAL, DC, _OIS, interpolation=HW)
PROJECTION = CurveBuild(EURIBOR_3M, Frequency.ANNUAL, DC, _IBOR, interpolation=HW)
GLOBAL_HW = CalibrationSpec(
    valuation_date=VAL, currency=Currency.EUR, discount=DISCOUNT, projection=PROJECTION,
    solve=SolveConfig(method=CalibrationMethod.SIMULTANEOUS),
)


def test_hagan_west_curve_reprices_to_par_through_engine() -> None:
    result = calibrate(GLOBAL_HW)
    model, calib = result  # simultaneous converges for this well-posed set
    assert calib.converged
    assert model.market.curves.discount(Currency.EUR).interpolation is HW
    for build in (DISCOUNT, PROJECTION):
        for quote in build.quotes:
            priced = price(product(GLOBAL_HW, build, quote), model)
            assert isinstance(priced, PricingResult)
            assert abs(priced.pv.amount) < 1e-9


def test_hagan_west_reproduces_pillar_discount_factors() -> None:
    model, _ = calibrate(GLOBAL_HW)
    curve = model.market.curves.discount(Currency.EUR)
    # the HW integral to each pillar time equals −ln(df), so exp(−∫) round-trips the pillar df
    for t, df in zip(curve.times[1:], curve.dfs[1:]):
        assert abs(math.exp(-curve._forward_reconstruction().integral(t)) - df) < 1e-12


def test_degeneracy_hagan_west_equals_log_linear_on_flat_forwards() -> None:
    tm = TimeMeasure(VAL, DC)
    times = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    dfs = tuple(math.exp(-0.03 * t) for t in times)  # constant continuously-compounded forward
    hw = DiscountCurve(tm, times, dfs, HW)
    ll = DiscountCurve(tm, times, dfs, Interpolation.LOG_LINEAR)
    mid = VAL + Tenor(30, TenorUnit.MONTH)
    assert abs(hw.df(mid) - ll.df(mid)) < 1e-12


def test_point_interpolate_rejects_hagan_west() -> None:
    with pytest.raises(NotImplementedError):
        interpolate((0.0, 1.0), (1.0, 0.97), 0.5, HW)
