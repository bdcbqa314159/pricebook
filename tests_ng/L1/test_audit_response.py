"""L1 audit-response oracles (slice 6d) — finding #2 (Hagan–West past-last-pillar) and
finding #8 (HW reconstruction caching). Each test EXPOSES the finding on v0.92.0.
"""

from datetime import date

import pytest

from pricebook_ng.foundation import (
    DayCountConvention,
    Interpolation,
    Tenor,
    TenorUnit,
    TimeMeasure,
    monotone_convex,
)
from pricebook_ng.market.curve import DiscountCurve

VAL = date(2026, 1, 15)
DC = DayCountConvention.ACT_365_FIXED
Y = TenorUnit.YEAR
TM = TimeMeasure(VAL, DC)
_TIMES = (0.0, 1.0, 2.0, 3.0, 5.0)
_DFS = (1.0, 0.97, 0.94, 0.90, 0.83)
HW = DiscountCurve(TM, _TIMES, _DFS, Interpolation.HAGAN_WEST)
LL = DiscountCurve(TM, _TIMES, _DFS, Interpolation.LOG_LINEAR)


def test_hw_df_raises_past_last_pillar_like_log_linear() -> None:
    past = VAL + Tenor(6, Y)  # year-fraction ~6 > last pillar 5.0
    with pytest.raises(ValueError):
        LL.df(past)  # log-linear already raises (the ratified RAISE policy) — sanity
    with pytest.raises(ValueError):
        HW.df(past)  # finding #2: HW currently returns df[-1] silently


def test_monotone_convex_integral_raises_past_last_knot() -> None:
    mc = monotone_convex((0.0, 1.0, 2.0), (0.03, 0.035))
    with pytest.raises(ValueError):
        mc.integral(2.5)  # past knots[-1]=2.0 — currently truncates silently


def test_monotone_convex_value_integral_agree_in_range() -> None:
    mc = monotone_convex((0.0, 1.0, 2.0, 3.0, 5.0), (0.030, 0.032, 0.035, 0.038))
    h = 1e-7
    for x in (0.5, 1.5, 2.7, 4.0):
        fd = (mc.integral(x + h) - mc.integral(x)) / h  # d/dx ∫ = value
        assert abs(fd - mc.value(x)) < 1e-5
